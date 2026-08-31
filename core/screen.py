"""
core/screen.py
==============
Screen a primary satellite against the full catalog for close approaches.

Pipeline:
1. Apogee/perigee pre-filter (altitude band culling, cheap, no SGP4)
2. Coarse SGP4 propagation at 60s steps using SatrecArray (vectorised)
   - SGP4 output is in km and km/s. Convert to SI (m, m/s) IMMEDIATELY.
   - Error codes from SatrecArray.sgp4 are checked and failing sats masked.
3. Keep approaches with range < coarse_range_m (50 km default)
4. Refine each coarse minimum to 1s resolution with golden-section search
5. Return list of ConjunctionEvent dataclasses

Frame: ALL positions in TEME. sgp4 library returns TEME directly.
Units: ALL internal values in SI (metres, m/s, seconds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sgp4.api import Satrec, SatrecArray, jday


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConjunctionEvent:
    """A screened conjunction event, all quantities in SI units."""
    primary_norad: int
    secondary_norad: int
    secondary_name: str

    # Time of closest approach
    tca: datetime               # UTC

    # Miss distance (metres, always positive)
    miss_m: float

    # Relative velocity magnitude (m/s)
    rel_v_ms: float

    # Relative state at TCA in TEME (metres, m/s)
    # r_rel = r_secondary - r_primary
    r_rel_teme_m: np.ndarray    # shape (3,)
    v_rel_teme_ms: np.ndarray   # shape (3,)

    # Position vectors at TCA in TEME (metres)
    r_primary_teme_m: np.ndarray   # shape (3,)
    r_secondary_teme_m: np.ndarray # shape (3,)

    # Time since TLE epoch at TCA (seconds) — used for covariance growth
    primary_epoch_offset_s: float
    secondary_epoch_offset_s: float

    # TLE epoch of each object
    primary_epoch: datetime
    secondary_epoch: datetime


# ---------------------------------------------------------------------------
# Helper: build Satrec from TLE lines
# ---------------------------------------------------------------------------

def _satrec_from_tle(line1: str, line2: str) -> Satrec:
    sat = Satrec.twoline2rv(line1, line2)
    return sat


# ---------------------------------------------------------------------------
# Helper: Julian date split from Python datetime
# ---------------------------------------------------------------------------

def _jd_split(dt: datetime) -> tuple[float, float]:
    """Return (jd_integer_day, fraction_of_day) for a UTC datetime."""
    # jday(year, mon, day, hr, minute, sec)
    return jday(dt.year, dt.month, dt.day,
                dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)


# ---------------------------------------------------------------------------
# Helper: propagate a single Satrec to a datetime
# Returns (r_m, v_ms) in TEME, or (None, None) on error
# ---------------------------------------------------------------------------

def _propagate_one(sat: Satrec, dt: datetime) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    jd, fr = _jd_split(dt)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    # sgp4 returns km and km/s — convert to SI immediately
    r_m = np.array(r) * 1000.0
    v_ms = np.array(v) * 1000.0
    return r_m, v_ms


# ---------------------------------------------------------------------------
# Step 1: Apogee/perigee pre-filter
# ---------------------------------------------------------------------------

_RE_M = 6_378_137.0  # WGS84 equatorial radius (metres)


def _tle_apogee_perigee_m(sat: Satrec) -> tuple[float, float]:
    """
    Extract apogee and perigee from a Satrec object.
    sgp4 stores mean motion in rad/min, eccentricity and semi-major axis.
    Returns (apogee_m, perigee_m) above Earth's surface.
    """
    # mean motion in rad/min -> rad/s -> semi-major axis via Kepler's 3rd law
    mu = 3.986004418e14      # m³/s²
    n_rad_s = sat.no_kozai / 60.0  # rad/min → rad/s
    a_m = (mu / (n_rad_s ** 2)) ** (1.0 / 3.0)  # metres (geocentric)
    e = sat.ecco
    r_a = a_m * (1 + e) - _RE_M   # altitude at apogee
    r_p = a_m * (1 - e) - _RE_M   # altitude at perigee
    return r_a, r_p


def _apogee_perigee_filter(
    primary_sat: Satrec,
    catalog_df: pd.DataFrame,
    margin_m: float,
) -> pd.DataFrame:
    """
    Discard secondaries whose [perigee, apogee] altitude band does not
    overlap with the primary's band (±margin_m). This is a geometry-only
    filter — no propagation, very fast.
    """
    p_apo, p_per = _tle_apogee_perigee_m(primary_sat)
    p_apo_m = p_apo + margin_m
    p_per_m = p_per - margin_m

    # Vectorised extraction of orbital elements
    rows = []
    for _, row in catalog_df.iterrows():
        try:
            s = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
            apo, per = _tle_apogee_perigee_m(s)
        except Exception:
            continue
        # Keep if bands overlap: secondary_apo >= primary_per AND secondary_per <= primary_apo
        if apo >= p_per_m and per <= p_apo_m:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=catalog_df.columns)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: Coarse SGP4 propagation (vectorised with SatrecArray)
# ---------------------------------------------------------------------------

def _build_satrec_array(catalog_df: pd.DataFrame) -> tuple[SatrecArray, list[int], list[str]]:
    """
    Build a SatrecArray from filtered catalog rows.
    Returns (satrec_array, norad_ids, names).
    Dead/bad TLEs are silently dropped here; they'd return error codes anyway.
    """
    sats = []
    norad_ids = []
    names = []
    for _, row in catalog_df.iterrows():
        try:
            s = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
            sats.append(s)
            norad_ids.append(int(row["NORAD_CAT_ID"]))
            names.append(str(row.get("OBJECT_NAME", "UNKNOWN")))
        except Exception:
            continue
    sat_arr = SatrecArray(sats)
    return sat_arr, norad_ids, names


def _coarse_propagation(
    primary_sat: Satrec,
    sat_arr: SatrecArray,
    norad_ids: list[int],
    names: list[str],
    start_dt: datetime,
    end_dt: datetime,
    step_s: float,
    range_threshold_m: float,
) -> list[dict]:
    """
    Propagate primary and all secondaries at coarse step intervals.
    Identifies time indices where range < threshold.
    Returns list of dicts: {norad, name, t_idx, range_m, dt}

    IMPORTANT: SatrecArray.sgp4 signature is:
        e, r, v = sat_arr.sgp4(jd_array, fr_array)
        shapes: e=(N_sats, N_times), r=(N_sats, N_times, 3)

    This means we call it ONCE with all times, getting all satellites × all times.
    """
    # Build time array
    n_steps = max(int((end_dt - start_dt).total_seconds() / step_s) + 1, 1)
    times = [start_dt + timedelta(seconds=i * step_s) for i in range(n_steps)]

    jd_arr = np.zeros(n_steps)
    fr_arr = np.zeros(n_steps)
    for i, dt in enumerate(times):
        jd_arr[i], fr_arr[i] = _jd_split(dt)

    # Propagate ALL secondaries at ALL times in one call
    # e_sec shape: (N_sats, N_times) — error codes
    # r_sec shape: (N_sats, N_times, 3) — positions in km (SGP4 native)
    e_sec, r_sec, v_sec = sat_arr.sgp4(jd_arr, fr_arr)

    # Convert secondary positions to metres immediately
    # r_sec is in km; any error-code entry should be masked
    r_sec_m = r_sec * 1000.0  # (N_sats, N_times, 3) in metres

    # Propagate primary at all times
    e_pri = np.zeros(n_steps, dtype=int)
    r_pri_m = np.zeros((n_steps, 3))
    for i in range(n_steps):
        err, r, v = primary_sat.sgp4(jd_arr[i], fr_arr[i])
        e_pri[i] = err
        if err == 0:
            r_pri_m[i] = np.array(r) * 1000.0  # km → m
        # else: leave as zeros; will be masked below

    # Build mask for valid primary steps
    valid_pri = (e_pri == 0)   # shape (N_times,)

    # For each secondary, build valid mask and compute ranges
    N_sats = len(norad_ids)
    coarse_hits = []

    for i_sat in range(N_sats):
        # Error codes for this satellite across all times
        sat_errors = e_sec[i_sat]  # shape (N_times,)

        # Only keep timesteps where BOTH primary and secondary propagated OK
        valid = valid_pri & (sat_errors == 0)

        if valid.sum() == 0:
            continue

        # Relative position (secondary - primary) in metres, TEME
        dr = r_sec_m[i_sat] - r_pri_m   # (N_times, 3)

        # Range array — only at valid steps
        ranges_m = np.full(n_steps, np.inf)
        ranges_m[valid] = np.linalg.norm(dr[valid], axis=1)

        # Find all local minima below threshold
        below = ranges_m < range_threshold_m
        if not np.any(below):
            continue

        # Find local minima: timestep t is a minimum if range[t] < range[t-1] and range[t] < range[t+1]
        # Also catch endpoints
        for t in range(n_steps):
            if not below[t]:
                continue
            is_local_min = True
            if t > 0 and ranges_m[t - 1] < ranges_m[t]:
                is_local_min = False
            if t < n_steps - 1 and ranges_m[t + 1] < ranges_m[t]:
                is_local_min = False
            if is_local_min:
                coarse_hits.append({
                    "norad": norad_ids[i_sat],
                    "name": names[i_sat],
                    "t_idx": t,
                    "range_m": float(ranges_m[t]),
                    "dt": times[t],
                    "i_sat": i_sat,
                })

    return coarse_hits


# ---------------------------------------------------------------------------
# Step 3: TCA refinement (golden-section search around each coarse minimum)
# ---------------------------------------------------------------------------

def _range_at_time(
    primary_sat: Satrec,
    secondary_sat: Satrec,
    dt: datetime,
) -> float:
    """Return range in metres between primary and secondary at dt. +inf on error."""
    jd, fr = _jd_split(dt)
    e1, r1, _ = primary_sat.sgp4(jd, fr)
    e2, r2, _ = secondary_sat.sgp4(jd, fr)
    if e1 != 0 or e2 != 0:
        return math.inf
    # km → m
    dr = (np.array(r2) - np.array(r1)) * 1000.0
    return float(np.linalg.norm(dr))


def _golden_section_min(
    f,
    a: datetime,
    b: datetime,
    tol_s: float = 0.5,
) -> datetime:
    """
    Golden-section search for the minimum of f in [a, b].
    f must accept a datetime and return a float.
    Returns the datetime of the minimum.
    """
    golden = (math.sqrt(5) - 1) / 2
    total_s = (b - a).total_seconds()

    while total_s > tol_s:
        c = a + timedelta(seconds=(1 - golden) * total_s)
        d = a + timedelta(seconds=golden * total_s)
        if f(c) < f(d):
            b = d
        else:
            a = c
        total_s = (b - a).total_seconds()

    return a + timedelta(seconds=total_s / 2)


def _refine_tca(
    primary_sat: Satrec,
    secondary_sat: Satrec,
    coarse_dt: datetime,
    bracket_s: float,
    step_s: float = 1.0,
) -> ConjunctionEvent | None:
    """
    Refine a coarse conjunction time to sub-second precision.
    Uses golden-section search in [coarse_dt - bracket_s, coarse_dt + bracket_s].
    Returns a fully populated ConjunctionEvent or None if refinement fails.
    """
    a = coarse_dt - timedelta(seconds=bracket_s)
    b = coarse_dt + timedelta(seconds=bracket_s)

    f = lambda dt: _range_at_time(primary_sat, secondary_sat, dt)

    tca_dt = _golden_section_min(f, a, b, tol_s=0.1)

    # Get state vectors at TCA
    jd, fr = _jd_split(tca_dt)
    e1, r1_km, v1_kms = primary_sat.sgp4(jd, fr)
    e2, r2_km, v2_kms = secondary_sat.sgp4(jd, fr)

    if e1 != 0 or e2 != 0:
        return None

    # Convert km → m, km/s → m/s immediately
    r1_m = np.array(r1_km) * 1000.0
    r2_m = np.array(r2_km) * 1000.0
    v1_ms = np.array(v1_kms) * 1000.0
    v2_ms = np.array(v2_kms) * 1000.0

    r_rel_m = r2_m - r1_m
    v_rel_ms = v2_ms - v1_ms
    miss_m = float(np.linalg.norm(r_rel_m))
    rel_v_ms = float(np.linalg.norm(v_rel_ms))

    # Compute epoch offsets for covariance scaling
    def _epoch_dt(sat: Satrec) -> datetime:
        # sgp4 stores epoch as jdsatepoch + jdsatepochF
        jd_epoch = sat.jdsatepoch + sat.jdsatepochF
        # Convert Julian day to datetime
        # JD 2451545.0 = J2000 = 2000-01-01 12:00:00 UTC
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        delta_days = jd_epoch - 2451545.0
        return j2000 + timedelta(days=delta_days)

    pri_epoch = _epoch_dt(primary_sat)
    sec_epoch = _epoch_dt(secondary_sat)
    pri_offset_s = (tca_dt - pri_epoch).total_seconds()
    sec_offset_s = (tca_dt - sec_epoch).total_seconds()

    return ConjunctionEvent(
        primary_norad=-1,       # filled by caller
        secondary_norad=-1,     # filled by caller
        secondary_name="",      # filled by caller
        tca=tca_dt,
        miss_m=miss_m,
        rel_v_ms=rel_v_ms,
        r_rel_teme_m=r_rel_m,
        v_rel_teme_ms=v_rel_ms,
        r_primary_teme_m=r1_m,
        r_secondary_teme_m=r2_m,
        primary_epoch_offset_s=abs(pri_offset_s),
        secondary_epoch_offset_s=abs(sec_offset_s),
        primary_epoch=pri_epoch,
        secondary_epoch=sec_epoch,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def screen(
    norad_id: int,
    window_days: Optional[float] = None,
    catalog: Optional[pd.DataFrame] = None,
    start_dt: Optional[datetime] = None,
) -> list[ConjunctionEvent]:
    """
    Screen a primary satellite against the full catalog.

    Parameters
    ----------
    norad_id : int
        NORAD catalog number of the primary satellite.
    window_days : float, optional
        Screening window in days (default from config).
    catalog : pd.DataFrame, optional
        Pre-loaded catalog. If None, loads from cache.
    start_dt : datetime, optional
        Start of screening window. Default: now UTC.

    Returns
    -------
    list[ConjunctionEvent]
        All conjunctions found, sorted by TCA.
    """
    from data.fetch_catalog import load_catalog, get_object_tle

    cfg = _load_config()
    sc = cfg["screen"]

    if window_days is None:
        window_days = sc["window_days"]
    if catalog is None:
        catalog = load_catalog()
    if start_dt is None:
        start_dt = datetime.now(timezone.utc)

    end_dt = start_dt + timedelta(days=window_days)

    # Get primary TLE
    try:
        line1, line2 = get_object_tle(norad_id, catalog)
    except KeyError:
        raise KeyError(f"Primary NORAD ID {norad_id} not found in catalog")

    primary_sat = _satrec_from_tle(line1, line2)

    # Remove primary from catalog before screening
    cat_secondaries = catalog[catalog["NORAD_CAT_ID"] != norad_id].copy()

    print(f"[screen] Catalog size before filter: {len(cat_secondaries)}")

    # Step 1: Apogee/perigee pre-filter
    margin_m = sc["apogee_perigee_margin_m"]
    cat_filtered = _apogee_perigee_filter(primary_sat, cat_secondaries, margin_m)
    print(f"[screen] After apogee/perigee filter: {len(cat_filtered)}")

    if cat_filtered.empty:
        return []

    # Step 2: Build SatrecArray and run coarse propagation
    sat_arr, sec_norads, sec_names = _build_satrec_array(cat_filtered)
    if len(sec_norads) == 0:
        return []

    coarse_hits = _coarse_propagation(
        primary_sat, sat_arr, sec_norads, sec_names,
        start_dt, end_dt,
        step_s=sc["coarse_step_s"],
        range_threshold_m=sc["coarse_range_m"],
    )
    print(f"[screen] Coarse hits: {len(coarse_hits)}")

    # Step 3: Refine each coarse hit
    # Build secondary Satrec objects for refinement (only those that had hits)
    hit_norads = {h["norad"] for h in coarse_hits}
    sec_satrec_map: dict[int, Satrec] = {}
    for _, row in cat_filtered[cat_filtered["NORAD_CAT_ID"].isin(hit_norads)].iterrows():
        nid = int(row["NORAD_CAT_ID"])
        try:
            sec_satrec_map[nid] = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
        except Exception:
            pass

    events = []
    bracket_s = sc["refine_bracket_s"]

    for hit in coarse_hits:
        sec_norad = hit["norad"]
        if sec_norad not in sec_satrec_map:
            continue
        sec_sat = sec_satrec_map[sec_norad]

        event = _refine_tca(primary_sat, sec_sat, hit["dt"], bracket_s)
        if event is None:
            continue

        event.primary_norad = norad_id
        event.secondary_norad = sec_norad
        event.secondary_name = hit["name"]
        events.append(event)

    # Sort by TCA
    events.sort(key=lambda e: e.tca)
    print(f"[screen] Refined conjunctions: {len(events)}")
    return events


def screen_for_trajectory(
    r_teme_m: np.ndarray,
    v_teme_ms: np.ndarray,
    times: list[datetime],
    primary_norad: int,
    catalog: Optional[pd.DataFrame] = None,
    range_threshold_m: Optional[float] = None,
) -> list[ConjunctionEvent]:
    """
    Screen a custom trajectory (e.g. post-maneuver propagation) against
    the full catalog. Used by core/rescreen.py.

    Parameters
    ----------
    r_teme_m : np.ndarray
        Primary positions in TEME, shape (N_times, 3), metres.
    v_teme_ms : np.ndarray
        Primary velocities in TEME, shape (N_times, 3), m/s.
    times : list[datetime]
        UTC datetimes corresponding to each row of r_teme_m.
    primary_norad : int
        NORAD ID of the primary (for output labelling).
    catalog : pd.DataFrame, optional
        Catalog. If None, loads from cache.
    range_threshold_m : float, optional
        Coarse range filter. Default from config.

    Returns
    -------
    list[ConjunctionEvent]
    """
    from data.fetch_catalog import load_catalog

    cfg = _load_config()
    sc = cfg["screen"]

    if catalog is None:
        catalog = load_catalog()
    if range_threshold_m is None:
        range_threshold_m = sc["coarse_range_m"]

    cat_secondaries = catalog[catalog["NORAD_CAT_ID"] != primary_norad].copy()

    # Build time arrays for SGP4
    n_steps = len(times)
    jd_arr = np.zeros(n_steps)
    fr_arr = np.zeros(n_steps)
    for i, dt in enumerate(times):
        jd_arr[i], fr_arr[i] = _jd_split(dt)

    # Build SatrecArray for all secondaries
    sat_arr, sec_norads, sec_names = _build_satrec_array(cat_secondaries)
    if len(sec_norads) == 0:
        return []

    # Propagate secondaries
    e_sec, r_sec, v_sec = sat_arr.sgp4(jd_arr, fr_arr)
    r_sec_m = r_sec * 1000.0   # km → m
    v_sec_ms = v_sec * 1000.0  # km/s → m/s

    # r_teme_m is already in metres
    coarse_hits = []
    N_sats = len(sec_norads)

    for i_sat in range(N_sats):
        sat_errors = e_sec[i_sat]
        valid = (sat_errors == 0)
        if valid.sum() == 0:
            continue

        dr = r_sec_m[i_sat] - r_teme_m   # (N_times, 3)
        ranges_m = np.full(n_steps, np.inf)
        ranges_m[valid] = np.linalg.norm(dr[valid], axis=1)

        below = ranges_m < range_threshold_m
        if not np.any(below):
            continue

        for t in range(n_steps):
            if not below[t]:
                continue
            is_local_min = True
            if t > 0 and ranges_m[t - 1] < ranges_m[t]:
                is_local_min = False
            if t < n_steps - 1 and ranges_m[t + 1] < ranges_m[t]:
                is_local_min = False
            if is_local_min:
                coarse_hits.append({
                    "norad": sec_norads[i_sat],
                    "name": sec_names[i_sat],
                    "t_idx": t,
                    "range_m": float(ranges_m[t]),
                    "dt": times[t],
                    "i_sat": i_sat,
                    "r_pri_m": r_teme_m[t],
                    "v_pri_ms": v_teme_ms[t],
                    "r_sec_m": r_sec_m[i_sat, t],
                    "v_sec_ms": v_sec_ms[i_sat, t],
                })

    # For trajectory-based screening, we build ConjunctionEvents directly
    # (no SGP4 refinement on the primary since we have explicit trajectory)
    events = []
    hit_norads = {h["norad"] for h in coarse_hits}
    sec_satrec_map: dict[int, Satrec] = {}
    for _, row in cat_secondaries[cat_secondaries["NORAD_CAT_ID"].isin(hit_norads)].iterrows():
        nid = int(row["NORAD_CAT_ID"])
        try:
            sec_satrec_map[nid] = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
        except Exception:
            pass

    for hit in coarse_hits:
        sec_norad = hit["norad"]
        if sec_norad not in sec_satrec_map:
            continue
        sec_sat = sec_satrec_map[sec_norad]

        # Find refined TCA by stepping 1s around coarse point
        coarse_t = hit["dt"]
        bracket_s = sc["refine_bracket_s"]

        # Build local time grid for refinement
        refine_times = [
            coarse_t + timedelta(seconds=s)
            for s in range(-int(bracket_s), int(bracket_s) + 1, 1)
        ]
        best_range = np.inf
        best_t = coarse_t
        best_r_pri = hit["r_pri_m"]
        best_r_sec = hit["r_sec_m"]
        best_v_sec = hit["v_sec_ms"]

        # For trajectory screen, primary position at nearby steps interpolated
        # via the surrounding known positions
        t_idx = hit["t_idx"]
        for rt in refine_times:
            jd, fr = _jd_split(rt)
            e2, r2_km, v2_kms = sec_sat.sgp4(jd, fr)
            if e2 != 0:
                continue
            r2_m = np.array(r2_km) * 1000.0
            v2_ms = np.array(v2_kms) * 1000.0

            # Interpolate primary position from trajectory
            # Find nearest trajectory index
            t0 = times[0]
            dt_s = (rt - t0).total_seconds()
            f_idx = dt_s / ((times[1] - times[0]).total_seconds() if len(times) > 1 else 60.0)
            i_lo = max(0, min(int(f_idx), n_steps - 2))
            alpha = f_idx - i_lo
            alpha = max(0.0, min(1.0, alpha))
            r_pri_interp = r_teme_m[i_lo] * (1 - alpha) + r_teme_m[min(i_lo + 1, n_steps - 1)] * alpha
            v_pri_interp = v_teme_ms[i_lo] * (1 - alpha) + v_teme_ms[min(i_lo + 1, n_steps - 1)] * alpha

            dist = float(np.linalg.norm(r2_m - r_pri_interp))
            if dist < best_range:
                best_range = dist
                best_t = rt
                best_r_pri = r_pri_interp
                best_r_sec = r2_m
                best_v_sec = v2_ms

        if best_range == np.inf:
            continue

        r_rel_m = best_r_sec - best_r_pri

        # Get primary velocity at TCA (interpolated)
        t0 = times[0]
        dt_s_tca = (best_t - t0).total_seconds()
        step_s = (times[1] - times[0]).total_seconds() if len(times) > 1 else 60.0
        f_idx = dt_s_tca / step_s
        i_lo = max(0, min(int(f_idx), n_steps - 2))
        alpha = max(0.0, min(1.0, f_idx - i_lo))
        v_pri_at_tca = v_teme_ms[i_lo] * (1 - alpha) + v_teme_ms[min(i_lo + 1, n_steps - 1)] * alpha

        v_rel_ms_vec = best_v_sec - v_pri_at_tca

        # Get primary epoch from catalog if available
        pri_rows = catalog[catalog["NORAD_CAT_ID"] == primary_norad]
        if not pri_rows.empty and pd.notna(pri_rows.iloc[0].get("EPOCH")):
            pri_epoch = pri_rows.iloc[0]["EPOCH"].to_pydatetime()
        else:
            pri_epoch = best_t  # fallback
        pri_offset_s = abs((best_t - pri_epoch).total_seconds())

        sec_sat_obj = sec_satrec_map[sec_norad]
        from datetime import timezone as _tz
        j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd_epoch = sec_sat_obj.jdsatepoch + sec_sat_obj.jdsatepochF
        sec_epoch = j2000 + timedelta(days=jd_epoch - 2451545.0)
        sec_offset_s = abs((best_t - sec_epoch).total_seconds())

        event = ConjunctionEvent(
            primary_norad=primary_norad,
            secondary_norad=sec_norad,
            secondary_name=hit["name"],
            tca=best_t,
            miss_m=best_range,
            rel_v_ms=float(np.linalg.norm(v_rel_ms_vec)),
            r_rel_teme_m=r_rel_m,
            v_rel_teme_ms=v_rel_ms_vec,
            r_primary_teme_m=best_r_pri,
            r_secondary_teme_m=best_r_sec,
            primary_epoch_offset_s=pri_offset_s,
            secondary_epoch_offset_s=sec_offset_s,
            primary_epoch=pri_epoch if isinstance(pri_epoch, datetime) else best_t,
            secondary_epoch=sec_epoch,
        )
        events.append(event)

    events.sort(key=lambda e: e.tca)
    return events
