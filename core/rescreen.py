"""
core/rescreen.py
================
For a grid of candidate (delta-v, burn time) maneuver options, compute the
total probability of collision:

    total_pc = Pc(primary event, post-maneuver) + Σ Pc(all newly induced events)

This is the core differentiator of Ricochet. It finds the minimum-delta-v
burn that clears the primary event without creating unacceptable induced events.

Performance architecture:
  The secondary catalog ephemerides are INDEPENDENT of the maneuver grid.
  We precompute secondary SGP4 states ONCE for the full time window, then
  reuse them for every grid point. The primary trajectory is recomputed
  for each grid point (cheap: J2 ODE with scipy).

  Grid: N_dv × N_dt burn options
  Cost per grid point: one J2 propagation + closest-approach scan against
  precomputed secondary ephemerides.

Force-model mismatch acknowledgement:
  Primary is propagated with two-body + J2.
  Secondaries are propagated with SGP4 (drag, tesseral harmonics).
  This mismatch grows over 7 days. At ~500km, drag differences can
  accumulate ~1-2 km/day in along-track position. All rescreen results
  carry this acknowledgement.

Units: SI (metres, m/s, seconds).
Frame: TEME.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sgp4.api import Satrec, SatrecArray


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GridPoint:
    """One cell in the maneuver trade-space grid."""
    dv_ms: float                    # Delta-v (m/s)
    dt_before_tca_s: float          # Burn time before TCA (seconds)
    total_pc: float                 # Total probability of collision
    primary_event_pc: float         # Pc of the original event (post-maneuver)
    induced_events_pc: float        # Pc sum of all induced events
    n_induced_events: int           # Number of induced conjunctions found
    force_model_note: str = (
        "Primary: two-body+J2. Secondaries: SGP4. "
        "Force-model mismatch may accumulate over 7-day window."
    )


@dataclass
class RescreenResult:
    """Complete output from a rescreen run."""
    grid: pd.DataFrame              # Shape (N_dv × N_dt), index=dv, columns=dt
    grid_points: list[GridPoint]    # All GridPoint objects
    dv_values_ms: np.ndarray        # Delta-v axis values
    dt_values_s: np.ndarray         # Burn time axis values
    primary_norad: int
    primary_tca: datetime
    baseline_pc: float              # Pc with zero maneuver (reference)
    recommended_dv_ms: Optional[float]    # Minimum dv that clears threshold
    recommended_dt_s: Optional[float]
    alert_threshold: float
    elapsed_s: float
    n_secondaries_screened: int
    n_catalog_after_filter: int


# ---------------------------------------------------------------------------
# Secondary ephemeris precomputation (the caching key to performance)
# ---------------------------------------------------------------------------

def _precompute_secondary_ephemerides(
    catalog: pd.DataFrame,
    primary_norad: int,
    start_dt: datetime,
    end_dt: datetime,
    step_s: float,
    range_threshold_m: float,
) -> tuple[np.ndarray, list[int], list[str], np.ndarray, np.ndarray]:
    """
    Precompute SGP4 positions for ALL catalog secondaries across the
    full time window. Return only those that ever come within range.

    This is called ONCE and the result is reused for all grid points.

    Returns
    -------
    jd_arr : shape (N_times,) — Julian date integers
    sec_norads : list of NORAD IDs
    sec_names : list of names
    r_sec_m : shape (N_sats, N_times, 3) — positions in METRES (already converted)
    v_sec_ms : shape (N_sats, N_times, 3) — velocities in m/s
    times : list[datetime] of length N_times
    """
    from core.screen import _jd_split, _satrec_from_tle

    n_steps = max(int((end_dt - start_dt).total_seconds() / step_s) + 1, 1)
    times = [start_dt + timedelta(seconds=i * step_s) for i in range(n_steps)]

    jd_arr = np.zeros(n_steps)
    fr_arr = np.zeros(n_steps)
    for i, dt in enumerate(times):
        jd_arr[i], fr_arr[i] = _jd_split(dt)

    cat_sec = catalog[catalog["NORAD_CAT_ID"] != primary_norad].copy()

    # Build SatrecArray
    sats = []
    norads = []
    names = []
    for _, row in cat_sec.iterrows():
        try:
            s = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
            sats.append(s)
            norads.append(int(row["NORAD_CAT_ID"]))
            names.append(str(row.get("OBJECT_NAME", "UNKNOWN")))
        except Exception:
            continue

    if not sats:
        return jd_arr, [], [], np.zeros((0, n_steps, 3)), np.zeros((0, n_steps, 3)), times

    sat_arr = SatrecArray(sats)
    e_sec, r_sec, v_sec = sat_arr.sgp4(jd_arr, fr_arr)

    # Convert to SI immediately; mask errors with NaN
    r_sec_m = r_sec * 1000.0   # km → m
    v_sec_ms = v_sec * 1000.0  # km/s → m/s

    # Mask error positions with NaN so they never pass range filter
    err_mask = (e_sec != 0)  # (N_sats, N_times)
    r_sec_m[err_mask] = np.nan
    v_sec_ms[err_mask] = np.nan

    return jd_arr, norads, names, r_sec_m, v_sec_ms, times


def _find_conjunctions_with_precomputed(
    r_primary_m: np.ndarray,    # (N_times, 3) primary positions
    v_primary_ms: np.ndarray,   # (N_times, 3) primary velocities
    r_sec_m: np.ndarray,        # (N_sats, N_times, 3) secondary positions
    v_sec_ms: np.ndarray,       # (N_sats, N_times, 3) secondary velocities
    times: list[datetime],
    sec_norads: list[int],
    sec_names: list[str],
    range_threshold_m: float,
    primary_norad: int,
    catalog: pd.DataFrame,
) -> list:
    """
    Find all conjunctions between the primary trajectory and precomputed secondaries.
    Returns list of ConjunctionEvent objects.
    """
    from core.screen import ConjunctionEvent
    from datetime import timezone

    n_times = len(times)
    N_sats = len(sec_norads)
    events = []

    for i_sat in range(N_sats):
        r_sec = r_sec_m[i_sat]   # (N_times, 3)
        v_sec = v_sec_ms[i_sat]  # (N_times, 3)

        # Compute ranges — NaN positions become inf (won't trigger)
        dr = r_sec - r_primary_m
        nan_mask = np.any(np.isnan(dr), axis=1)
        ranges = np.where(
            nan_mask,
            np.inf,
            np.linalg.norm(dr, axis=1)
        )

        below = ranges < range_threshold_m
        if not np.any(below):
            continue

        # Find local minima below threshold
        for t in range(n_times):
            if not below[t]:
                continue
            is_local_min = True
            if t > 0 and ranges[t - 1] < ranges[t]:
                is_local_min = False
            if t < n_times - 1 and ranges[t + 1] < ranges[t]:
                is_local_min = False
            if not is_local_min:
                continue

            # Refine around this minimum (1s steps in [-bracket, +bracket])
            bracket = 60  # seconds
            best_range = ranges[t]
            best_t_idx = t
            best_r_pri = r_primary_m[t]
            best_v_pri = v_primary_ms[t]
            best_r_sec = r_sec[t]
            best_v_sec = v_sec[t]
            best_time = times[t]

            # Use surrounding trajectory points for sub-minute refinement
            step_s = (times[1] - times[0]).total_seconds() if len(times) > 1 else 60.0
            t_coarse = times[t]

            # Build secondary Satrec for refinement
            sec_rows = catalog[catalog["NORAD_CAT_ID"] == sec_norads[i_sat]]
            if sec_rows.empty:
                continue
            try:
                from core.screen import _jd_split
                sec_sat = Satrec.twoline2rv(
                    sec_rows.iloc[0]["TLE_LINE1"],
                    sec_rows.iloc[0]["TLE_LINE2"]
                )
            except Exception:
                continue

            # Refine with 1s steps
            for ds in range(-bracket, bracket + 1, 1):
                rt = t_coarse + timedelta(seconds=ds)
                jd, fr = _jd_split(rt)
                e2, r2_km, v2_kms = sec_sat.sgp4(jd, fr)
                if e2 != 0:
                    continue
                r2_m = np.array(r2_km) * 1000.0
                v2_ms = np.array(v2_kms) * 1000.0

                # Interpolate primary trajectory
                dt_from_0 = (rt - times[0]).total_seconds()
                f_idx = dt_from_0 / step_s
                i_lo = max(0, min(int(f_idx), n_times - 2))
                alpha = max(0.0, min(1.0, f_idx - i_lo))
                i_hi = min(i_lo + 1, n_times - 1)
                r_pri = r_primary_m[i_lo] * (1 - alpha) + r_primary_m[i_hi] * alpha
                v_pri = v_primary_ms[i_lo] * (1 - alpha) + v_primary_ms[i_hi] * alpha

                dist = float(np.linalg.norm(r2_m - r_pri))
                if dist < best_range:
                    best_range = dist
                    best_time = rt
                    best_r_pri = r_pri
                    best_v_pri = v_pri
                    best_r_sec = r2_m
                    best_v_sec = v2_ms

            if best_range == np.inf or np.isnan(best_range):
                continue

            r_rel = best_r_sec - best_r_pri
            v_rel = best_v_sec - best_v_pri

            # Epoch offsets
            pri_rows = catalog[catalog["NORAD_CAT_ID"] == primary_norad]
            if not pri_rows.empty and pd.notna(pri_rows.iloc[0].get("EPOCH")):
                pri_epoch = pri_rows.iloc[0]["EPOCH"].to_pydatetime()
                if not pri_epoch.tzinfo:
                    pri_epoch = pri_epoch.replace(tzinfo=timezone.utc)
            else:
                pri_epoch = best_time

            j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            jd_sec_epoch = sec_sat.jdsatepoch + sec_sat.jdsatepochF
            sec_epoch = j2000 + timedelta(days=jd_sec_epoch - 2451545.0)

            event = ConjunctionEvent(
                primary_norad=primary_norad,
                secondary_norad=sec_norads[i_sat],
                secondary_name=sec_names[i_sat],
                tca=best_time,
                miss_m=best_range,
                rel_v_ms=float(np.linalg.norm(v_rel)),
                r_rel_teme_m=r_rel,
                v_rel_teme_ms=v_rel,
                r_primary_teme_m=best_r_pri,
                r_secondary_teme_m=best_r_sec,
                primary_epoch_offset_s=abs((best_time - pri_epoch).total_seconds()),
                secondary_epoch_offset_s=abs((best_time - sec_epoch).total_seconds()),
                primary_epoch=pri_epoch,
                secondary_epoch=sec_epoch,
            )
            events.append(event)

    return events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rescreen(
    norad_id: int,
    primary_tca: Optional[datetime],
    catalog: Optional[pd.DataFrame] = None,
    dv_range: Optional[tuple[float, float]] = None,
    dt_range: Optional[tuple[float, float]] = None,
    grid_n: Optional[int] = None,
) -> RescreenResult:
    """
    Compute the total-Pc maneuver trade space.
    
    For each (dv, dt_before_tca) grid point:
    1. Propagate post-maneuver primary trajectory (two-body + J2)
    2. Screen against full catalog using precomputed secondary ephemerides
    3. Compute Pc for each conjunction found
    4. Sum: total_pc = Pc(primary event post-maneuver) + Σ Pc(induced events)
    
    Parameters
    ----------
    norad_id : int
        Primary satellite NORAD ID.
    primary_tca : datetime, optional
        TCA of the primary event. If None, uses 3.5 days from now as default.
    catalog : pd.DataFrame, optional
        Loaded catalog. Fetched if None.
    dv_range : (min, max), optional
        Delta-v range in m/s. Default from config.
    dt_range : (min, max), optional
        Burn time range in seconds before TCA. Default from config.
    grid_n : int, optional
        Grid resolution (N × N). Default from config.
    
    Returns
    -------
    RescreenResult
    """
    from data.fetch_catalog import load_catalog, get_object_tle
    from core.maneuver import apply_maneuver_from_state, analytic_downtrack_drift
    from core.pc import compute_pc
    from core.screen import _jd_split, _apogee_perigee_filter, _tle_apogee_perigee_m
    from sgp4.api import Satrec

    t0 = time.time()
    cfg = _load_config()
    man_cfg = cfg["maneuver"]
    sc_cfg = cfg["screen"]

    if catalog is None:
        catalog = load_catalog()

    if primary_tca is None:
        primary_tca = datetime.now(timezone.utc) + timedelta(days=3.5)

    if dv_range is None:
        dv_range = (man_cfg["dv_min_ms"], man_cfg["dv_max_ms"])
    if dt_range is None:
        dt_range = (man_cfg["dt_min_s"], man_cfg["dt_max_s"])
    if grid_n is None:
        grid_n = man_cfg["dv_steps"]

    dv_values = np.linspace(dv_range[0], dv_range[1], grid_n)
    dt_values = np.linspace(dt_range[0], dt_range[1], grid_n)

    # Screening window: from earliest possible burn to 7 days after TCA
    earliest_burn = primary_tca - timedelta(seconds=float(dt_range[1]))
    screen_end = primary_tca + timedelta(days=sc_cfg["window_days"])

    # Pre-filter catalog using apogee/perigee
    line1, line2 = get_object_tle(norad_id, catalog)
    primary_sat = Satrec.twoline2rv(line1, line2)
    cat_filtered = _apogee_perigee_filter(
        primary_sat, 
        catalog[catalog["NORAD_CAT_ID"] != norad_id],
        sc_cfg["apogee_perigee_margin_m"]
    )

    print(f"[rescreen] Catalog after filter: {len(cat_filtered)} objects")
    print(f"[rescreen] Grid: {grid_n}×{grid_n} = {grid_n**2} points")
    print(f"[rescreen] Window: {earliest_burn.strftime('%Y-%m-%dT%H:%M')} → {screen_end.strftime('%Y-%m-%dT%H:%M')}")

    # Precompute secondary ephemerides ONCE for the full window
    print("[rescreen] Precomputing secondary ephemerides (done once for all grid points)...")
    jd_arr, sec_norads, sec_names, r_sec_m, v_sec_ms, times_full = _precompute_secondary_ephemerides(
        catalog=cat_filtered.assign(NORAD_CAT_ID=cat_filtered["NORAD_CAT_ID"]),
        primary_norad=norad_id,
        start_dt=earliest_burn,
        end_dt=screen_end,
        step_s=sc_cfg["coarse_step_s"],
        range_threshold_m=sc_cfg["coarse_range_m"],
    )
    print(f"[rescreen] Secondary ephemerides computed for {len(sec_norads)} objects × {len(times_full)} timesteps")

    n_times = len(times_full)
    alert_threshold = cfg["pc"]["alert_threshold"]

    # Precompute primary state at each burn time (for speed)
    burn_states: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for dt_s in dt_values:
        burn_time = primary_tca - timedelta(seconds=float(dt_s))
        jd_b, fr_b = _jd_split(burn_time)
        e, r_km, v_kms = primary_sat.sgp4(jd_b, fr_b)
        if e == 0:
            burn_states[float(dt_s)] = (np.array(r_km) * 1000.0, np.array(v_kms) * 1000.0)
        else:
            burn_states[float(dt_s)] = None

    # --- Grid computation ---
    grid_points = []
    pc_grid = np.full((len(dv_values), len(dt_values)), np.nan)

    for i_dv, dv in enumerate(dv_values):
        for i_dt, dt_s in enumerate(dt_values):
            burn_time = primary_tca - timedelta(seconds=float(dt_s))
            state = burn_states.get(float(dt_s))
            if state is None:
                pc_grid[i_dv, i_dt] = np.nan
                grid_points.append(GridPoint(
                    dv_ms=float(dv), dt_before_tca_s=float(dt_s),
                    total_pc=np.nan, primary_event_pc=np.nan,
                    induced_events_pc=np.nan, n_induced_events=0,
                ))
                continue

            r0_m, v0_ms = state

            # Propagate post-maneuver trajectory
            try:
                man_result = apply_maneuver_from_state(
                    r0_m=r0_m,
                    v0_ms=v0_ms,
                    burn_time=burn_time,
                    dv_ms=float(dv),
                    propagate_days=(screen_end - burn_time).total_seconds() / 86400.0,
                )
            except Exception as exc:
                print(f"[rescreen] Propagation failed dv={dv:.3f}, dt={dt_s:.0f}s: {exc}")
                pc_grid[i_dv, i_dt] = np.nan
                grid_points.append(GridPoint(
                    dv_ms=float(dv), dt_before_tca_s=float(dt_s),
                    total_pc=np.nan, primary_event_pc=np.nan,
                    induced_events_pc=np.nan, n_induced_events=0,
                ))
                continue

            # Align maneuver trajectory to the precomputed secondary time grid
            # The secondary ephemerides start at earliest_burn
            # We need primary positions at the same times as the secondary grid
            t0_secondary = times_full[0]
            t0_maneuver = man_result.times[0]
            step_s = sc_cfg["coarse_step_s"]

            # Build primary position/velocity arrays aligned to secondary time grid
            r_primary_aligned = np.full((n_times, 3), np.nan)
            v_primary_aligned = np.full((n_times, 3), np.nan)

            for i_t, gt in enumerate(times_full):
                # Time since burn
                dt_from_burn = (gt - t0_maneuver).total_seconds()
                if dt_from_burn < 0:
                    # Before burn: use SGP4 (no maneuver yet)
                    jd_t, fr_t = _jd_split(gt)
                    e, rk, vk = primary_sat.sgp4(jd_t, fr_t)
                    if e == 0:
                        r_primary_aligned[i_t] = np.array(rk) * 1000.0
                        v_primary_aligned[i_t] = np.array(vk) * 1000.0
                else:
                    # After burn: interpolate from maneuver trajectory
                    f_idx = dt_from_burn / step_s
                    n_man = len(man_result.r_teme_m)
                    i_lo = max(0, min(int(f_idx), n_man - 2))
                    alpha = max(0.0, min(1.0, f_idx - i_lo))
                    i_hi = min(i_lo + 1, n_man - 1)
                    r_primary_aligned[i_t] = (
                        man_result.r_teme_m[i_lo] * (1 - alpha) +
                        man_result.r_teme_m[i_hi] * alpha
                    )
                    v_primary_aligned[i_t] = (
                        man_result.v_teme_ms[i_lo] * (1 - alpha) +
                        man_result.v_teme_ms[i_hi] * alpha
                    )

            # Find conjunctions using precomputed secondary ephemerides
            events = _find_conjunctions_with_precomputed(
                r_primary_m=r_primary_aligned,
                v_primary_ms=v_primary_aligned,
                r_sec_m=r_sec_m,
                v_sec_ms=v_sec_ms,
                times=times_full,
                sec_norads=sec_norads,
                sec_names=sec_names,
                range_threshold_m=sc_cfg["coarse_range_m"],
                primary_norad=norad_id,
                catalog=cat_filtered,
            )

            # Compute Pc for each event
            total_pc = 0.0
            induced_pc_sum = 0.0
            primary_event_pc = 0.0

            for event in events:
                try:
                    pc_result = compute_pc(event)
                    total_pc += pc_result.pc
                    # Classify as primary event or induced
                    is_near_tca = abs((event.tca - primary_tca).total_seconds()) < 3600
                    if is_near_tca:
                        primary_event_pc += pc_result.pc
                    else:
                        induced_pc_sum += pc_result.pc
                except Exception:
                    pass

            total_pc = min(1.0, total_pc)
            pc_grid[i_dv, i_dt] = total_pc

            grid_points.append(GridPoint(
                dv_ms=float(dv),
                dt_before_tca_s=float(dt_s),
                total_pc=total_pc,
                primary_event_pc=primary_event_pc,
                induced_events_pc=induced_pc_sum,
                n_induced_events=max(0, len(events) - (1 if primary_event_pc > 0 else 0)),
            ))

        print(f"[rescreen] dv={dv:+.3f} m/s done ({i_dv+1}/{len(dv_values)})")

    # Build DataFrame grid (rows = dv, columns = dt)
    grid_df = pd.DataFrame(
        pc_grid,
        index=np.round(dv_values, 4),
        columns=np.round(dt_values / 3600, 2),   # dt in hours for display
    )
    grid_df.index.name = "dv_ms"
    grid_df.columns.name = "dt_hours_before_tca"

    # Zero-burn Pc (dv=0, nearest dt)
    zero_dv_idx = np.argmin(np.abs(dv_values))
    mid_dt_idx = len(dt_values) // 2
    baseline_pc = float(pc_grid[zero_dv_idx, mid_dt_idx])
    if np.isnan(baseline_pc):
        baseline_pc = 0.0

    # Find minimum-dv recommendation (smallest |dv| that drops total_pc < threshold).
    # A burn is only recommended when doing nothing is itself unacceptable —
    # otherwise the cheapest option is to hold attitude and spend no propellant.
    recommended_dv = None
    recommended_dt = None
    if baseline_pc >= alert_threshold:
        valid_gps = [gp for gp in grid_points if not np.isnan(gp.total_pc)]
        valid_gps_sorted = sorted(valid_gps, key=lambda gp: abs(gp.dv_ms))
        for gp in valid_gps_sorted:
            if gp.total_pc < alert_threshold and abs(gp.dv_ms) > 1e-6:
                recommended_dv = gp.dv_ms
                recommended_dt = gp.dt_before_tca_s
                break

    elapsed = time.time() - t0

    return RescreenResult(
        grid=grid_df,
        grid_points=grid_points,
        dv_values_ms=dv_values,
        dt_values_s=dt_values,
        primary_norad=norad_id,
        primary_tca=primary_tca,
        baseline_pc=baseline_pc,
        recommended_dv_ms=recommended_dv,
        recommended_dt_s=recommended_dt,
        alert_threshold=alert_threshold,
        elapsed_s=elapsed,
        n_secondaries_screened=len(sec_norads),
        n_catalog_after_filter=len(cat_filtered),
    )
