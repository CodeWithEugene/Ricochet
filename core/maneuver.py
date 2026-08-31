"""
core/maneuver.py
================
Apply an along-track delta-v and propagate the post-maneuver trajectory
using numerical integration (two-body + J2 perturbation).

Design decisions:
- Maneuver is an impulsive along-track burn (velocity change in tangential direction)
- Propagation: adaptive Runge-Kutta integrator (scipy, DOP853 by default) with J2 perturbation
- All constants from config.yaml
- The first-order analytic drift (3 * dv * dt) is exposed as a TEST ORACLE ONLY
  and is clearly labelled as such; it is never used in the actual implementation

Force model note:
- This integrator uses two-body + J2 only.
- SGP4 (used for secondaries) additionally includes drag, lunar/solar, tesseral harmonics.
- The resulting force-model mismatch grows over 7 days, particularly at <600km altitude
  where drag is significant. This is acknowledged in ASSUMPTIONS.md and documented
  on every rescreen output.

Units: SI throughout (metres, m/s, seconds).
Frame: TEME (positions and velocities).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.integrate import solve_ivp


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ManeuverResult:
    """Post-maneuver trajectory and metadata."""
    # Burn parameters
    dv_ms: float                # Delta-v magnitude, positive = prograde (m/s)
    burn_time: datetime         # UTC time of burn
    burn_r_teme_m: np.ndarray   # Position at burn time (metres, TEME)
    burn_v_pre_teme_ms: np.ndarray   # Velocity before burn (m/s, TEME)
    burn_v_post_teme_ms: np.ndarray  # Velocity after burn (m/s, TEME)

    # Propagated trajectory
    times: list[datetime]       # UTC datetimes (length N)
    r_teme_m: np.ndarray        # Shape (N, 3), metres
    v_teme_ms: np.ndarray       # Shape (N, 3), m/s

    # Verification metrics
    downtrack_offset_m: float   # Downtrack displacement at end of propagation
    analytic_oracle_m: float    # First-order analytic prediction (TEST ORACLE)
    analytic_ratio: float       # downtrack_offset_m / analytic_oracle_m

    # Force model
    force_model: str = "two_body_J2"


# ---------------------------------------------------------------------------
# Force model: two-body + J2
# ---------------------------------------------------------------------------

def _equations_of_motion(t: float, y: np.ndarray, mu: float, re: float, j2: float) -> np.ndarray:
    """
    Equations of motion for two-body + J2 perturbation.
    
    State vector y = [x, y, z, vx, vy, vz] in metres and m/s.
    Returns dy/dt = [vx, vy, vz, ax, ay, az].
    
    J2 acceleration:
    a_J2 = -(3/2) * J2 * mu * Re² / r^5 * [x*(1 - 5z²/r²),
                                              y*(1 - 5z²/r²),
                                              z*(3 - 5z²/r²)]
    """
    x, y_, z, vx, vy, vz = y
    r = math.sqrt(x*x + y_*y_ + z*z)
    r2 = r * r
    r3 = r2 * r
    r5 = r3 * r2

    # Two-body acceleration
    ax_2b = -mu * x / r3
    ay_2b = -mu * y_ / r3
    az_2b = -mu * z / r3

    # J2 perturbation
    z2_over_r2 = (z * z) / r2
    factor = -(3.0 / 2.0) * j2 * mu * re * re / r5
    ax_j2 = factor * x * (1.0 - 5.0 * z2_over_r2)
    ay_j2 = factor * y_ * (1.0 - 5.0 * z2_over_r2)
    az_j2 = factor * z * (3.0 - 5.0 * z2_over_r2)

    return np.array([
        vx, vy, vz,
        ax_2b + ax_j2,
        ay_2b + ay_j2,
        az_2b + az_j2,
    ])


def _integrator_settings(prop_cfg: dict, man_cfg: dict) -> dict:
    """Integrator keyword arguments for `_propagate_j2`, drawn from config.yaml."""
    max_step = man_cfg.get("integrator_max_step_s")
    return {
        "method": str(prop_cfg.get("integrator_method", "DOP853")),
        "rtol": float(prop_cfg.get("rtol", 1e-11)),
        "atol": float(prop_cfg.get("atol", 1e-4)),
        "max_step_s": None if max_step is None else float(max_step),
    }


def _propagate_j2(
    r0_m: np.ndarray,
    v0_ms: np.ndarray,
    t_span_s: tuple[float, float],
    output_times_s: np.ndarray,
    mu: float,
    re: float,
    j2: float,
    max_step_s: Optional[float] = None,
    method: str = "DOP853",
    rtol: float = 1e-11,
    atol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate state (r, v) with J2 perturbation using an adaptive Runge-Kutta
    integrator.
    
    Parameters
    ----------
    r0_m, v0_ms : initial state (metres, m/s)
    t_span_s : (t_start, t_end) relative seconds from burn
    output_times_s : array of times (seconds) at which to evaluate state
    max_step_s : optional cap on integrator step size. None = adaptive only.
    method, rtol, atol : integrator settings (see config.yaml `propagation`)
    
    Returns
    -------
    r_out : shape (N, 3), metres
    v_out : shape (N, 3), m/s
    """
    y0 = np.concatenate([r0_m, v0_ms])

    sol = solve_ivp(
        fun=lambda t, y: _equations_of_motion(t, y, mu, re, j2),
        t_span=t_span_s,
        y0=y0,
        method=method,
        t_eval=output_times_s,
        rtol=rtol,
        atol=atol,
        max_step=np.inf if max_step_s is None else max_step_s,
    )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    r_out = sol.y[:3, :].T  # shape (N, 3)
    v_out = sol.y[3:, :].T  # shape (N, 3)
    return r_out, v_out


# ---------------------------------------------------------------------------
# Delta-v application
# ---------------------------------------------------------------------------

def _along_track_unit(r_teme_m: np.ndarray, v_teme_ms: np.ndarray) -> np.ndarray:
    """
    Return the unit vector in the along-track (tangential) direction.
    Along-track = direction of velocity in the orbital plane.
    For a simple prograde burn, dv * along_track_unit is the impulse.
    """
    v_mag = np.linalg.norm(v_teme_ms)
    if v_mag < 1.0:
        raise ValueError("Velocity too small to determine along-track direction")
    return v_teme_ms / v_mag


def analytic_downtrack_drift(dv_ms: float, dt_s: float) -> float:
    """
    FIRST-ORDER ANALYTIC APPROXIMATION — TEST ORACLE ONLY.
    
    Never use this as the implementation. It is only here to provide a
    sanity-check bound for the numerical propagation.
    
    For a small along-track impulse dv at time dt before a reference time,
    the downtrack displacement at the reference time is approximately:
    
        Δs ≈ 3 * |dv| * |dt|
    
    This comes from the Clohessy-Wiltshire (Hill's) equations for relative
    motion in a circular orbit:
        x(t) = -(3/2) * n * dv_x / n * t + ...   (not shown)
        y(t) ≈ 3 * dv_T * t                        (dominant term)
    where dv_T is the tangential impulse and t is time after the burn.
    
    Reference: Schaub & Junkins, "Analytical Mechanics of Space Systems", Eq. 14.81
    
    Parameters
    ----------
    dv_ms : float
        Along-track delta-v (m/s). Sign: + = prograde.
    dt_s : float
        Time between burn and reference time (seconds). Must be > 0.
    
    Returns
    -------
    float
        Approximate downtrack displacement at reference time (metres).
        Sign: + for prograde burn (forward displacement).
    """
    return 3.0 * dv_ms * dt_s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_maneuver(
    norad_id: int,
    dv_ms: float,
    dt_before_ref_s: float,
    ref_time: Optional[datetime] = None,
    propagate_days: Optional[float] = None,
    catalog=None,
) -> ManeuverResult:
    """
    Apply an along-track delta-v to a satellite and propagate its post-maneuver
    trajectory for the specified duration.
    
    Parameters
    ----------
    norad_id : int
        NORAD ID of the satellite.
    dv_ms : float
        Along-track delta-v (m/s). Positive = prograde, negative = retrograde.
    dt_before_ref_s : float
        Burn occurs this many seconds before ref_time (seconds).
    ref_time : datetime, optional
        Reference time (typically TCA). Default: now UTC.
    propagate_days : float, optional
        Duration to propagate post-maneuver (days). Default from config.
    catalog : pd.DataFrame, optional
        Catalog. Loaded from cache if None.
    
    Returns
    -------
    ManeuverResult
        Contains the post-maneuver trajectory and verification metrics.
    """
    from data.fetch_catalog import load_catalog, get_object_tle
    from sgp4.api import Satrec
    from core.screen import _jd_split

    cfg = _load_config()
    prop_cfg = cfg["propagation"]
    man_cfg = cfg["maneuver"]

    if ref_time is None:
        ref_time = datetime.now(timezone.utc)
    if propagate_days is None:
        propagate_days = cfg["screen"]["window_days"]
    if catalog is None:
        catalog = load_catalog()

    # Burn time
    burn_time = ref_time - timedelta(seconds=dt_before_ref_s)

    # Get TLE and propagate to burn time to get initial state
    line1, line2 = get_object_tle(norad_id, catalog)
    sat = Satrec.twoline2rv(line1, line2)

    jd_burn, fr_burn = _jd_split(burn_time)
    e, r_km, v_kms = sat.sgp4(jd_burn, fr_burn)
    if e != 0:
        raise RuntimeError(f"SGP4 propagation error {e} at burn time for NORAD {norad_id}")

    r0_m = np.array(r_km) * 1000.0
    v0_ms = np.array(v_kms) * 1000.0

    # Apply along-track impulse
    dv_vec_ms = _along_track_unit(r0_m, v0_ms) * dv_ms
    v_post_ms = v0_ms + dv_vec_ms

    # Build output time grid
    t_end_s = propagate_days * 86400.0
    step_s = cfg["screen"]["coarse_step_s"]  # 60s default
    t_eval_s = np.arange(0.0, t_end_s + step_s, step_s)
    t_eval_s = t_eval_s[t_eval_s <= t_end_s]

    # Propagate post-maneuver trajectory
    r_out, v_out = _propagate_j2(
        r0_m=r0_m,
        v0_ms=v_post_ms,
        t_span_s=(0.0, t_end_s),
        output_times_s=t_eval_s,
        mu=float(prop_cfg["mu_m3s2"]),
        re=float(prop_cfg["re_m"]),
        j2=float(prop_cfg["j2"]),
        **_integrator_settings(prop_cfg, man_cfg),
    )

    # Build datetime list for output
    times_out = [burn_time + timedelta(seconds=float(t)) for t in t_eval_s]

    # Compute downtrack offset at 0.5 orbital periods after burn.
    # At t = 0.5*T, the Hill's equation gives: delta_s ≈ 3*dv*(0.5*T)
    # and the periodic oscillation term is near its maximum.
    # This gives the most reliable analytic comparison.
    mu_loc = float(prop_cfg["mu_m3s2"])
    r_mag = float(np.linalg.norm(r0_m))
    T_orb_s = 2 * math.pi * math.sqrt(r_mag**3 / mu_loc)
    t_check_s = min(0.5 * T_orb_s, dt_before_ref_s)  # half orbital period
    ref_idx = np.argmin(np.abs(t_eval_s - t_check_s))
    t_ref = float(t_eval_s[ref_idx])

    e_ref, r_ref_km, v_ref_kms = sat.sgp4(*_jd_split(burn_time + timedelta(seconds=t_ref)))
    if e_ref == 0:
        r_unman_m_ref = np.array(r_ref_km) * 1000.0
        r_man_m_ref = r_out[ref_idx]

        # Downtrack offset using burn-time velocity direction as reference
        v_hat = v0_ms / (np.linalg.norm(v0_ms) + 1e-30)
        downtrack_m = float(np.dot(r_man_m_ref - r_unman_m_ref, v_hat))
    else:
        downtrack_m = 0.0

    # Analytic oracle: 3 * dv * t (valid at half orbital period)
    oracle_m = analytic_downtrack_drift(dv_ms, t_ref)
    ratio = downtrack_m / oracle_m if abs(oracle_m) > 1.0 else float('nan')

    return ManeuverResult(
        dv_ms=dv_ms,
        burn_time=burn_time,
        burn_r_teme_m=r0_m,
        burn_v_pre_teme_ms=v0_ms,
        burn_v_post_teme_ms=v_post_ms,
        times=times_out,
        r_teme_m=r_out,
        v_teme_ms=v_out,
        downtrack_offset_m=downtrack_m,
        analytic_oracle_m=oracle_m,
        analytic_ratio=ratio,
    )


def apply_maneuver_from_state(
    r0_m: np.ndarray,
    v0_ms: np.ndarray,
    burn_time: datetime,
    dv_ms: float,
    propagate_days: float,
) -> ManeuverResult:
    """
    Apply delta-v from a known state vector (used internally by rescreen).
    Avoids repeated SGP4 calls for the same burn time across the grid.
    
    Parameters
    ----------
    r0_m : np.ndarray, shape (3,)
        Position at burn time (metres, TEME).
    v0_ms : np.ndarray, shape (3,)
        Velocity at burn time (m/s, TEME).
    burn_time : datetime
        UTC time of burn.
    dv_ms : float
        Along-track delta-v (m/s).
    propagate_days : float
        Duration to propagate (days).
    
    Returns
    -------
    ManeuverResult
    """
    cfg = _load_config()
    prop_cfg = cfg["propagation"]
    man_cfg = cfg["maneuver"]

    dv_vec_ms = _along_track_unit(r0_m, v0_ms) * dv_ms
    v_post_ms = v0_ms + dv_vec_ms

    t_end_s = propagate_days * 86400.0
    step_s = cfg["screen"]["coarse_step_s"]
    t_eval_s = np.arange(0.0, t_end_s + step_s, step_s)
    t_eval_s = t_eval_s[t_eval_s <= t_end_s]

    r_out, v_out = _propagate_j2(
        r0_m=r0_m,
        v0_ms=v_post_ms,
        t_span_s=(0.0, t_end_s),
        output_times_s=t_eval_s,
        mu=float(prop_cfg["mu_m3s2"]),
        re=float(prop_cfg["re_m"]),
        j2=float(prop_cfg["j2"]),
        **_integrator_settings(prop_cfg, man_cfg),
    )

    times_out = [burn_time + timedelta(seconds=float(t)) for t in t_eval_s]

    return ManeuverResult(
        dv_ms=dv_ms,
        burn_time=burn_time,
        burn_r_teme_m=r0_m,
        burn_v_pre_teme_ms=v0_ms,
        burn_v_post_teme_ms=v_post_ms,
        times=times_out,
        r_teme_m=r_out,
        v_teme_ms=v_out,
        downtrack_offset_m=float("nan"),
        analytic_oracle_m=analytic_downtrack_drift(dv_ms, t_end_s),
        analytic_ratio=float("nan"),
    )
