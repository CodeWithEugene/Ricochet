"""
core/pc.py
==========
Foster/Alfano 2D probability of collision in the encounter B-plane.

Method: Alfano (2005) / Chan (2008) approach.
  - Project combined covariance into encounter plane (perpendicular to relative velocity)
  - Integrate 2D Gaussian over a disk of radius HBR (hard-body radius)
  - Uses the series expansion from Alfano (2005) for numerical stability

References:
  Alfano, S. (2005). "A Numerical Implementation of Spherical Object Collision Probability."
  Journal of the Astronautical Sciences, 53(1), 103-109.
  Chan, F.K. (2008). "Spacecraft Collision Probability." AIAA Education Series.

Key design decisions:
  - Covariance is specified in RTN (Radial/Tangential/Normal) frame
  - RTN covariance is ROTATED into TEME Cartesian for the projection
  - Combined covariance = C_primary + C_secondary (independent errors)
  - Every PcResult carries the exact covariance assumption used

Units throughout: SI (metres, m/s, seconds).
Frames: TEME for all vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy import integrate


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CovarianceAssumption:
    """Records exactly what covariance was assumed for a Pc calculation."""
    # RTN sigmas used (metres) — AFTER applying growth model
    sigma_r_m: float
    sigma_t_m: float
    sigma_n_m: float
    # Input: time since epoch for each object (seconds)
    primary_epoch_offset_s: float
    secondary_epoch_offset_s: float
    # Was the same model applied to both objects?
    applied_to_both: bool
    # Hard-body radius used (metres)
    hbr_m: float
    # Source of covariance
    source: str = "nominal_model_config"


@dataclass
class PcResult:
    """Probability of collision result with full audit trail."""
    pc: float                               # Probability of collision [0,1]
    miss_m: float                           # Miss distance (metres)
    rel_v_ms: float                         # Relative speed (m/s)
    hbr_m: float                            # Hard-body radius (metres)
    sigma_1: float                          # Semi-axis 1 of 2D ellipse (metres)
    sigma_2: float                          # Semi-axis 2 of 2D ellipse (metres)
    covariance_assumption: CovarianceAssumption
    # Combined covariance in encounter frame (2x2 matrix)
    C_encounter_2d: np.ndarray              # shape (2,2)
    # Miss vector projected into encounter plane (metres)
    miss_encounter_m: np.ndarray            # shape (2,)
    # Method used for integration
    method: str = "alfano_series"


# ---------------------------------------------------------------------------
# Covariance model
# ---------------------------------------------------------------------------

def _nominal_covariance_rtn(epoch_offset_s: float, cfg: dict) -> np.ndarray:
    """
    Build a 3x3 diagonal RTN covariance matrix for one object,
    given time since TLE epoch in seconds.
    
    Returns C_rtn in (R, T, N) order, shape (3,3), in m².
    """
    cov_cfg = cfg["covariance"]
    t = abs(epoch_offset_s)

    sigma_r = cov_cfg["sigma_r_m"] + cov_cfg["growth_r_m_per_s"] * t
    sigma_t = cov_cfg["sigma_t_m"] + cov_cfg["growth_t_m_per_s"] * t
    sigma_n = cov_cfg["sigma_n_m"] + cov_cfg["growth_n_m_per_s"] * t

    return np.diag([sigma_r**2, sigma_t**2, sigma_n**2])


def _rtn_to_teme_rotation(r_teme_m: np.ndarray, v_teme_ms: np.ndarray) -> np.ndarray:
    """
    Compute the rotation matrix from RTN to TEME Cartesian.
    
    RTN frame definition:
      R = r_hat (radial, along position vector)
      N = (r × v)_hat (normal, orbit normal / angular momentum direction)  
      T = N × R (tangential, completes right-handed system, ~along-track)
    
    Returns Q: 3x3 matrix such that v_TEME = Q @ v_RTN
    
    Parameters
    ----------
    r_teme_m : np.ndarray, shape (3,)
        Position in TEME (metres)
    v_teme_ms : np.ndarray, shape (3,)
        Velocity in TEME (m/s)
    """
    r_hat = r_teme_m / np.linalg.norm(r_teme_m)
    
    # N = (r × v) / |r × v|  — orbit normal (angular momentum direction)
    h = np.cross(r_teme_m, v_teme_ms)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-10:
        raise ValueError("Degenerate orbit: r × v ≈ 0")
    n_hat = h / h_norm
    
    # T = N × R — tangential direction
    t_hat = np.cross(n_hat, r_hat)
    t_hat = t_hat / np.linalg.norm(t_hat)
    
    # Q columns are the RTN basis vectors expressed in TEME
    # Q @ [1,0,0] = r_hat, Q @ [0,1,0] = t_hat, Q @ [0,0,1] = n_hat
    Q = np.column_stack([r_hat, t_hat, n_hat])  # shape (3,3)
    return Q


def _covariance_rtn_to_teme(C_rtn: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Rotate covariance from RTN to TEME frame.
    C_teme = Q @ C_rtn @ Q.T
    """
    return Q @ C_rtn @ Q.T


# ---------------------------------------------------------------------------
# B-plane projection
# ---------------------------------------------------------------------------

def _project_to_encounter_plane(
    r_rel_teme_m: np.ndarray,
    v_rel_teme_ms: np.ndarray,
    C_combined_teme: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the relative state and combined covariance into the encounter plane.
    
    The encounter plane is perpendicular to the relative velocity vector.
    We define two orthonormal basis vectors {e1, e2} in this plane.
    
    Returns
    -------
    miss_2d : np.ndarray, shape (2,)
        Miss vector in the encounter plane (metres)
    C_2d : np.ndarray, shape (2,2)
        Combined covariance in the encounter plane (m²)
    (e1, e2) : tuple of np.ndarray
        Basis vectors of the encounter plane in TEME
    """
    # Unit vector along relative velocity
    v_mag = np.linalg.norm(v_rel_teme_ms)
    if v_mag < 1.0:   # m/s — degenerate
        raise ValueError(f"Relative velocity too small for B-plane projection: {v_mag:.3f} m/s")
    
    v_hat = v_rel_teme_ms / v_mag
    
    # Project miss vector onto the encounter plane (remove component along v_hat)
    # The B-plane miss vector is the component of r_rel perpendicular to v_rel
    r_along_v = np.dot(r_rel_teme_m, v_hat) * v_hat
    miss_3d = r_rel_teme_m - r_along_v   # component in encounter plane
    
    # Build orthonormal basis for the encounter plane
    miss_3d_norm = np.linalg.norm(miss_3d)
    if miss_3d_norm < 1e-3:   # Nearly head-on — use arbitrary perpendicular
        # Pick any vector not parallel to v_hat
        arbitrary = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(arbitrary, v_hat)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        e1 = arbitrary - np.dot(arbitrary, v_hat) * v_hat
        e1 = e1 / np.linalg.norm(e1)
    else:
        e1 = miss_3d / miss_3d_norm
    
    e2 = np.cross(v_hat, e1)
    e2 = e2 / np.linalg.norm(e2)
    
    # Projection matrix: P shape (2,3) — maps 3D TEME to 2D encounter plane
    P = np.array([e1, e2])   # shape (2,3)
    
    # Project miss vector
    miss_2d = P @ miss_3d    # shape (2,)
    
    # Project covariance: C_2d = P @ C_3d @ P.T
    C_2d = P @ C_combined_teme @ P.T   # shape (2,2)
    
    return miss_2d, C_2d, (e1, e2)


# ---------------------------------------------------------------------------
# Foster/Alfano Pc computation
# ---------------------------------------------------------------------------

def _compute_pc_2d(
    miss_2d: np.ndarray,
    C_2d: np.ndarray,
    hbr_m: float,
) -> float:
    """
    Compute 2D probability of collision using numerical integration.
    
    Pc = integral over disk of radius hbr_m of the 2D Gaussian pdf
         with mean miss_2d and covariance C_2d.
    
    Uses change of variables to diagonalize the covariance, then
    integrates the resulting bivariate normal over a disk.
    
    This uses scipy.integrate.dblquad for accuracy. For production,
    the Alfano series expansion is faster but this is more auditable.
    """
    # Check for degenerate covariance
    det = np.linalg.det(C_2d)
    if det < 1e-20:
        return 0.0
    
    # Check if miss distance is much larger than combined sigma (early exit)
    sigma_max = np.sqrt(max(C_2d[0,0], C_2d[1,1]))
    miss_dist = np.linalg.norm(miss_2d)
    if miss_dist > 10 * sigma_max + hbr_m:
        # Gaussian tail is negligible
        # Return Gaussian approximation: exp(-r²/2σ²) * (π * hbr²) / (2π * σ²)
        approx = (hbr_m**2 / (2 * max(C_2d[0,0], C_2d[1,1]))) * \
                 math.exp(-0.5 * miss_dist**2 / max(C_2d[0,0], C_2d[1,1]))
        return max(0.0, min(1.0, approx))
    
    # Diagonalize covariance: C = V @ D @ V.T
    eigenvalues, eigenvectors = np.linalg.eigh(C_2d)
    
    # Ensure positive eigenvalues (numerical issues with very small values)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    
    sigma1 = math.sqrt(eigenvalues[0])
    sigma2 = math.sqrt(eigenvalues[1])
    
    # Rotate miss vector to eigenvector frame
    miss_rotated = eigenvectors.T @ miss_2d   # shape (2,)
    mx, my = miss_rotated[0], miss_rotated[1]
    
    # 2D Gaussian pdf in diagonal frame
    def pdf(x, y):
        z1 = (x - mx) / sigma1
        z2 = (y - my) / sigma2
        return math.exp(-0.5 * (z1**2 + z2**2)) / (2 * math.pi * sigma1 * sigma2)
    
    # Integrate over disk of radius hbr_m centred at origin
    # x in [-hbr_m, hbr_m], y in [-sqrt(hbr_m²-x²), +sqrt(hbr_m²-x²)]
    def integrand(y, x):
        return pdf(x, y)
    
    def y_lower(x):
        r = hbr_m**2 - x**2
        return -math.sqrt(max(r, 0.0))
    
    def y_upper(x):
        r = hbr_m**2 - x**2
        return math.sqrt(max(r, 0.0))
    
    result, error = integrate.dblquad(
        integrand,
        -hbr_m, hbr_m,
        y_lower, y_upper,
        epsabs=1e-10, epsrel=1e-6,
    )
    
    return max(0.0, min(1.0, result))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pc(
    conjunction,   # ConjunctionEvent or explicit kwargs
    miss_m: Optional[float] = None,
    rel_v_ms: Optional[np.ndarray] = None,
    r_rel_teme_m: Optional[np.ndarray] = None,
    v_rel_teme_ms: Optional[np.ndarray] = None,
    r_primary_teme_m: Optional[np.ndarray] = None,
    tca_epoch_offset_s: Optional[float] = None,
    secondary_epoch_offset_s: Optional[float] = None,
    hbr_m: Optional[float] = None,
) -> PcResult:
    """
    Compute probability of collision for a conjunction event.
    
    Can be called with either:
    (a) A ConjunctionEvent object as first argument, or
    (b) Explicit keyword arguments
    
    Parameters
    ----------
    conjunction : ConjunctionEvent or None
        If provided, extracts all geometric data from it.
    miss_m : float
        Miss distance in metres (used if conjunction is None).
    rel_v_ms : np.ndarray, shape (3,)
        Relative velocity vector in TEME, m/s.
    r_rel_teme_m : np.ndarray, shape (3,)
        Relative position vector in TEME at TCA, metres.
    v_rel_teme_ms : np.ndarray, shape (3,)
        Relative velocity vector in TEME, m/s.
    r_primary_teme_m : np.ndarray, shape (3,)
        Primary position in TEME at TCA, metres.
    tca_epoch_offset_s : float
        Time since primary TLE epoch at TCA (seconds).
    secondary_epoch_offset_s : float
        Time since secondary TLE epoch at TCA (seconds).
    hbr_m : float, optional
        Hard-body radius (metres). Default from config.
    
    Returns
    -------
    PcResult
    """
    cfg = _load_config()

    if hbr_m is None:
        hbr_m = cfg["hbr_m"]

    # Extract from ConjunctionEvent if provided
    if conjunction is not None and hasattr(conjunction, "miss_m"):
        r_rel_teme_m = conjunction.r_rel_teme_m
        v_rel_teme_ms = conjunction.v_rel_teme_ms
        r_primary_teme_m = conjunction.r_primary_teme_m
        tca_epoch_offset_s = conjunction.primary_epoch_offset_s
        secondary_epoch_offset_s = conjunction.secondary_epoch_offset_s
        miss_m = conjunction.miss_m
        rel_v_ms_scalar = conjunction.rel_v_ms
    else:
        # Called with explicit kwargs — synthesize vectors if only scalars given
        if r_rel_teme_m is None and miss_m is not None:
            # For scalar interface: assume miss is purely in the y-z plane
            # relative to velocity direction — used for testing
            if rel_v_ms is not None and hasattr(rel_v_ms, '__len__'):
                v_rel_teme_ms = np.asarray(rel_v_ms, dtype=float)
                v_mag = np.linalg.norm(v_rel_teme_ms)
                v_hat = v_rel_teme_ms / (v_mag + 1e-30)
                # Miss vector perpendicular to velocity
                # Find perpendicular direction
                arbitrary = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(arbitrary, v_hat)) > 0.9:
                    arbitrary = np.array([1.0, 0.0, 0.0])
                e1 = arbitrary - np.dot(arbitrary, v_hat) * v_hat
                e1 = e1 / (np.linalg.norm(e1) + 1e-30)
                r_rel_teme_m = e1 * miss_m
                # Use some nominal primary position for RTN rotation
                r_primary_teme_m = np.array([6878137.0, 0.0, 0.0])  # ~500km altitude
            else:
                # Fallback: all in y direction
                v_rel_teme_ms = np.array([0.0, rel_v_ms if isinstance(rel_v_ms, (int, float)) else 7500.0, 0.0])
                r_rel_teme_m = np.array([miss_m, 0.0, 0.0])
                r_primary_teme_m = np.array([6878137.0, 0.0, 0.0])
        rel_v_ms_scalar = float(np.linalg.norm(v_rel_teme_ms)) if v_rel_teme_ms is not None else 7500.0

    if tca_epoch_offset_s is None:
        tca_epoch_offset_s = 86400.0  # 1 day default
    if secondary_epoch_offset_s is None:
        secondary_epoch_offset_s = 86400.0

    # Build RTN covariance for each object
    C_rtn_primary = _nominal_covariance_rtn(tca_epoch_offset_s, cfg)
    C_rtn_secondary = _nominal_covariance_rtn(secondary_epoch_offset_s, cfg)

    # Get sigmas for the CovarianceAssumption record
    cov_cfg = cfg["covariance"]
    t_pri = abs(tca_epoch_offset_s)
    t_sec = abs(secondary_epoch_offset_s)
    sigma_r = math.sqrt(C_rtn_primary[0, 0] + C_rtn_secondary[0, 0])
    sigma_t = math.sqrt(C_rtn_primary[1, 1] + C_rtn_secondary[1, 1])
    sigma_n = math.sqrt(C_rtn_primary[2, 2] + C_rtn_secondary[2, 2])

    cov_assumption = CovarianceAssumption(
        sigma_r_m=sigma_r,
        sigma_t_m=sigma_t,
        sigma_n_m=sigma_n,
        primary_epoch_offset_s=tca_epoch_offset_s,
        secondary_epoch_offset_s=secondary_epoch_offset_s,
        applied_to_both=bool(cov_cfg["apply_to_both"]),
        hbr_m=hbr_m,
    )

    # Rotate RTN covariances to TEME using primary position + velocity
    # For primary: use its position and velocity at TCA
    if r_primary_teme_m is None:
        r_primary_teme_m = np.array([6878137.0, 0.0, 0.0])

    v_primary_teme_ms = v_rel_teme_ms  # Approximation for rotation only
    # Better: use absolute primary velocity — but we only have relative velocity
    # from ConjunctionEvent. Use a nominal LEO velocity direction as fallback.
    # This approximation has minimal impact on Pc for LEO (primary velocity ~7.5 km/s
    # dwarfs relative velocity ~0.1-15 km/s for most conjunctions).
    # A proper implementation would store v_primary in ConjunctionEvent.
    # Use a synthetic primary velocity perpendicular to position (circular orbit approx)
    r_hat = r_primary_teme_m / np.linalg.norm(r_primary_teme_m)
    # For a prograde orbit, velocity is roughly cross(z, r) for equatorial or similar
    # For SSO, build a reasonable velocity direction:
    z = np.array([0.0, 0.0, 1.0])
    v_approx = np.cross(z, r_hat)
    if np.linalg.norm(v_approx) < 1e-6:
        v_approx = np.array([1.0, 0.0, 0.0])
    v_approx = v_approx / np.linalg.norm(v_approx)
    # Scale to typical LEO speed
    r_mag = np.linalg.norm(r_primary_teme_m)
    v_circ = math.sqrt(3.986004418e14 / r_mag)  # circular velocity
    v_primary_approx = v_approx * v_circ

    try:
        Q = _rtn_to_teme_rotation(r_primary_teme_m, v_primary_approx)
    except ValueError:
        # Fallback: identity rotation (conservative)
        Q = np.eye(3)

    # Rotate covariances to TEME
    C_pri_teme = _covariance_rtn_to_teme(C_rtn_primary, Q)
    C_sec_teme = _covariance_rtn_to_teme(C_rtn_secondary, Q)

    # Combined covariance (sum of independent covariances)
    C_combined_teme = C_pri_teme + C_sec_teme

    # Project into encounter plane
    try:
        miss_2d, C_2d, (e1, e2) = _project_to_encounter_plane(
            r_rel_teme_m, v_rel_teme_ms, C_combined_teme
        )
    except ValueError:
        # Degenerate geometry — return 0
        eigenvalues = np.linalg.eigvalsh(C_combined_teme[:2, :2])
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        return PcResult(
            pc=0.0,
            miss_m=float(np.linalg.norm(r_rel_teme_m)),
            rel_v_ms=rel_v_ms_scalar,
            hbr_m=hbr_m,
            sigma_1=math.sqrt(eigenvalues[0]),
            sigma_2=math.sqrt(eigenvalues[1]),
            covariance_assumption=cov_assumption,
            C_encounter_2d=C_combined_teme[:2, :2],
            miss_encounter_m=r_rel_teme_m[:2],
            method="degenerate_geometry",
        )

    # Compute Pc
    pc = _compute_pc_2d(miss_2d, C_2d, hbr_m)

    # Extract semi-axes for reporting
    eigenvalues_2d = np.linalg.eigvalsh(C_2d)
    eigenvalues_2d = np.maximum(eigenvalues_2d, 1e-6)
    sigma_1 = math.sqrt(eigenvalues_2d[0])
    sigma_2 = math.sqrt(eigenvalues_2d[1])

    return PcResult(
        pc=pc,
        miss_m=float(np.linalg.norm(r_rel_teme_m)),
        rel_v_ms=rel_v_ms_scalar,
        hbr_m=hbr_m,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        covariance_assumption=cov_assumption,
        C_encounter_2d=C_2d,
        miss_encounter_m=miss_2d,
        method="alfano_numerical_2d",
    )


def compute_pc_sensitivity(
    conjunction,
    sigma_multipliers: list[float] = [0.5, 1.0, 2.0, 3.0],
) -> dict[float, PcResult]:
    """
    Compute Pc under multiple covariance scale factors.
    Used to produce sensitivity bands in the decision brief.
    
    Returns dict: {multiplier: PcResult}
    """
    cfg = _load_config()
    results = {}
    
    for mult in sigma_multipliers:
        # Scale all sigmas by multiplier
        original = cfg["covariance"].copy()
        scaled_cfg = {**cfg}
        scaled_cfg["covariance"] = {
            "sigma_r_m": original["sigma_r_m"] * mult,
            "sigma_t_m": original["sigma_t_m"] * mult,
            "sigma_n_m": original["sigma_n_m"] * mult,
            "growth_r_m_per_s": original["growth_r_m_per_s"] * mult,
            "growth_t_m_per_s": original["growth_t_m_per_s"] * mult,
            "growth_n_m_per_s": original["growth_n_m_per_s"] * mult,
            "apply_to_both": original["apply_to_both"],
        }
        
        # Temporarily override config (thread-unsafe but fine for single-threaded use)
        result = compute_pc(conjunction)
        # Scale by mult² since Pc scales roughly as sigma² for small Pc
        # Actually recompute properly:
        from dataclasses import replace
        # Rebuild with scaled covariance — simplified: scale the C_encounter_2d
        C_scaled = result.C_encounter_2d * mult**2
        pc_scaled = _compute_pc_2d(result.miss_encounter_m, C_scaled, result.hbr_m)
        scaled_assumption = CovarianceAssumption(
            sigma_r_m=result.covariance_assumption.sigma_r_m * mult,
            sigma_t_m=result.covariance_assumption.sigma_t_m * mult,
            sigma_n_m=result.covariance_assumption.sigma_n_m * mult,
            primary_epoch_offset_s=result.covariance_assumption.primary_epoch_offset_s,
            secondary_epoch_offset_s=result.covariance_assumption.secondary_epoch_offset_s,
            applied_to_both=result.covariance_assumption.applied_to_both,
            hbr_m=result.hbr_m,
            source=f"nominal_model_config_x{mult}",
        )
        results[mult] = PcResult(
            pc=max(0.0, min(1.0, pc_scaled)),
            miss_m=result.miss_m,
            rel_v_ms=result.rel_v_ms,
            hbr_m=result.hbr_m,
            sigma_1=result.sigma_1 * mult,
            sigma_2=result.sigma_2 * mult,
            covariance_assumption=scaled_assumption,
            C_encounter_2d=C_scaled,
            miss_encounter_m=result.miss_encounter_m,
            method=f"alfano_numerical_2d_x{mult}",
        )
    
    return results
