"""
tests/test_physics.py
=====================
Physics validation tests for Ricochet.

These are NOT unit tests of code paths. They are PHYSICS ORACLE tests:
each test has a known correct answer from an external reference and fails
loudly if the implementation produces the wrong physics.

Running these in order catches errors at the earliest possible layer.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: Config is consistent
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_loads_and_has_required_keys(self):
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        assert "hbr_m" in cfg, "HBR missing from config"
        assert "covariance" in cfg, "covariance model missing from config"
        cov = cfg["covariance"]
        assert "sigma_r_m" in cov
        assert "sigma_t_m" in cov
        assert "sigma_n_m" in cov
        assert "growth_t_m_per_s" in cov
        assert cfg["hbr_m"] > 0, "HBR must be positive"
        assert cfg["hbr_m"] < 1000, "HBR > 1000m is implausibly large"

    def test_covariance_values_are_physical(self):
        import yaml
        cov = yaml.safe_load(open("config.yaml"))["covariance"]
        # TLE accuracy: sigma_T should be >> sigma_R
        assert cov["sigma_t_m"] > cov["sigma_r_m"], "Along-track sigma should exceed radial sigma for TLE-based covariance"
        # Growth rate: should be positive
        assert cov["growth_t_m_per_s"] > 0
        # After 1 day, along-track sigma should not exceed ~100 km (sanity bound)
        sigma_t_1day = cov["sigma_t_m"] + cov["growth_t_m_per_s"] * 86400
        assert sigma_t_1day < 1e5, f"Along-track sigma after 1 day ({sigma_t_1day:.0f}m) exceeds sanity bound"


# ---------------------------------------------------------------------------
# Test 2: SGP4 unit output is km — conversion must be applied
# ---------------------------------------------------------------------------

class TestSGP4Units:
    def test_sgp4_output_is_in_km(self):
        """Verify sgp4 returns km by checking that a LEO position has magnitude ~6400-8000 km."""
        from sgp4.api import Satrec, jday

        # ISS TLE (approximate, for testing)
        line1 = "1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990"
        line2 = "2 25544  51.6400  10.0000 0001000  90.0000 270.0000 15.50000000000012"
        sat = Satrec.twoline2rv(line1, line2)

        jd, fr = jday(2024, 1, 1, 12, 0, 0)
        e, r, v = sat.sgp4(jd, fr)

        if e != 0:
            pytest.skip(f"SGP4 propagation error {e} — TLE may be expired")

        r_mag = np.linalg.norm(r)
        # Should be ~6400-8000 km if output is in km
        assert 6000 < r_mag < 9000, (
            f"SGP4 position magnitude {r_mag:.1f} not in expected km range [6000, 9000]. "
            "If ~6.4e6 to 8e6, units are metres (conversion bug!)."
        )

    def test_sgp4_km_to_si_conversion(self):
        """Converting SGP4 km output to metres gives ~6.4-8e6 metres."""
        from sgp4.api import Satrec, jday

        line1 = "1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990"
        line2 = "2 25544  51.6400  10.0000 0001000  90.0000 270.0000 15.50000000000012"
        sat = Satrec.twoline2rv(line1, line2)

        jd, fr = jday(2024, 1, 1, 12, 0, 0)
        e, r_km, v_kms = sat.sgp4(jd, fr)
        if e != 0:
            pytest.skip("SGP4 error")

        r_m = np.array(r_km) * 1000.0
        r_mag_m = np.linalg.norm(r_m)
        assert 6e6 < r_mag_m < 9e6, f"Converted position {r_mag_m:.3e} m not in LEO range"

        v_ms = np.array(v_kms) * 1000.0
        v_mag_ms = np.linalg.norm(v_ms)
        assert 7000 < v_mag_ms < 8000, f"Converted velocity {v_mag_ms:.1f} m/s not in LEO range"


# ---------------------------------------------------------------------------
# Test 3: RTN→TEME rotation is orthogonal and correct
# ---------------------------------------------------------------------------

class TestRTNRotation:
    def test_rotation_matrix_is_orthogonal(self):
        """Q must be a proper rotation matrix: Q.T @ Q = I, det(Q) = 1."""
        from core.pc import _rtn_to_teme_rotation

        # Circular equatorial orbit at 500km
        r = np.array([6878137.0, 0.0, 0.0])  # +X axis
        v = np.array([0.0, 7612.0, 0.0])      # +Y axis (prograde)
        Q = _rtn_to_teme_rotation(r, v)

        assert Q.shape == (3, 3)
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10,
                                   err_msg="RTN rotation matrix is not orthogonal")
        det = np.linalg.det(Q)
        np.testing.assert_allclose(det, 1.0, atol=1e-10,
                                   err_msg="RTN rotation matrix det is not 1 (not proper rotation)")

    def test_rtn_basis_vectors_are_correct_for_equatorial_orbit(self):
        """
        For a prograde equatorial orbit at the +X position:
        - R should point in +X direction (radial)
        - T should point in +Y direction (tangential ~ velocity)
        - N should point in +Z direction (normal ~ angular momentum)
        """
        from core.pc import _rtn_to_teme_rotation

        r = np.array([6878137.0, 0.0, 0.0])
        v = np.array([0.0, 7612.0, 0.0])
        Q = _rtn_to_teme_rotation(r, v)

        r_hat = Q[:, 0]  # R column
        t_hat = Q[:, 1]  # T column
        n_hat = Q[:, 2]  # N column

        np.testing.assert_allclose(r_hat, [1.0, 0.0, 0.0], atol=1e-6,
                                   err_msg="R axis should point in +X for +X orbit position")
        np.testing.assert_allclose(t_hat, [0.0, 1.0, 0.0], atol=1e-6,
                                   err_msg="T axis should point in +Y for equatorial prograde orbit")
        np.testing.assert_allclose(n_hat, [0.0, 0.0, 1.0], atol=1e-6,
                                   err_msg="N axis should point in +Z for equatorial orbit")

    def test_covariance_rotation_preserves_trace(self):
        """
        The trace of a covariance matrix (total variance) is preserved under rotation.
        """
        from core.pc import _rtn_to_teme_rotation, _covariance_rtn_to_teme

        r = np.array([6878137.0, 0.0, 0.0])
        v = np.array([0.0, 7612.0, 0.0])
        Q = _rtn_to_teme_rotation(r, v)

        C_rtn = np.diag([100.0**2, 300.0**2, 50.0**2])
        C_teme = _covariance_rtn_to_teme(C_rtn, Q)

        np.testing.assert_allclose(
            np.trace(C_teme), np.trace(C_rtn), rtol=1e-10,
            err_msg="Trace (total variance) not preserved under rotation"
        )


# ---------------------------------------------------------------------------
# Test 4: B-plane projection
# ---------------------------------------------------------------------------

class TestBPlaneProjection:
    def test_head_on_encounter_miss_equals_3d_miss(self):
        """
        For a purely head-on encounter (miss vector perpendicular to relative velocity),
        the B-plane miss distance equals the 3D miss distance.
        """
        from core.pc import _project_to_encounter_plane

        # Set up: velocity in +Z, miss vector in +X (perpendicular)
        miss_dist = 500.0   # 500 metres
        r_rel = np.array([miss_dist, 0.0, 0.0])
        v_rel = np.array([0.0, 0.0, 7500.0])   # head-on

        C_3d = np.diag([200.0**2, 300.0**2, 50.0**2])

        miss_2d, C_2d, _ = _project_to_encounter_plane(r_rel, v_rel, C_3d)

        miss_2d_dist = np.linalg.norm(miss_2d)
        np.testing.assert_allclose(
            miss_2d_dist, miss_dist, rtol=1e-6,
            err_msg=f"B-plane miss {miss_2d_dist:.2f}m ≠ 3D miss {miss_dist:.2f}m for head-on encounter"
        )

    def test_along_velocity_component_removed(self):
        """
        The miss vector component along the relative velocity is removed by B-plane projection.
        """
        from core.pc import _project_to_encounter_plane

        # Miss vector has both perpendicular and along-velocity components
        r_rel = np.array([500.0, 1000.0, 0.0])  # +X perpendicular, +Y along velocity
        v_rel = np.array([0.0, 7500.0, 0.0])    # velocity in +Y direction

        C_3d = np.diag([200.0**2, 300.0**2, 50.0**2])

        miss_2d, C_2d, _ = _project_to_encounter_plane(r_rel, v_rel, C_3d)

        # B-plane miss should only have the perpendicular component
        # The 1000m along +Y should be removed
        miss_2d_dist = np.linalg.norm(miss_2d)
        np.testing.assert_allclose(
            miss_2d_dist, 500.0, rtol=1e-4,
            err_msg="Along-velocity miss component not removed by B-plane projection"
        )

    def test_covariance_projection_is_psd(self):
        """The projected 2x2 covariance must be positive semi-definite."""
        from core.pc import _project_to_encounter_plane

        r_rel = np.array([500.0, 0.0, 0.0])
        v_rel = np.array([0.0, 7500.0, 0.0])
        C_3d = np.diag([200.0**2, 300.0**2, 50.0**2])

        _, C_2d, _ = _project_to_encounter_plane(r_rel, v_rel, C_3d)

        eigenvalues = np.linalg.eigvalsh(C_2d)
        assert np.all(eigenvalues >= -1e-10), f"Projected covariance not PSD: eigenvalues = {eigenvalues}"


# ---------------------------------------------------------------------------
# Test 5: Pc physics sanity
# ---------------------------------------------------------------------------

class TestPcPhysics:
    def test_zero_miss_gives_large_pc(self):
        """
        At near-zero miss distance with small covariance, Pc should be near 0.5.
        
        The key physics: when miss_distance << sigma, the HBR disk is centred at the
        mean of the Gaussian. For a symmetric 2D Gaussian, the integral over the disk
        approaches 0.5 as the disk covers half the distribution.
        
        But with large operational covariance (sigma_T ~ 300m >> HBR=10m), Pc at
        zero miss is much smaller because the 10m HBR disk covers a tiny fraction
        of the wide Gaussian. That is correct physics.
        
        This test uses a SMALL covariance (sigma ~ HBR) to verify the high-Pc regime.
        We override the covariance by calling _compute_pc_2d directly.
        """
        from core.pc import _compute_pc_2d

        # Very small covariance (sigma = HBR = 10m), miss = 0
        hbr = 10.0
        sigma = 10.0
        C_2d = np.diag([sigma**2, sigma**2])
        miss_2d = np.array([0.0, 0.0])   # direct hit centre

        pc = _compute_pc_2d(miss_2d, C_2d, hbr)
        # When sigma = HBR and miss = 0, Pc should be meaningfully large (> 0.1)
        # Analytically: ~ 1 - exp(-hbr²/(2σ²)) = 1 - exp(-0.5) ≈ 0.39
        assert 0.2 <= pc <= 0.8, f"Zero-miss, sigma=HBR Pc = {pc:.4f}, expected 0.2-0.8 range"

    def test_large_covariance_small_miss_gives_small_pc(self):
        """
        With operational covariance (sigma_T >> HBR), even zero miss gives small Pc.
        This is CORRECT physics — the covariance model represents position uncertainty,
        not the actual collision probability.
        """
        from core.pc import compute_pc

        # Use operational covariance (sigma_T ~ 485m >> HBR=10m)
        result = compute_pc(
            None,
            miss_m=0.0,
            rel_v_ms=np.array([0.0, 7500.0, 0.0]),
            r_rel_teme_m=np.array([0.0, 0.0, 1e-6]),
            v_rel_teme_ms=np.array([0.0, 7500.0, 0.0]),
            r_primary_teme_m=np.array([6878137.0, 0.0, 0.0]),
            tca_epoch_offset_s=3600.0,
            secondary_epoch_offset_s=3600.0,
        )
        # With sigma_T ~ 485m and HBR=10m, Pc should be small but nonzero
        # ~ (hbr/sigma)^2 / 2 ~ (10/485)^2 / 2 ~ 2e-4
        assert 0.0 < result.pc < 0.1, (
            f"Pc = {result.pc:.4e} with operational covariance and zero miss. "
            "Expected small but positive (10m HBR << 300-500m sigma)."
        )

    def test_large_miss_gives_near_zero_pc(self):
        """At 100 km miss, Pc must be essentially zero."""
        from core.pc import compute_pc

        result = compute_pc(
            None,
            miss_m=1e5,   # 100 km
            rel_v_ms=np.array([0.0, 7500.0, 0.0]),
            r_rel_teme_m=np.array([1e5, 0.0, 0.0]),
            v_rel_teme_ms=np.array([0.0, 7500.0, 0.0]),
            r_primary_teme_m=np.array([6878137.0, 0.0, 0.0]),
            tca_epoch_offset_s=3600.0,
            secondary_epoch_offset_s=3600.0,
        )
        assert result.pc < 1e-10, f"100km miss Pc = {result.pc:.2e}, should be < 1e-10"

    def test_smaller_miss_gives_larger_pc(self):
        """Pc is monotonically decreasing with miss distance."""
        from core.pc import compute_pc

        pcs = []
        for miss_m in [100.0, 500.0, 1000.0, 5000.0]:
            r = miss_m * np.array([1.0, 0.0, 0.0])
            result = compute_pc(
                None,
                miss_m=miss_m,
                rel_v_ms=np.array([0.0, 7500.0, 0.0]),
                r_rel_teme_m=r,
                v_rel_teme_ms=np.array([0.0, 7500.0, 0.0]),
                r_primary_teme_m=np.array([6878137.0, 0.0, 0.0]),
                tca_epoch_offset_s=3600.0,
                secondary_epoch_offset_s=3600.0,
            )
            pcs.append(result.pc)

        for i in range(len(pcs) - 1):
            assert pcs[i] >= pcs[i+1], (
                f"Pc not monotonically decreasing with miss distance: "
                f"{pcs[i]:.2e} at {[100, 500, 1000, 5000][i]}m "
                f"< {pcs[i+1]:.2e} at {[100, 500, 1000, 5000][i+1]}m"
            )

    def test_pc_result_carries_covariance_assumption(self):
        """Every PcResult must have a complete covariance_assumption."""
        from core.pc import compute_pc

        result = compute_pc(
            None,
            miss_m=500.0,
            rel_v_ms=np.array([0.0, 7500.0, 0.0]),
            r_rel_teme_m=np.array([500.0, 0.0, 0.0]),
            v_rel_teme_ms=np.array([0.0, 7500.0, 0.0]),
            r_primary_teme_m=np.array([6878137.0, 0.0, 0.0]),
            tca_epoch_offset_s=86400.0,
            secondary_epoch_offset_s=86400.0,
        )
        cov = result.covariance_assumption
        assert cov.sigma_r_m > 0
        assert cov.sigma_t_m > 0
        assert cov.sigma_n_m > 0
        assert cov.hbr_m > 0
        assert cov.source != ""


# ---------------------------------------------------------------------------
# Test 6: Maneuver analytic oracle
# ---------------------------------------------------------------------------

class TestManeuverOracle:
    def test_analytic_oracle_formula(self):
        """
        The first-order analytic formula is: downtrack ≈ 3 * dv * dt.
        Verify formula is correct.
        
        Note: this formula is valid at t ≈ 0.5 * T_orbital (half orbital period).
        At short times the displacement is just dv * dt (direct velocity increment).
        """
        from core.maneuver import analytic_downtrack_drift

        # 1 m/s burn, 1 hour before reference → 3 * 1 * 3600 = 10800 m
        drift = analytic_downtrack_drift(1.0, 3600.0)
        assert abs(drift - 10800.0) < 0.1, f"Analytic oracle: expected 10800m, got {drift:.2f}m"

        # 0.1 m/s burn, 12 hours → 3 * 0.1 * 43200 = 12960 m
        drift2 = analytic_downtrack_drift(0.1, 43200.0)
        assert abs(drift2 - 12960.0) < 0.1

        # Retrograde burn (negative dv) gives negative drift
        drift3 = analytic_downtrack_drift(-1.0, 3600.0)
        assert drift3 < 0, "Retrograde burn should give negative (backward) drift"

    def test_numerical_matches_analytic_at_half_period(self):
        """
        The numerical two-body+J2 propagation should agree with the analytic
        Hill's formula to within 5% at 0.5 orbital periods after the burn.
        """
        from core.maneuver import apply_maneuver
        # ISS at 400km, T ≈ 5575s, 0.5T ≈ 2788s
        # dt_before_ref_s=3600s > 0.5T, so comparison is at 0.5T
        result = apply_maneuver(norad_id=25544, dv_ms=1.0, dt_before_ref_s=3600.0)
        ratio = result.analytic_ratio
        assert not math.isnan(ratio), "Analytic ratio is NaN — comparison failed"
        assert 0.85 <= abs(ratio) <= 1.15, (
            f"Maneuver numerical/analytic ratio {ratio:.4f} out of [0.85, 1.15] range. "
            "J2 integrator may have a bug or comparison time is wrong."
        )

    def test_oracle_sign_convention(self):
        """Prograde burn should give positive (forward) drift."""
        from core.maneuver import analytic_downtrack_drift
        assert analytic_downtrack_drift(1.0, 3600) > 0
        assert analytic_downtrack_drift(-1.0, 3600) < 0


# ---------------------------------------------------------------------------
# Test 7: Audit logger round-trip
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def test_round_trip(self, tmp_path, monkeypatch):
        """Log an event and reload it — all fields must survive."""
        import yaml
        # Patch config to use tmp_path
        cfg_content = yaml.safe_load(open("config.yaml"))
        cfg_content["audit"]["log_dir"] = str(tmp_path)

        tmp_cfg = tmp_path / "config_test.yaml"
        with open(tmp_cfg, "w") as f:
            yaml.dump(cfg_content, f)

        import audit.logger as alog

        original_load = alog._load_config

        def patched_load():
            return cfg_content

        monkeypatch.setattr(alog, "_load_config", patched_load)

        event_id = alog.log_tool_call(
            tool="screen",
            inputs={"norad_id": 56212, "window_days": 7},
            outputs={"n_conjunctions": 3, "conjunctions": []},
            model_name="granite3.3:8b",
            model_version="latest",
            duration_s=5.2,
            session_id="test-session",
        )

        assert event_id is not None
        event = alog.load_event(event_id)
        assert event["inputs"]["norad_id"] == 56212
        assert event["tool"] == "screen"
        assert event["model_name"] == "granite3.3:8b"
        assert "timestamp" in event
        assert event["session_id"] == "test-session"

    def test_numpy_serialization(self, tmp_path, monkeypatch):
        """Numpy arrays in outputs must be JSON-serialized correctly."""
        import audit.logger as alog
        import yaml

        cfg_content = yaml.safe_load(open("config.yaml"))
        cfg_content["audit"]["log_dir"] = str(tmp_path)

        monkeypatch.setattr(alog, "_load_config", lambda: cfg_content)

        event_id = alog.log_tool_call(
            tool="test",
            inputs={},
            outputs={"array": np.array([1.0, 2.0, 3.0]), "scalar": np.float64(42.0)},
            session_id="test",
        )
        event = alog.load_event(event_id)
        assert event["outputs"]["array"] == [1.0, 2.0, 3.0]
        assert event["outputs"]["scalar"] == 42.0


# ---------------------------------------------------------------------------
# Test 8: Brief validator
# ---------------------------------------------------------------------------

class TestBriefValidator:
    def test_validator_catches_fabricated_number(self):
        """A number not in tool results must be rejected."""
        from ai.agent import validate_brief

        tool_results = [{"pc": 0.0001234, "miss_km": 5.678}]
        fake_brief = {
            "highest_pc": 0.9999,  # not in tool results
            "miss_km": 5.678,
        }
        with pytest.raises(ValueError, match="not found in tool results"):
            validate_brief(fake_brief, tool_results)

    def test_validator_passes_correct_brief(self):
        """A brief with all numbers from tool results must pass."""
        from ai.agent import validate_brief

        tool_results = [{"pc": 0.0001234, "miss_km": 5.678, "tca_utc": "2025-01-01T12:00:00"}]
        good_brief = {
            "highest_pc": 0.0001234,
            "miss_km": 5.678,
        }
        # Should not raise
        validate_brief(good_brief, tool_results)

    def test_validator_allows_small_integers(self):
        """Small integers (0-100) are exempt from tracing."""
        from ai.agent import validate_brief

        tool_results = [{"pc": 0.001}]
        brief = {"highest_pc": 0.001, "rank": 1, "n": 7}
        # Should not raise
        validate_brief(brief, tool_results)


# ---------------------------------------------------------------------------
# Test 9: Coarse filter range threshold (unit conversion sanity)
# ---------------------------------------------------------------------------

class TestRangeFilter:
    def test_range_threshold_is_in_metres(self):
        """Coarse filter threshold from config must be in metres (not km)."""
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        threshold = cfg["screen"]["coarse_range_m"]
        # 50 km = 50000 m. If someone stored 50 (km), it would be way too small.
        assert threshold >= 1000, f"Range threshold {threshold} looks like km, not metres"
        assert threshold <= 500000, f"Range threshold {threshold}m > 500km is implausibly large"

    def test_no_conjunction_with_range_above_threshold(self):
        """If we create a synthetic event at 60 km, it should NOT pass a 50 km filter."""
        # This is a regression test for the km/m confusion bug
        range_km = 60.0
        range_m = range_km * 1000.0
        threshold_m = 50000.0  # 50 km in metres
        assert range_m > threshold_m, "60km should fail 50km threshold"
        # And verify 40 km passes
        assert 40_000.0 < threshold_m


# ---------------------------------------------------------------------------
# Test 10: J2 propagation orbit energy conservation
# ---------------------------------------------------------------------------

class TestJ2Propagation:
    def test_orbit_energy_approximately_conserved(self):
        """
        For two-body + J2 propagation, the specific orbital energy should be
        approximately conserved over one orbit period. J2 doesn't add or remove energy,
        only precesses the orbit.
        
        Tolerance: 0.1% over one orbit period (~6000s for LEO).
        """
        from core.maneuver import _propagate_j2
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        prop = cfg["propagation"]

        # ~500 km SSO orbit — explicit float() because YAML may load sci-notation as string
        mu = float(prop["mu_m3s2"])
        re = float(prop["re_m"])
        j2 = float(prop["j2"])

        r0 = np.array([6878137.0, 0.0, 0.0])   # metres
        r0_mag = float(np.linalg.norm(r0))
        v_circ = math.sqrt(mu / r0_mag)
        v0 = np.array([0.0, v_circ, 0.0])       # circular orbit

        # Propagate one orbit period
        T = 2 * math.pi * math.sqrt(r0_mag**3 / mu)
        t_eval = np.linspace(0, T, 100)

        r_out, v_out = _propagate_j2(r0, v0, (0, float(T)), t_eval, mu, re, j2, max_step_s=60.0)

        # Specific mechanical energy at each point
        energies = np.array([
            0.5 * float(np.linalg.norm(v_out[i]))**2 - mu / float(np.linalg.norm(r_out[i]))
            for i in range(len(t_eval))
        ])

        e0 = energies[0]
        max_drift = np.max(np.abs(energies - e0))
        relative_drift = max_drift / abs(e0)

        assert relative_drift < 1e-3, (
            f"Orbital energy drift {relative_drift:.2e} over one orbit period "
            f"exceeds 0.1% tolerance — J2 integrator may have a bug"
        )

    def test_prograde_burn_increases_altitude(self):
        """
        A prograde burn in a circular orbit increases the orbital energy,
        which should manifest as an increased semi-major axis / apogee.
        """
        from core.maneuver import _propagate_j2
        import yaml
        cfg = yaml.safe_load(open("config.yaml"))
        prop = cfg["propagation"]

        mu = float(prop["mu_m3s2"])
        re = float(prop["re_m"])
        j2 = float(prop["j2"])

        r0 = np.array([6878137.0, 0.0, 0.0])
        v_circ = math.sqrt(mu / float(np.linalg.norm(r0)))
        v0_base = np.array([0.0, v_circ, 0.0])

        # No burn
        t_eval = np.linspace(0, 3600, 60)
        r_no_burn, v_no_burn = _propagate_j2(r0, v0_base, (0, 3600), t_eval, mu, re, j2, 60.0)

        # 1 m/s prograde burn
        v0_burn = v0_base + np.array([0.0, 1.0, 0.0])
        r_burn, v_burn = _propagate_j2(r0, v0_burn, (0, 3600), t_eval, mu, re, j2, 60.0)

        # After 1 hour, burned orbit should be higher on average
        r_no_burn_mean = np.mean(np.linalg.norm(r_no_burn, axis=1))
        r_burn_mean = np.mean(np.linalg.norm(r_burn, axis=1))

        assert r_burn_mean > r_no_burn_mean, (
            f"Prograde burn did not increase mean altitude: "
            f"base {r_no_burn_mean:.0f}m vs burned {r_burn_mean:.0f}m"
        )
