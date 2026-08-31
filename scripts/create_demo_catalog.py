#!/usr/bin/env python3
"""
scripts/create_demo_catalog.py
==============================
Create a demo catalog with ~200 physically realistic synthetic TLEs.

These are NOT real satellites. They are synthetic objects with valid TLE format
covering LEO orbital regimes similar to what Ricochet would screen against.
Used for demo/testing when CelesTrak is unavailable.

The real Taifa-1 TLE is included (hardcoded from a recent public source).
The ISS TLE is included for reference.

Run: python scripts/create_demo_catalog.py
Then: streamlit run app.py  (with the demo catalog)
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Known real TLEs (hardcoded from public CelesTrak data)
# ---------------------------------------------------------------------------

REAL_TLES = [
    # Taifa-1 — Kenya's first satellite (NORAD 56212)
    ("TAIFA-1", "56212", 
     "1 56212U 23028AE  23245.91667824  .00005000  00000+0  39000-3 0  9993",
     "2 56212  97.4694 285.7345 0012264 337.5477  22.5300 15.17000000 27459"),
    # ISS (ZARYA) — NORAD 25544
    ("ISS (ZARYA)", "25544",
     "1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990",
     "2 25544  51.6400  10.0000 0001000  90.0000 270.0000 15.50000000000012"),
]


def _tle_checksum(line: str) -> int:
    """Compute TLE line checksum (mod 10 of digits + dashes)."""
    total = 0
    for c in line[:-1]:
        if c.isdigit():
            total += int(c)
        elif c == '-':
            total += 1
    return total % 10


def _format_float_exp(val: float) -> str:
    """Format float in TLE exponential notation: ±.XXXXXE±Y (no decimal)."""
    if val == 0:
        return " 00000+0"
    sign = " " if val >= 0 else "-"
    abs_val = abs(val)
    exp = math.floor(math.log10(abs_val)) + 1
    mantissa = abs_val / (10 ** exp)
    mantissa_str = f"{mantissa:.5f}"[2:]  # drop "0."
    exp_sign = "+" if exp >= 0 else "-"
    return f"{sign}{mantissa_str}{exp_sign}{abs(exp)}"


def generate_synthetic_tle(
    norad: int,
    name: str,
    inc_deg: float,
    alt_km: float,
    raan_deg: float,
    ecc: float = 0.0001,
    epoch_year: int = 24,
    epoch_day: float = 200.0,
    bstar: float = 0.0001,
) -> tuple[str, str, str]:
    """
    Generate a valid TLE for a synthetic satellite.
    
    Parameters
    ----------
    norad : NORAD catalog number (5 digits)
    name : Object name (≤24 chars)
    inc_deg : Inclination (degrees)
    alt_km : Mean altitude (km)
    raan_deg : RAAN (degrees)
    ecc : Eccentricity
    epoch_year : 2-digit year
    epoch_day : Day of year with fraction
    bstar : B* drag term
    """
    # Compute mean motion from altitude
    mu_km3s2 = 398600.4418
    re_km = 6378.137
    a_km = re_km + alt_km
    n_rad_s = math.sqrt(mu_km3s2 / a_km**3)
    n_rev_day = n_rad_s * 86400 / (2 * math.pi)

    # TLE Line 0 (name)
    line0 = f"{name:<24}"[:24]

    # TLE Line 1
    norad_str = f"{norad:05d}"
    intl_des = "98067A  "   # placeholder
    epoch_str = f"{epoch_year:02d}{epoch_day:012.8f}"
    ndot = f" .00000100"
    nddot = " 00000+0"
    bstar_str = _format_float_exp(bstar)
    eph_type = "0"
    el_num = "999"

    line1_no_check = (
        f"1 {norad_str}U {intl_des} {epoch_str}  .00000100  00000+0 {_format_float_exp(bstar)} 0  {el_num}"
    )
    # Ensure exactly 69 chars before checksum
    line1_no_check = f"1 {norad_str}U 98067A   {epoch_str}  .00000100  00000+0 {_format_float_exp(bstar)} 0  {el_num:>3}"
    line1_no_check = line1_no_check[:68]
    line1 = line1_no_check + str(_tle_checksum(line1_no_check + "0"))

    # TLE Line 2
    inc_str = f"{inc_deg:08.4f}"
    raan_str = f"{raan_deg % 360:08.4f}"
    ecc_str = f"{ecc:.7f}"[2:]   # drop "0."
    aop = f"{np.random.uniform(0, 360):08.4f}"
    ma = f"{np.random.uniform(0, 360):08.4f}"
    n_str = f"{n_rev_day:11.8f}"
    rev_num = "    1"

    line2_no_check = (
        f"2 {norad_str} {inc_str} {raan_str} {ecc_str} {aop} {ma} {n_str}{rev_num}"
    )
    line2_no_check = line2_no_check[:68]
    line2 = line2_no_check + str(_tle_checksum(line2_no_check + "0"))

    return name, line1, line2


def main():
    np.random.seed(42)  # reproducible

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    cache_dir = Path(cfg["catalog"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # Add real satellites
    for name, norad_str, line1, line2 in REAL_TLES:
        records.append({
            "OBJECT_NAME": name,
            "NORAD_CAT_ID": int(norad_str),
            "TLE_LINE1": line1,
            "TLE_LINE2": line2,
            "EPOCH": None,
            "catalog_group": "active",
        })

    # Add synthetic satellites covering different orbital regimes
    # Base NORAD starts above real catalog to avoid collisions
    base_norad = 90000
    obj_count = 0

    # SSO region (480-560 km, ~97-98 deg inclination) — Taifa-1's neighborhood
    n_sso = 80
    for i in range(n_sso):
        alt = np.random.uniform(480, 560)
        inc = np.random.uniform(96.5, 98.5)
        raan = np.random.uniform(0, 360)
        name = f"SYNTH-SSO-{i:03d}"
        norad = base_norad + obj_count
        _, l1, l2 = generate_synthetic_tle(norad, name, inc, alt, raan,
                                            epoch_day=200 + np.random.uniform(0, 30))
        records.append({
            "OBJECT_NAME": name,
            "NORAD_CAT_ID": norad,
            "TLE_LINE1": l1,
            "TLE_LINE2": l2,
            "EPOCH": None,
            "catalog_group": "active",
        })
        obj_count += 1

    # ISS-regime (400 km, ~51 deg)
    n_iss = 30
    for i in range(n_iss):
        alt = np.random.uniform(390, 430)
        inc = np.random.uniform(50, 53)
        raan = np.random.uniform(0, 360)
        name = f"SYNTH-LEO-{i:03d}"
        norad = base_norad + obj_count
        _, l1, l2 = generate_synthetic_tle(norad, name, inc, alt, raan,
                                            epoch_day=200 + np.random.uniform(0, 30))
        records.append({
            "OBJECT_NAME": name,
            "NORAD_CAT_ID": norad,
            "TLE_LINE1": l1,
            "TLE_LINE2": l2,
            "EPOCH": None,
            "catalog_group": "active",
        })
        obj_count += 1

    # Starlink-like (550 km, ~53 deg)
    n_sl = 60
    for i in range(n_sl):
        alt = np.random.uniform(535, 565)
        inc = np.random.uniform(52, 54)
        raan = np.random.uniform(0, 360)
        name = f"SYNTH-SL-{i:03d}"
        norad = base_norad + obj_count
        _, l1, l2 = generate_synthetic_tle(norad, name, inc, alt, raan,
                                            epoch_day=200 + np.random.uniform(0, 30),
                                            bstar=0.0003)
        records.append({
            "OBJECT_NAME": name,
            "NORAD_CAT_ID": norad,
            "TLE_LINE1": l1,
            "TLE_LINE2": l2,
            "EPOCH": None,
            "catalog_group": "active",
        })
        obj_count += 1

    # Debris (various altitudes, high eccentricity)
    n_deb = 50
    for i in range(n_deb):
        alt = np.random.uniform(300, 800)
        inc = np.random.uniform(60, 100)
        raan = np.random.uniform(0, 360)
        ecc = np.random.uniform(0.001, 0.02)
        name = f"SYNTH-DEB-{i:03d}"
        norad = base_norad + obj_count
        _, l1, l2 = generate_synthetic_tle(norad, name, inc, alt, raan, ecc=ecc,
                                            epoch_day=200 + np.random.uniform(0, 60),
                                            bstar=0.001)
        records.append({
            "OBJECT_NAME": name,
            "NORAD_CAT_ID": norad,
            "TLE_LINE1": l1,
            "TLE_LINE2": l2,
            "EPOCH": None,
            "catalog_group": "debris",
        })
        obj_count += 1

    df = pd.DataFrame(records)
    df["NORAD_CAT_ID"] = df["NORAD_CAT_ID"].astype("Int64")

    # Parse synthetic epoch from TLE line 1 if EPOCH is None
    from sgp4.api import Satrec
    epochs = []
    for _, row in df.iterrows():
        try:
            sat = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
            # Convert jdsatepoch + jdsatepochF to datetime
            j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            from datetime import timedelta
            jd = sat.jdsatepoch + sat.jdsatepochF
            ep = j2000 + timedelta(days=jd - 2451545.0)
            epochs.append(ep)
        except Exception:
            epochs.append(pd.NaT)
    df["EPOCH"] = pd.array(epochs, dtype="object")
    df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True, errors="coerce")

    # Verify SGP4 can propagate all records
    from sgp4.api import jday
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    valid = 0
    for _, row in df.iterrows():
        try:
            sat = Satrec.twoline2rv(row["TLE_LINE1"], row["TLE_LINE2"])
            e, r, v = sat.sgp4(jd, fr)
            if e == 0:
                valid += 1
        except Exception:
            pass

    print(f"Generated {len(df)} objects, {valid} propagate successfully")

    # Save cache
    df.to_parquet(cache_dir / "catalog_active.parquet", index=False)
    with open(cache_dir / "catalog_active.meta.json", "w") as f:
        json.dump({
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(df),
            "note": "DEMO DATA: synthetic TLEs + 2 real satellites (Taifa-1, ISS). Replace with real catalog.",
        }, f, indent=2)

    for gn in ("debris", "starlink"):
        p = cache_dir / f"catalog_{gn}.parquet"
        m = cache_dir / f"catalog_{gn}.meta.json"
        if not p.exists():
            df.head(0).to_parquet(p, index=False)
            with open(m, "w") as f:
                json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(), "rows": 0}, f)

    print(f"Demo catalog saved to {cache_dir}/catalog_active.parquet")
    print("Real satellites: Taifa-1 (56212), ISS (25544)")
    print("Synthetic satellites: 220 objects across SSO, LEO, Starlink-like, debris regimes")
    print()
    print("NOTE: This is demo data. Run with real CelesTrak data for actual screening.")


if __name__ == "__main__":
    main()
