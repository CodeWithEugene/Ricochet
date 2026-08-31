#!/usr/bin/env python3
"""
scripts/seed_catalog.py
=======================
Seed the local catalog cache with a representative sample of satellites.

Used when CelesTrak's GROUP endpoint is unavailable (503, rate-limited, etc.).
Fetches individual satellites by CATNR — slower but more reliable.

This script populates the cache so Ricochet can run without waiting for
CelesTrak's GROUP endpoint to become available.

Run: python scripts/seed_catalog.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# --------------------------------------------------------------------------
# Satellite list: selected to cover different orbit regimes
# Includes Taifa-1, ISS, Starlink samples, debris samples, and SSO objects
# --------------------------------------------------------------------------

SEED_NORADS = [
    # Key reference objects
    25544,   # ISS (ZARYA)
    56212,   # TAIFA-1 (Kenya)
    43205,   # LEMUR-2 series (SSO)
    # Starlink samples
    44235, 44236, 44237, 44238, 44239,
    44713, 44714, 44715, 44716, 44717,
    47162, 47163, 47164, 47165, 47166,
    # OneWeb
    47844, 47845, 47846, 47847,
    # Planet Labs
    43174, 43175, 43176, 43177,
    # ISS crew vehicles
    43205, 45701,
    # Debris (Cosmos-2251 fragment samples)
    35700, 35701, 35702, 35703, 35704,
    33500, 33501, 33502, 33503, 33504,
    # Active scientific satellites
    27386,   # Aqua
    25994,   # Terra
    39084,   # Suomi NPP
    43013,   # NOAA-20
    # GPS
    37753, 39533, 40294, 40730, 41019,
    # Galileo
    37846, 38857, 40128, 41174, 41550,
    # GLONASS
    32393, 32394, 32395, 33405, 36111,
    # Additional SSO objects
    42063, 42064, 42065, 42066, 42067,
    43206, 43207, 43208, 43209, 43210,
    44878, 44879, 44880, 44881, 44882,
    # Iridium NEXT
    41917, 41918, 41919, 41920,
    42803, 42804, 42805, 42806,
    # Random LEO catalog expansion
    45016, 45017, 45018, 45019, 45020,
    46069, 46070, 46071, 46072, 46073,
    47492, 47493, 47494, 47495, 47496,
    48274, 48275, 48276, 48277, 48278,
    49260, 49261, 49262, 49263, 49264,
    50001, 50002, 50003, 50004, 50005,
    51000, 51001, 51002, 51003, 51004,
    52000, 52001, 52002, 52003, 52004,
    53000, 53001, 53002, 53003, 53004,
    54000, 54001, 54002, 54003, 54004,
    55000, 55001, 55002, 55003, 55004,
    56000, 56001, 56002, 56003,
    # 500-600km SSO region objects (Taifa-1 shell)
    44878, 45358, 46495, 47706, 48259,
    48766, 49017, 49018, 49019, 49020,
    49799, 50002, 50003, 50004, 50005,
]

SEED_NORADS = list(set(SEED_NORADS))  # deduplicate

BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
HEADERS = {"User-Agent": "Ricochet-CAM/1.0 (research; github.com/ricochet-cdm)"}


def fetch_single(norad: int) -> dict | None:
    """Fetch a single satellite by CATNR."""
    try:
        r = requests.get(
            BASE_URL,
            params={"CATNR": norad, "FORMAT": "json"},
            headers=HEADERS,
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            if data and isinstance(data, list):
                return data[0]
        return None
    except Exception:
        return None


def main():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    cache_dir = Path(cfg["catalog"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Seeding catalog cache with {len(SEED_NORADS)} satellites...")
    print("(This is a fallback for when CelesTrak GROUP endpoint is unavailable)")
    print()

    records = []
    failed = []

    for i, norad in enumerate(SEED_NORADS):
        result = fetch_single(norad)
        if result:
            records.append(result)
            print(f"  [{i+1}/{len(SEED_NORADS)}] {norad}: {result.get('OBJECT_NAME', '?')}")
        else:
            failed.append(norad)
            if len(failed) <= 5:
                print(f"  [{i+1}/{len(SEED_NORADS)}] {norad}: FAILED (expired/not found)")

        # Be gentle with CelesTrak — 3 requests per second max
        time.sleep(0.35)

    print(f"\nFetched {len(records)} satellites, {len(failed)} failed")

    if not records:
        print("ERROR: No satellites fetched. Check network connectivity.")
        sys.exit(1)

    df = pd.DataFrame(records)

    # Parse epoch
    if "EPOCH" in df.columns:
        df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True)

    if "NORAD_CAT_ID" in df.columns:
        df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"], errors="coerce").astype("Int64")

    for col in ("TLE_LINE1", "TLE_LINE2"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = df.drop_duplicates(subset=["NORAD_CAT_ID"], keep="first")
    df["catalog_group"] = "seeded"

    # Save as "active" group cache
    parquet_path = cache_dir / "catalog_active.parquet"
    meta_path = cache_dir / "catalog_active.meta.json"

    df.to_parquet(parquet_path, index=False)
    with open(meta_path, "w") as f:
        json.dump({
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(df),
            "note": "Seeded via scripts/seed_catalog.py (CelesTrak GROUP endpoint unavailable)",
        }, f, indent=2)

    # Create empty parquet files for other groups to prevent load errors
    empty_df = pd.DataFrame(columns=df.columns)
    for group_name in ("debris", "starlink"):
        p_path = cache_dir / f"catalog_{group_name}.parquet"
        m_path = cache_dir / f"catalog_{group_name}.meta.json"
        if not p_path.exists():
            empty_df.to_parquet(p_path, index=False)
            with open(m_path, "w") as f:
                json.dump({
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "rows": 0,
                    "note": "Empty placeholder — seeded via seed_catalog.py",
                }, f)

    print(f"\nCatalog seeded: {len(df)} objects → {parquet_path}")
    print("You can now run: streamlit run app.py")

    # Verify Taifa-1 is in the catalog
    taifa = df[df["NORAD_CAT_ID"] == 56212]
    if not taifa.empty:
        print(f"\nTaifa-1 found: {taifa.iloc[0].get('OBJECT_NAME', '?')}")
    else:
        print("\nWARNING: Taifa-1 (56212) not in catalog. Fetch may have failed for this object.")


if __name__ == "__main__":
    main()
