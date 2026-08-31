"""
data/fetch_catalog.py
=====================
Fetch CelesTrak GP/OMM JSON → local Parquet cache.

Rules:
- Never refetch within 4 hours (TTL in config.yaml)
- Credit CelesTrak in User-Agent
- Return a single merged DataFrame with all configured groups
- All column names match the OMM field names exactly
- EPOCH parsed to UTC-aware datetime64
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yaml


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _cache_path(group_name: str, cache_dir: str) -> Path:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return Path(cache_dir) / f"catalog_{group_name}.parquet"


def _meta_path(group_name: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"catalog_{group_name}.meta.json"


def _is_cache_fresh(meta_path: Path, ttl_hours: int) -> bool:
    """Return True if cache was written within TTL window."""
    if not meta_path.exists():
        return False
    with open(meta_path) as f:
        meta = json.load(f)
    fetched_at = datetime.fromisoformat(meta["fetched_at"])
    age = datetime.now(timezone.utc) - fetched_at
    return age < timedelta(hours=ttl_hours)


def _fetch_group(group: dict, cfg: dict) -> pd.DataFrame:
    """Fetch one CelesTrak group and return a DataFrame.
    
    Handles CelesTrak's 403 'not updated' response, which means the data
    hasn't changed since our last download — not a real error.
    Returns None if the server indicates data is unchanged (caller uses cache).
    """
    url = cfg["catalog"]["celestrak_url"]
    headers = {"User-Agent": cfg["catalog"]["user_agent"]}
    params = group["params"]

    resp = requests.get(url, params=params, headers=headers, timeout=45)

    # CelesTrak returns 403 with a text body when data hasn't changed since our
    # last download (identified by their server-side tracking). This is NOT an error
    # when we have a cached copy. When we have NO cached copy, we retry without the
    # implied "since last download" assumption — but the body tells us the data IS
    # current, so we trust it and return None to signal "use cache or refetch".
    if resp.status_code == 403:
        body = resp.text or ""
        if "not updated since your last" in body or "updated once every" in body:
            return None   # Caller will use cache or raise a clear error
        # Real 403 — raise
        resp.raise_for_status()

    resp.raise_for_status()

    try:
        records = resp.json()
    except Exception:
        raise RuntimeError(
            f"CelesTrak returned non-JSON response ({resp.status_code}): {resp.text[:200]}"
        )
    df = pd.DataFrame(records)
    return df


def _fetch_group_tle_fallback(group: dict, cfg: dict) -> Optional[pd.DataFrame]:
    """
    Fallback: fetch 3LE/TLE text format from CelesTrak and convert to a minimal DataFrame.
    Used when the JSON endpoint returns 403 "not updated" with no local cache.
    
    The TLE text format does not have a rate-limit 403 issue.
    Returns a DataFrame with NORAD_CAT_ID, OBJECT_NAME, TLE_LINE1, TLE_LINE2 columns,
    or None on failure.
    """
    url = cfg["catalog"]["celestrak_url"]
    headers = {"User-Agent": cfg["catalog"]["user_agent"]}
    # Request TLE text format
    tle_params = {k: v for k, v in group["params"].items()}
    tle_params["FORMAT"] = "tle"

    try:
        resp = requests.get(url, params=tle_params, headers=headers, timeout=60)
        if resp.status_code != 200:
            return None
        text = resp.text
    except Exception:
        return None

    # Parse 3LE format: NAME / LINE1 / LINE2 triplets
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    records = []
    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            try:
                norad_id = int(line1[2:7].strip())
            except ValueError:
                i += 1
                continue
            records.append({
                "OBJECT_NAME": name,
                "NORAD_CAT_ID": norad_id,
                "TLE_LINE1": line1,
                "TLE_LINE2": line2,
                "EPOCH": None,   # Will be filled by _normalise_df if needed
            })
            i += 3
        else:
            i += 1

    if not records:
        return None
    return pd.DataFrame(records)




def _save_group(df: pd.DataFrame, group_name: str, cache_dir: str) -> None:
    cache_p = _cache_path(group_name, cache_dir)
    meta_p = _meta_path(group_name, cache_dir)

    df.to_parquet(cache_p, index=False)
    with open(meta_p, "w") as f:
        json.dump({"fetched_at": datetime.now(timezone.utc).isoformat(), "rows": len(df)}, f)


def _load_group_cache(group_name: str, cache_dir: str) -> pd.DataFrame:
    return pd.read_parquet(_cache_path(group_name, cache_dir))


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw OMM DataFrame:
    - Parse EPOCH to UTC-aware Timestamp
    - Ensure NORAD_CAT_ID is int
    - Ensure TLE_LINE1, TLE_LINE2 are strings
    - Drop rows with null TLE lines (can't propagate)
    """
    if "EPOCH" in df.columns:
        df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True)

    if "NORAD_CAT_ID" in df.columns:
        df["NORAD_CAT_ID"] = pd.to_numeric(df["NORAD_CAT_ID"], errors="coerce").astype("Int64")

    for col in ("TLE_LINE1", "TLE_LINE2"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Drop rows that can't be propagated
    mask = df["TLE_LINE1"].notna() & df["TLE_LINE2"].notna()
    mask &= df["TLE_LINE1"] != "nan"
    mask &= df["TLE_LINE2"] != "nan"
    df = df[mask].copy()

    return df


def load_catalog(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load the merged satellite catalog.

    Parameters
    ----------
    force_refresh : bool
        If True, bypass the TTL cache and fetch fresh data.

    Returns
    -------
    pd.DataFrame
        Merged catalog with columns including:
        NORAD_CAT_ID, OBJECT_NAME, TLE_LINE1, TLE_LINE2, EPOCH, ...
        EPOCH is UTC-aware datetime64.
    """
    cfg = _load_config()
    cache_dir = cfg["catalog"]["cache_dir"]
    ttl_hours = cfg["catalog"]["cache_ttl_hours"]
    groups = cfg["catalog"]["groups"]

    frames = []
    for group in groups:
        name = group["name"]
        meta_p = _meta_path(name, cache_dir)
        cache_p = _cache_path(name, cache_dir)

        if not force_refresh and _is_cache_fresh(meta_p, ttl_hours) and cache_p.exists():
            df = _load_group_cache(name, cache_dir)
        else:
            print(f"[catalog] Fetching group '{name}' from CelesTrak...")
            df = _fetch_group(group, cfg)
            if df is None:
                # CelesTrak says data hasn't changed — use existing cache if available
                if cache_p.exists():
                    print(f"[catalog] CelesTrak: data unchanged, using cached copy for '{name}'")
                    df = _load_group_cache(name, cache_dir)
                else:
                    raise RuntimeError(
                        f"CelesTrak indicates data unchanged for '{name}' but no local cache exists. "
                        "Try again after CelesTrak updates (every 2 hours)."
                    )
            else:
                _save_group(df, name, cache_dir)
                print(f"[catalog] Cached {len(df)} records for '{name}'")

        df["catalog_group"] = name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged = _normalise_df(merged)

    # Deduplicate by NORAD_CAT_ID, keeping first (active > debris in group order)
    merged = merged.drop_duplicates(subset=["NORAD_CAT_ID"], keep="first")
    merged = merged.reset_index(drop=True)

    return merged


def get_object_tle(norad_id: int, catalog: Optional[pd.DataFrame] = None) -> tuple[str, str]:
    """
    Return (TLE_LINE1, TLE_LINE2) for a given NORAD ID.
    Raises KeyError if not found.
    """
    if catalog is None:
        catalog = load_catalog()
    row = catalog[catalog["NORAD_CAT_ID"] == norad_id]
    if row.empty:
        raise KeyError(f"NORAD ID {norad_id} not found in catalog")
    return row.iloc[0]["TLE_LINE1"], row.iloc[0]["TLE_LINE2"]
