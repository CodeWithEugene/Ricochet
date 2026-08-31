"""
ai/tools.py
===========
Tool definitions for the Granite AI agent.

Each tool wraps one core module function.
Tool schemas follow the Ollama tool-calling format (OpenAI-compatible).

Design rules:
- The LLM receives ONLY tool results. It does not compute physics.
- Every tool logs its call to audit/
- Tools return dicts (JSON-serializable, not dataclasses) so the LLM can read them
- All numeric values are rounded for display but the raw value is always in the audit log
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import yaml
from pathlib import Path

from audit.logger import log_tool_call


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


_SESSION_ID = str(uuid.uuid4())   # One session per process


def _fmt_pc(pc: float) -> str:
    """Format Pc for LLM consumption."""
    if pc < 1e-10:
        return f"{pc:.2e}"
    return f"{pc:.3e}"


# ---------------------------------------------------------------------------
# Tool: screen
# ---------------------------------------------------------------------------

def tool_screen(norad_id: int, window_days: float = 7) -> dict:
    """
    Screen a satellite for close approaches over the specified window.
    Returns all conjunctions with miss distance < 50 km.
    """
    from data.fetch_catalog import load_catalog
    from core.screen import screen

    t0 = time.time()
    catalog = load_catalog()
    events = screen(norad_id=norad_id, window_days=window_days, catalog=catalog)

    output = {
        "norad_id": norad_id,
        "window_days": window_days,
        "n_conjunctions": len(events),
        "conjunctions": [
            {
                "rank": i + 1,
                "secondary_norad": ev.secondary_norad,
                "secondary_name": ev.secondary_name,
                "tca_utc": ev.tca.isoformat(),
                "miss_km": round(ev.miss_m / 1000.0, 3),
                "rel_v_kms": round(ev.rel_v_ms / 1000.0, 3),
                "primary_epoch_age_days": round(ev.primary_epoch_offset_s / 86400, 1),
                "secondary_epoch_age_days": round(ev.secondary_epoch_offset_s / 86400, 1),
            }
            for i, ev in enumerate(events)
        ],
    }

    log_tool_call(
        tool="screen",
        inputs={"norad_id": norad_id, "window_days": window_days},
        outputs=output,
        model_name="",
        duration_s=time.time() - t0,
        session_id=_SESSION_ID,
    )
    return output


# ---------------------------------------------------------------------------
# Tool: compute_pc
# ---------------------------------------------------------------------------

def tool_compute_pc(norad_id: int, secondary_norad: int, window_days: float = 7) -> dict:
    """
    Compute probability of collision for the closest approach between
    a primary and a specific secondary satellite.
    """
    from data.fetch_catalog import load_catalog
    from core.screen import screen
    from core.pc import compute_pc, compute_pc_sensitivity

    t0 = time.time()
    catalog = load_catalog()
    events = screen(norad_id=norad_id, window_days=window_days, catalog=catalog)

    # Find the conjunction with this secondary
    matching = [ev for ev in events if ev.secondary_norad == secondary_norad]
    if not matching:
        output = {
            "error": f"No conjunction found between {norad_id} and {secondary_norad} within {window_days} days",
            "norad_id": norad_id,
            "secondary_norad": secondary_norad,
        }
    else:
        # Pick closest approach
        event = min(matching, key=lambda e: e.miss_m)
        pc_result = compute_pc(event)
        sensitivity = compute_pc_sensitivity(event)

        output = {
            "norad_id": norad_id,
            "secondary_norad": secondary_norad,
            "tca_utc": event.tca.isoformat(),
            "miss_km": round(event.miss_m / 1000.0, 3),
            "rel_v_kms": round(event.rel_v_ms / 1000.0, 3),
            "pc": pc_result.pc,
            "pc_formatted": _fmt_pc(pc_result.pc),
            "sigma_1_km": round(pc_result.sigma_1 / 1000.0, 3),
            "sigma_2_km": round(pc_result.sigma_2 / 1000.0, 3),
            "hbr_m": pc_result.hbr_m,
            "covariance_assumption": {
                "sigma_r_m": round(pc_result.covariance_assumption.sigma_r_m, 1),
                "sigma_t_m": round(pc_result.covariance_assumption.sigma_t_m, 1),
                "sigma_n_m": round(pc_result.covariance_assumption.sigma_n_m, 1),
                "primary_epoch_age_days": round(pc_result.covariance_assumption.primary_epoch_offset_s / 86400, 1),
                "secondary_epoch_age_days": round(pc_result.covariance_assumption.secondary_epoch_offset_s / 86400, 1),
                "source": pc_result.covariance_assumption.source,
                "applied_to_both": pc_result.covariance_assumption.applied_to_both,
            },
            "sensitivity": {
                str(k): {"pc": round(v.pc, 4), "pc_formatted": _fmt_pc(v.pc)}
                for k, v in sensitivity.items()
            },
            "alert_threshold": _load_config()["pc"]["alert_threshold"],
            "above_threshold": pc_result.pc >= _load_config()["pc"]["alert_threshold"],
        }

    log_tool_call(
        tool="compute_pc",
        inputs={"norad_id": norad_id, "secondary_norad": secondary_norad, "window_days": window_days},
        outputs=output,
        duration_s=time.time() - t0,
        session_id=_SESSION_ID,
    )
    return output


# ---------------------------------------------------------------------------
# Tool: rescreen
# ---------------------------------------------------------------------------

def tool_rescreen(
    norad_id: int,
    tca_utc: str,
    dv_min_ms: float = -2.0,
    dv_max_ms: float = 2.0,
    grid_n: int = 7,
) -> dict:
    """
    Compute the total-Pc maneuver trade space for a maneuverable satellite.
    Returns a grid of total_pc values over (delta-v × burn time) options.
    """
    from core.rescreen import rescreen

    t0 = time.time()
    tca = datetime.fromisoformat(tca_utc)
    if tca.tzinfo is None:
        tca = tca.replace(tzinfo=timezone.utc)

    result = rescreen(
        norad_id=norad_id,
        primary_tca=tca,
        dv_range=(dv_min_ms, dv_max_ms),
        grid_n=grid_n,
    )

    best_gp = None
    if result.recommended_dv_ms is not None:
        best_gps = [gp for gp in result.grid_points 
                    if abs(gp.dv_ms - result.recommended_dv_ms) < 1e-6]
        if best_gps:
            best_gp = best_gps[0]

    output = {
        "norad_id": norad_id,
        "tca_utc": tca_utc,
        "baseline_pc": result.baseline_pc,
        "baseline_pc_formatted": _fmt_pc(result.baseline_pc),
        "alert_threshold": result.alert_threshold,
        "grid_shape": list(result.grid.shape),
        "dv_values_ms": np.round(result.dv_values_ms, 4).tolist(),
        "dt_values_hours": np.round(result.dt_values_s / 3600, 2).tolist(),
        "grid_total_pc": result.grid.values.tolist(),
        "recommended_dv_ms": result.recommended_dv_ms,
        "recommended_dt_hours": (
            round(result.recommended_dt_s / 3600, 2)
            if result.recommended_dt_s is not None else None
        ),
        "recommended_total_pc": (
            round(best_gp.total_pc, 8) if best_gp else None
        ),
        "n_secondaries_screened": result.n_secondaries_screened,
        "elapsed_s": round(result.elapsed_s, 1),
        "force_model_note": (
            "Primary propagated with two-body+J2. Secondaries with SGP4. "
            "Force-model mismatch may accumulate ~1-2km/day at 500km altitude."
        ),
    }

    log_tool_call(
        tool="rescreen",
        inputs={
            "norad_id": norad_id,
            "tca_utc": tca_utc,
            "dv_min_ms": dv_min_ms,
            "dv_max_ms": dv_max_ms,
            "grid_n": grid_n,
        },
        outputs=output,
        duration_s=time.time() - t0,
        session_id=_SESSION_ID,
    )
    return output


# ---------------------------------------------------------------------------
# Tool: risk_timeline (non-maneuverable mode — Taifa-1 style)
# ---------------------------------------------------------------------------

def tool_risk_timeline(norad_id: int, window_days: float = 7) -> dict:
    """
    For a non-maneuverable satellite, produce:
    - Risk timeline (all conjunctions with Pc and TCA)
    - Payload-safing window recommendation
    - Notification checklist
    - Co-located maneuverable objects in the same orbital shell
    """
    from data.fetch_catalog import load_catalog
    from core.screen import screen
    from core.pc import compute_pc

    t0 = time.time()
    cfg = _load_config()
    catalog = load_catalog()
    events = screen(norad_id=norad_id, window_days=window_days, catalog=catalog)

    # Compute Pc for each event
    timeline = []
    max_pc = 0.0
    peak_tca = None
    for ev in events:
        try:
            pc_result = compute_pc(ev)
            pc = pc_result.pc
        except Exception:
            pc = 0.0
        max_pc = max(max_pc, pc)
        if pc >= max_pc:
            peak_tca = ev.tca

        alert_level = (
            "RED" if pc >= cfg["pc"]["alert_threshold"] else
            "YELLOW" if pc >= cfg["pc"]["elevated_threshold"] else
            "GREEN"
        )
        timeline.append({
            "rank": len(timeline) + 1,
            "secondary_norad": ev.secondary_norad,
            "secondary_name": ev.secondary_name,
            "tca_utc": ev.tca.isoformat(),
            "miss_km": round(ev.miss_m / 1000.0, 3),
            "rel_v_kms": round(ev.rel_v_ms / 1000.0, 3),
            "pc": pc,
            "pc_formatted": _fmt_pc(pc),
            "alert_level": alert_level,
        })

    # Sort by Pc descending
    timeline.sort(key=lambda x: x["pc"], reverse=True)

    # Payload safing window: 12h before peak TCA
    safing_recommendation = None
    if peak_tca:
        safing_start = peak_tca - timedelta(hours=12)
        safing_end = peak_tca + timedelta(hours=2)
        safing_recommendation = {
            "start_utc": safing_start.isoformat(),
            "end_utc": safing_end.isoformat(),
            "rationale": "12h pre-TCA to 2h post-TCA covers maneuver execution window for co-located assets",
        }

    # Co-located maneuverable objects (within 20 km altitude band)
    # (Simplified: return top-10 by miss distance from our conjunctions that are likely maneuverable)
    # Real check would use propulsion flags — not available in public TLEs
    co_located = [
        {
            "norad": ev.secondary_norad,
            "name": ev.secondary_name,
            "miss_km": round(ev.miss_m / 1000, 3),
            "note": "Maneuverability unknown from public TLE data"
        }
        for ev in events[:10]
    ]

    notification_checklist = [
        "Contact the 18th Space Defense Squadron (18 SDS) for authoritative CDM",
        "Notify spacecraft manufacturer / ground station operator",
        "Alert payload users of possible safing window",
        "Log event in Mission Operations Record",
        f"Re-check screening 24h before TCA if TLE age > 2 days",
        "PUBLIC TLE DISCLAIMER: This assessment is NOT for operational use. Contact 18 SDS.",
    ]

    output = {
        "norad_id": norad_id,
        "maneuverable": False,
        "window_days": window_days,
        "n_conjunctions": len(events),
        "max_pc": max_pc,
        "max_pc_formatted": _fmt_pc(max_pc),
        "peak_tca_utc": peak_tca.isoformat() if peak_tca else None,
        "timeline": timeline[:20],   # top 20
        "payload_safing_window": safing_recommendation,
        "co_located_objects": co_located,
        "notification_checklist": notification_checklist,
        "disclaimer": "PUBLIC TLEs MUST NOT BE USED FOR OPERATIONAL CONJUNCTION ASSESSMENT. Contact 18th Space Defense Squadron.",
    }

    log_tool_call(
        tool="risk_timeline",
        inputs={"norad_id": norad_id, "window_days": window_days},
        outputs=output,
        duration_s=time.time() - t0,
        session_id=_SESSION_ID,
    )
    return output


# ---------------------------------------------------------------------------
# Tool registry (for Ollama tool-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "screen",
            "description": "Screen a satellite for close approaches against the full catalog. Returns all conjunctions within 50 km over the specified window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "norad_id": {"type": "integer", "description": "NORAD catalog ID of the primary satellite"},
                    "window_days": {"type": "number", "description": "Screening window in days (default 7)"},
                },
                "required": ["norad_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_pc",
            "description": "Compute probability of collision between a primary and secondary satellite. Returns Pc, covariance assumptions, and sensitivity analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "norad_id": {"type": "integer", "description": "Primary satellite NORAD ID"},
                    "secondary_norad": {"type": "integer", "description": "Secondary satellite NORAD ID"},
                    "window_days": {"type": "number", "description": "Screening window in days (default 7)"},
                },
                "required": ["norad_id", "secondary_norad"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rescreen",
            "description": "For a maneuverable satellite, compute the total probability of collision across a grid of candidate maneuvers. Returns a heatmap grid and the recommended minimum-delta-v burn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "norad_id": {"type": "integer", "description": "Primary satellite NORAD ID"},
                    "tca_utc": {"type": "string", "description": "TCA in ISO 8601 UTC format"},
                    "dv_min_ms": {"type": "number", "description": "Minimum delta-v in m/s (default -2.0)"},
                    "dv_max_ms": {"type": "number", "description": "Maximum delta-v in m/s (default 2.0)"},
                    "grid_n": {"type": "integer", "description": "Grid resolution (default 7)"},
                },
                "required": ["norad_id", "tca_utc"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "risk_timeline",
            "description": "For a non-maneuverable satellite, produce a risk timeline, payload-safing window recommendation, and notification checklist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "norad_id": {"type": "integer", "description": "Satellite NORAD ID"},
                    "window_days": {"type": "number", "description": "Screening window in days (default 7)"},
                },
                "required": ["norad_id"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "screen": tool_screen,
    "compute_pc": tool_compute_pc,
    "rescreen": tool_rescreen,
    "risk_timeline": tool_risk_timeline,
}
