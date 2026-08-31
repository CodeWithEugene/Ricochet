"""
audit/logger.py
===============
Structured, replayable audit log for every tool call.

Every tool invocation in the AI agent produces one JSON file:
    audit/{event_id}.json

The file contains:
- event_id (UUID4)
- timestamp (ISO 8601 UTC)
- tool name
- inputs (exactly as passed)
- outputs (exactly as returned, serialized)
- model name and version
- duration_s

No numeric value may appear in a decision brief unless it also appears
in at least one audit/{event_id}.json for this session. The brief
validator enforces this.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and dataclasses to JSON-serializable forms."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _make_serializable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return str(obj)  # "nan" or "inf" — JSON doesn't support these
    return obj


def log_tool_call(
    tool: str,
    inputs: dict,
    outputs: Any,
    model_name: str = "",
    model_version: str = "",
    duration_s: float = 0.0,
    session_id: str = "",
) -> str:
    """
    Log a tool call to disk and return the event_id.
    
    Parameters
    ----------
    tool : str
        Name of the tool called.
    inputs : dict
        Inputs to the tool (will be serialized).
    outputs : Any
        Outputs from the tool (will be serialized).
    model_name : str
        LLM model name that triggered this tool call.
    model_version : str
        LLM model version.
    duration_s : float
        Wall-clock seconds the tool took to run.
    session_id : str
        Session identifier for grouping related events.
    
    Returns
    -------
    str
        event_id (UUID4 string) that can be used to retrieve the event.
    """
    cfg = _load_config()
    log_dir = Path(cfg["audit"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    event = {
        "event_id": event_id,
        "timestamp": timestamp,
        "tool": tool,
        "inputs": _make_serializable(inputs),
        "outputs": _make_serializable(outputs),
        "model_name": model_name,
        "model_version": model_version,
        "duration_s": duration_s,
        "session_id": session_id,
    }

    event_path = log_dir / f"{event_id}.json"
    with open(event_path, "w") as f:
        json.dump(event, f, indent=2 if cfg["audit"]["pretty_print"] else None)

    return event_id


def load_event(event_id: str) -> dict:
    """
    Load a logged event by event_id.
    
    Parameters
    ----------
    event_id : str
        UUID string of the event.
    
    Returns
    -------
    dict
        The full event record.
    
    Raises
    ------
    FileNotFoundError
        If no event with this ID exists.
    """
    cfg = _load_config()
    log_dir = Path(cfg["audit"]["log_dir"])
    event_path = log_dir / f"{event_id}.json"
    if not event_path.exists():
        raise FileNotFoundError(f"Audit event not found: {event_id}")
    with open(event_path) as f:
        return json.load(f)


def list_session_events(session_id: str) -> list[dict]:
    """
    Return all events for a given session_id, sorted by timestamp.
    """
    cfg = _load_config()
    log_dir = Path(cfg["audit"]["log_dir"])
    events = []
    for p in log_dir.glob("*.json"):
        with open(p) as f:
            ev = json.load(f)
        if ev.get("session_id") == session_id:
            events.append(ev)
    events.sort(key=lambda e: e["timestamp"])
    return events


def collect_all_numbers_from_session(session_id: str) -> set[float]:
    """
    Extract all numeric values from all tool outputs in a session.
    Used by the brief validator to check that briefed numbers are grounded.
    
    Returns a set of all float values (rounded to 6 significant figures
    to handle floating-point display differences).
    """
    events = list_session_events(session_id)
    numbers = set()
    
    def _extract_numbers(obj: Any) -> None:
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            # Round to 6 significant figures for comparison
            if obj != 0:
                mag = 10 ** (math.floor(math.log10(abs(obj))) - 5)
                numbers.add(round(obj / mag) * mag)
            else:
                numbers.add(0.0)
        elif isinstance(obj, dict):
            for v in obj.values():
                _extract_numbers(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _extract_numbers(item)
    
    for event in events:
        _extract_numbers(event.get("outputs", {}))
    
    return numbers


import math  # needed for collect_all_numbers_from_session
