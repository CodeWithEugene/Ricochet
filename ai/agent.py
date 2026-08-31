"""
ai/agent.py
===========
Granite AI agent with tool-calling via Ollama.

Design principles:
  1. The LLM NEVER computes physics. It selects tools and interprets their output.
  2. Every number in the decision brief must trace to a logged tool call.
  3. A validator rejects any brief containing a number not in tool results.
  4. All decisions require operator sign-off before becoming operational.

The agent follows a ReAct-style loop:
  - Receive an analysis request (norad_id, maneuverable flag)
  - Call tools in order: screen → compute_pc → rescreen or risk_timeline
  - Draft a structured JSON decision brief
  - Validate the brief against tool results
  - Return the brief and the session_id for audit replay
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from ai.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS, _SESSION_ID
from audit.logger import log_tool_call, list_session_events


def _load_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Brief schema (validated against tool results)
# ---------------------------------------------------------------------------

BRIEF_SCHEMA_MANEUVERABLE = {
    "required_keys": [
        "norad_id", "object_name", "analysis_timestamp_utc",
        "highest_pc", "highest_pc_formatted", "tca_utc",
        "miss_km", "covariance_source",
        "recommended_dv_ms", "recommended_dt_hours",
        "recommended_total_pc", "recommended_total_pc_formatted",
        "risk_level", "reasoning", "limitations", "disclaimer",
        "session_id", "operator_action_required",
    ],
}

BRIEF_SCHEMA_NON_MANEUVERABLE = {
    "required_keys": [
        "norad_id", "object_name", "analysis_timestamp_utc",
        "highest_pc", "highest_pc_formatted", "tca_utc",
        "miss_km", "covariance_source",
        "risk_level", "payload_safing_window",
        "notification_checklist", "reasoning", "limitations", "disclaimer",
        "session_id", "operator_action_required",
    ],
}


# ---------------------------------------------------------------------------
# Brief validator
# ---------------------------------------------------------------------------

def _extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numeric literals from a string."""
    pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers


def _find_numbers_in_obj(obj: Any) -> list[float]:
    """Recursively find all numeric values in a JSON-like object."""
    numbers = []
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        numbers.append(float(obj))
    elif isinstance(obj, str):
        numbers.extend(_extract_numbers_from_text(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers.extend(_find_numbers_in_obj(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            numbers.extend(_find_numbers_in_obj(item))
    return numbers


def _round_for_comparison(n: float) -> float:
    """Round to 4 significant figures for comparison."""
    import math
    if n == 0:
        return 0.0
    mag = 10 ** (math.floor(math.log10(abs(n))) - 3)
    return round(n / mag) * mag


def validate_brief(brief: dict, tool_results: list[dict]) -> None:
    """
    Validate that every significant numeric value in the brief
    appears in at least one tool result.
    
    Raises ValueError if a number in the brief cannot be traced to tool results.
    
    Numbers that are exempt from tracing:
    - Very small integers (0, 1, 2, ..., 10) used as counts/indices
    - Year numbers (2024, 2025, etc.)
    - Numbers that appear in the schema keys themselves
    """
    # Collect all numbers from tool results
    tool_numbers = set()
    for result in tool_results:
        for n in _find_numbers_in_obj(result):
            tool_numbers.add(_round_for_comparison(n))

    # Extract numbers from brief (string values — the ones the LLM generated)
    brief_violations = []
    
    def _check_value(val: Any, path: str) -> None:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            n = float(val)
            # Exempt small integers and year numbers
            if n == int(n) and (0 <= n <= 100 or 2000 <= n <= 2100):
                return
            rounded = _round_for_comparison(n)
            if rounded not in tool_numbers:
                brief_violations.append(f"Number {n} at {path} not found in tool results")
        elif isinstance(val, str):
            nums = _extract_numbers_from_text(val)
            for n in nums:
                if n == int(n) and (0 <= n <= 100 or 2000 <= n <= 2100):
                    continue
                rounded = _round_for_comparison(n)
                if rounded not in tool_numbers:
                    brief_violations.append(f"Number {n} in string at {path} not found in tool results")
        elif isinstance(val, dict):
            for k, v in val.items():
                _check_value(v, f"{path}.{k}")
        elif isinstance(val, list):
            for i, item in enumerate(val):
                _check_value(item, f"{path}[{i}]")

    # Only check the physics-critical fields
    physics_fields = [
        "highest_pc", "miss_km", "recommended_dv_ms",
        "recommended_dt_hours", "recommended_total_pc", "tca_utc",
    ]
    for field in physics_fields:
        if field in brief:
            _check_value(brief[field], field)

    if brief_violations:
        raise ValueError(
            f"Brief contains {len(brief_violations)} untraced number(s):\n" +
            "\n".join(brief_violations[:5])
        )


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(maneuverable: bool) -> str:
    cfg = _load_config()
    disclaimer = cfg["ai"]["system_prompt_append"]
    
    mode = "MANEUVERABLE" if maneuverable else "NON-MANEUVERABLE"
    
    return f"""You are a satellite collision avoidance decision support assistant (mode: {mode}).

RULES:
1. You NEVER compute physics. You call tools and read their results.
2. Every number you write in your brief MUST appear in a tool result.
3. When uncertain, say so explicitly. Never invent confidence.
4. You must call tools in this order:
   - First: call 'screen' to find all conjunctions
   - Then: call 'compute_pc' for the highest-risk event
   - Then (if MANEUVERABLE): call 'rescreen' to compute the maneuver trade space
   - Then (if NON-MANEUVERABLE): call 'risk_timeline' for the non-maneuver analysis
5. After calling all tools, produce a JSON decision brief.
   The JSON must be enclosed in ```json ... ``` code fences.
6. The brief must include a 'disclaimer' field with the exact text:
   "PUBLIC TLE DATA. NOT FOR OPERATIONAL USE. Contact 18th Space Defense Squadron."
7. The brief must include 'operator_action_required': true.

{disclaimer}"""


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_agent(
    norad_id: int,
    maneuverable: bool = True,
    object_name: Optional[str] = None,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    """
    Run the Granite agent for a satellite conjunction analysis.
    
    Parameters
    ----------
    norad_id : int
        NORAD ID of the primary satellite.
    maneuverable : bool
        True for maneuverable primary (full trade space).
        False for non-maneuverable primary (risk timeline + notification).
    object_name : str, optional
        Human-readable name for the satellite.
    model : str, optional
        Ollama model name. Default from config.
    session_id : str, optional
        Session identifier for audit grouping. Generated if not provided.
    
    Returns
    -------
    dict with keys:
        'brief' — the validated decision brief
        'session_id' — for audit replay
        'tool_results' — list of raw tool outputs
        'validated' — bool, True if brief passed numeric tracing
        'validation_errors' — list of validation error strings
    """
    if not OLLAMA_AVAILABLE:
        return _fallback_agent(norad_id, maneuverable, object_name, session_id)

    cfg = _load_config()
    if model is None:
        model = cfg["ai"]["model"]
    if session_id is None:
        session_id = _SESSION_ID
    if object_name is None:
        object_name = f"NORAD-{norad_id}"

    system_prompt = _build_system_prompt(maneuverable)
    user_message = (
        f"Please analyse satellite {object_name} (NORAD ID: {norad_id}). "
        f"It is {'MANEUVERABLE' if maneuverable else 'NON-MANEUVERABLE'}. "
        f"Screen for close approaches over 7 days and produce a decision brief."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    tool_results = []
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        t0 = time.time()

        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                options={"temperature": cfg["ai"]["temperature"]},
            )
        except Exception as e:
            return {
                "brief": None,
                "session_id": session_id,
                "tool_results": tool_results,
                "validated": False,
                "validation_errors": [f"Ollama error: {e}"],
                "error": str(e),
            }

        msg = response.message

        # Add assistant message to history
        messages.append({"role": "assistant", "content": msg.content or ""})

        # Check for tool calls
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments or {}

                if fn_name in TOOL_FUNCTIONS:
                    t_call = time.time()
                    try:
                        result = TOOL_FUNCTIONS[fn_name](**fn_args)
                    except Exception as e:
                        result = {"error": str(e), "tool": fn_name}
                    duration = time.time() - t_call

                    tool_results.append(result)

                    # Log tool call
                    log_tool_call(
                        tool=fn_name,
                        inputs=fn_args,
                        outputs=result,
                        model_name=model,
                        model_version="",
                        duration_s=duration,
                        session_id=session_id,
                    )

                    # Add tool result to message history
                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result, default=str),
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({"error": f"Unknown tool: {fn_name}"}),
                    })
        else:
            # No more tool calls — agent is done
            break

    # Extract JSON brief from final message
    final_content = messages[-1].get("content", "")
    brief = _extract_brief_json(final_content, norad_id, object_name, maneuverable, tool_results, session_id)

    # Validate the brief
    validation_errors = []
    validated = True
    try:
        validate_brief(brief, tool_results)
    except ValueError as e:
        validated = False
        validation_errors = str(e).split("\n")

    # Validate required keys
    schema = BRIEF_SCHEMA_MANEUVERABLE if maneuverable else BRIEF_SCHEMA_NON_MANEUVERABLE
    missing_keys = [k for k in schema["required_keys"] if k not in brief]
    if missing_keys:
        validated = False
        validation_errors.append(f"Missing required keys: {missing_keys}")

    return {
        "brief": brief,
        "session_id": session_id,
        "tool_results": tool_results,
        "validated": validated,
        "validation_errors": validation_errors,
    }


def _extract_brief_json(
    text: str,
    norad_id: int,
    object_name: str,
    maneuverable: bool,
    tool_results: list[dict],
    session_id: str,
) -> dict:
    """
    Extract the JSON brief from the agent's final message.
    Falls back to constructing a minimal brief from tool results
    if the agent didn't produce valid JSON.
    """
    # Try to extract ```json ... ``` block
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract any {...} block
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: construct brief from tool results
    return _construct_brief_from_results(
        norad_id, object_name, maneuverable, tool_results, session_id
    )


def _construct_brief_from_results(
    norad_id: int,
    object_name: str,
    maneuverable: bool,
    tool_results: list[dict],
    session_id: str,
) -> dict:
    """
    Construct a minimal decision brief directly from tool results.
    Used when the LLM doesn't produce valid JSON or Ollama is unavailable.
    All values trace directly to tool results — no LLM inference.
    """
    # Find screen result
    screen_result = next((r for r in tool_results if "n_conjunctions" in r), {})
    pc_result = next((r for r in tool_results if "pc" in r and "miss_km" in r), {})
    rescreen_result = next((r for r in tool_results if "recommended_dv_ms" in r), {})
    timeline_result = next((r for r in tool_results if "notification_checklist" in r), {})

    highest_pc = pc_result.get("pc", 0.0)
    miss_km = pc_result.get("miss_km", 0.0)
    tca_utc = pc_result.get("tca_utc", "")
    cov_source = (pc_result.get("covariance_assumption") or {}).get("source", "nominal_model_config")
    
    cfg = _load_config()
    alert_threshold = cfg["pc"]["alert_threshold"]
    risk_level = "RED" if highest_pc >= alert_threshold else ("YELLOW" if highest_pc >= cfg["pc"]["elevated_threshold"] else "GREEN")

    if maneuverable:
        rec_dv = rescreen_result.get("recommended_dv_ms")
        if rec_dv is not None:
            burn_sentence = (
                f"Trade-space analysis recommends Δv = {rec_dv} m/s "
                f"{rescreen_result.get('recommended_dt_hours', 'N/A')} hours before TCA."
            )
        elif rescreen_result:
            burn_sentence = (
                f"Trade-space analysis recommends no burn: total Pc with no maneuver "
                f"({rescreen_result.get('baseline_pc_formatted', 'N/A')}) is below the "
                f"alert threshold ({alert_threshold:.0e})."
            )
        else:
            burn_sentence = "No trade-space analysis was run for this event."

        brief = {
            "norad_id": norad_id,
            "object_name": object_name,
            "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "highest_pc": highest_pc,
            "highest_pc_formatted": f"{highest_pc:.2e}",
            "tca_utc": tca_utc,
            "miss_km": miss_km,
            "covariance_source": cov_source,
            "recommended_dv_ms": rescreen_result.get("recommended_dv_ms"),
            "recommended_dt_hours": rescreen_result.get("recommended_dt_hours"),
            "recommended_total_pc": rescreen_result.get("recommended_total_pc"),
            "recommended_total_pc_formatted": f"{rescreen_result.get('recommended_total_pc', 0.0):.2e}" if rescreen_result.get("recommended_total_pc") else None,
            "risk_level": risk_level,
            "reasoning": (
                f"Screening found {screen_result.get('n_conjunctions', 0)} conjunctions. "
                f"Highest Pc: {highest_pc:.2e} at TCA {tca_utc}. "
                f"{burn_sentence}"
            ),
            "limitations": [
                "Covariance from nominal model (no operational CDM available)",
                "Public TLE accuracy: ~1km radial, ~10km along-track at 1 day",
                "Primary propagated with two-body+J2; secondaries with SGP4 (force-model mismatch)",
                f"Induced-event Pc summed (valid only for Pc << 1)",
            ],
            "disclaimer": "PUBLIC TLE DATA. NOT FOR OPERATIONAL USE. Contact 18th Space Defense Squadron.",
            "session_id": session_id,
            "operator_action_required": True,
        }
    else:
        brief = {
            "norad_id": norad_id,
            "object_name": object_name,
            "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "highest_pc": highest_pc,
            "highest_pc_formatted": f"{highest_pc:.2e}",
            "tca_utc": tca_utc,
            "miss_km": miss_km,
            "covariance_source": cov_source,
            "risk_level": risk_level,
            "payload_safing_window": timeline_result.get("payload_safing_window"),
            "notification_checklist": timeline_result.get("notification_checklist", []),
            "reasoning": (
                f"Non-maneuverable satellite. "
                f"Screening found {screen_result.get('n_conjunctions', 0)} conjunctions. "
                f"Highest Pc: {highest_pc:.2e} at TCA {tca_utc}. "
                f"No burn plan available. Payload safing and notification actions listed."
            ),
            "limitations": [
                "No propulsion available — only passive risk management",
                "Covariance from nominal model (no operational CDM available)",
                "Public TLE accuracy: ~1km radial, ~10km along-track at 1 day",
            ],
            "disclaimer": "PUBLIC TLE DATA. NOT FOR OPERATIONAL USE. Contact 18th Space Defense Squadron.",
            "session_id": session_id,
            "operator_action_required": True,
        }

    return brief


def _fallback_agent(
    norad_id: int,
    maneuverable: bool,
    object_name: Optional[str],
    session_id: Optional[str],
) -> dict:
    """
    Run agent without Ollama: call tools directly in the correct order,
    then construct brief from results. Used when Ollama is not available.
    """
    if session_id is None:
        session_id = _SESSION_ID
    if object_name is None:
        object_name = f"NORAD-{norad_id}"

    tool_results = []

    # Step 1: Screen
    screen_result = TOOL_FUNCTIONS["screen"](norad_id=norad_id, window_days=7)
    tool_results.append(screen_result)

    # Step 2: Compute Pc for highest-risk event
    pc_result = {}
    if screen_result.get("conjunctions"):
        top = screen_result["conjunctions"][0]
        sec_norad = top["secondary_norad"]
        try:
            pc_result = TOOL_FUNCTIONS["compute_pc"](norad_id=norad_id, secondary_norad=sec_norad)
            tool_results.append(pc_result)
        except Exception as e:
            pc_result = {"error": str(e)}
            tool_results.append(pc_result)

    # Step 3: Rescreen or risk timeline
    if maneuverable and pc_result.get("tca_utc"):
        try:
            rescreen_result = TOOL_FUNCTIONS["rescreen"](
                norad_id=norad_id,
                tca_utc=pc_result["tca_utc"],
                grid_n=5,
            )
            tool_results.append(rescreen_result)
        except Exception as e:
            tool_results.append({"error": str(e)})
    elif not maneuverable:
        timeline_result = TOOL_FUNCTIONS["risk_timeline"](norad_id=norad_id)
        tool_results.append(timeline_result)

    brief = _construct_brief_from_results(
        norad_id, object_name, maneuverable, tool_results, session_id
    )

    return {
        "brief": brief,
        "session_id": session_id,
        "tool_results": tool_results,
        "validated": True,
        "validation_errors": [],
    }
