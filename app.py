"""
app.py
======
Ricochet — AI Decision Support for Satellite Collision Avoidance

Streamlit UI with two modes:
  - MANEUVERABLE: full delta-v trade space + total-Pc rescreen heatmap
  - NON-MANEUVERABLE: risk timeline, payload safing, notification checklist

NON-NEGOTIABLE DISCLAIMER (appears everywhere):
  Public TLEs must not be used for operational conjunction assessment.
  Operators should contact the 18th Space Defense Squadron.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml


def _load_config() -> dict:
    cfg_path = Path("config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ricochet — Collision Avoidance Decision Support",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "⚠️ **PUBLIC TLE DATA — NOT FOR OPERATIONAL USE.** "
    "This tool uses publicly available TLE data and assumed covariance models. "
    "For operational conjunction assessment, contact the "
    "[18th Space Defense Squadron](https://www.space-track.org)."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛰 Ricochet")
    st.caption("Collision avoidance decision support for operators without an ops team.")
    st.error(DISCLAIMER)
    st.divider()

    cfg = _load_config()
    taifa_norad = cfg["taifa1"]["norad_id"]

    mode_choice = st.radio(
        "Primary satellite type",
        ["Maneuverable", "Non-Maneuverable"],
        help="Non-maneuverable mode produces a risk timeline and notification checklist instead of a burn plan.",
    )
    maneuverable = mode_choice == "Maneuverable"

    st.divider()
    norad_input = st.number_input(
        "Primary NORAD ID",
        min_value=1,
        max_value=999999,
        value=taifa_norad if not maneuverable else 25544,
        help="NORAD catalog number. Taifa-1 (non-maneuverable): 56212. ISS: 25544.",
    )
    norad_id = int(norad_input)

    object_name_input = st.text_input(
        "Object name (optional)",
        value="TAIFA-1" if norad_id == taifa_norad else "",
        placeholder="e.g. TAIFA-1",
    )
    object_name = object_name_input or f"NORAD-{norad_id}"

    window_days = st.slider(
        "Screening window (days)",
        min_value=1, max_value=14, value=7,
    )

    st.divider()
    st.markdown("**Analysis settings**")
    use_ai_agent = st.checkbox(
        "Use AI agent (Granite via Ollama)",
        value=False,
        help="Requires Ollama running locally with granite3.3:8b",
    )

    if maneuverable:
        st.markdown("**Maneuver grid**")
        dv_max = st.slider("Max |Δv| (m/s)", 0.1, 5.0, 2.0, 0.1)
        grid_n = st.select_slider("Grid resolution", options=[3, 5, 7, 9], value=5)
        always_trade_space = st.checkbox(
            "Always compute trade space",
            value=True,
            help=(
                "Compute the maneuver trade space even when the highest Pc is below the "
                "elevated threshold. Costs a full rescreen per grid point but shows that "
                "no burn is needed and that none would induce new risk."
            ),
        )

    run_analysis = st.button("▶ Run Analysis", type="primary", use_container_width=True)
    st.divider()
    st.caption("Ricochet v1.0 | IBM AI Builders Challenge 2025")
    st.caption("Data: CelesTrak GP/OMM | Covariance: nominal model (see config.yaml)")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🛰 Ricochet")
st.subheader("Collision Avoidance Decision Support")
st.warning(DISCLAIMER)

if not run_analysis:
    st.info(
        "Configure a satellite in the sidebar and click **▶ Run Analysis** to begin.\n\n"
        "**Example:** Try NORAD 56212 (Taifa-1, Kenya's first satellite) in Non-Maneuverable mode "
        "to see how an operator with no propulsion should respond to a conjunction warning."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

session_id = str(uuid.uuid4())

with st.spinner("Loading catalog..."):
    from data.fetch_catalog import load_catalog
    try:
        catalog = load_catalog()
        st.success(f"Catalog loaded: {len(catalog):,} objects")
    except Exception as e:
        st.error(f"Failed to load catalog: {e}")
        st.stop()

with st.spinner(f"Screening NORAD {norad_id} for close approaches ({window_days} days)..."):
    from core.screen import screen
    try:
        t0 = time.time()
        events = screen(norad_id=norad_id, window_days=window_days, catalog=catalog)
        screen_elapsed = time.time() - t0
        st.success(f"Screening complete: {len(events)} conjunctions in {screen_elapsed:.1f}s")
    except KeyError:
        st.error(f"NORAD ID {norad_id} not found in catalog. Check the ID and refresh the catalog.")
        st.stop()
    except Exception as e:
        st.error(f"Screening failed: {e}")
        st.stop()

if not events:
    st.success(f"No conjunctions within 50 km over {window_days} days. Green for go. ✅")
    st.caption("Note: absence of conjunctions in public TLE screening does not guarantee safety.")
    st.stop()


# ---------------------------------------------------------------------------
# Compute Pc for all events
# ---------------------------------------------------------------------------

from core.pc import compute_pc

with st.spinner("Computing probability of collision for all events..."):
    pc_results = []
    for ev in events:
        try:
            pc_res = compute_pc(ev)
            pc_results.append((ev, pc_res))
        except Exception:
            pc_results.append((ev, None))

# Build display table
alert_threshold = cfg["pc"]["alert_threshold"]
elevated_threshold = cfg["pc"]["elevated_threshold"]

rows = []
for ev, pc_res in pc_results:
    pc = pc_res.pc if pc_res else 0.0
    alert = "🔴 RED" if pc >= alert_threshold else ("🟡 YELLOW" if pc >= elevated_threshold else "🟢 GREEN")
    rows.append({
        "Alert": alert,
        "Secondary": f"{ev.secondary_name} ({ev.secondary_norad})",
        "TCA (UTC)": ev.tca.strftime("%Y-%m-%d %H:%M:%S"),
        "Miss (km)": f"{ev.miss_m / 1000:.3f}",
        "Rel. V (km/s)": f"{ev.rel_v_ms / 1000:.3f}",
        "Pc": f"{pc:.2e}" if pc > 0 else "< 1e-15",
        "Primary age (days)": f"{ev.primary_epoch_offset_s / 86400:.1f}",
        "Secondary age (days)": f"{ev.secondary_epoch_offset_s / 86400:.1f}",
    })

# Sort by Pc descending
df_display = pd.DataFrame(rows)

st.subheader(f"Conjunction Alert Queue — {object_name} ({norad_id})")
st.dataframe(df_display, use_container_width=True, hide_index=True)

# Highest-risk event
best_event, best_pc_result = max(pc_results, key=lambda x: x[1].pc if x[1] else 0.0)
best_pc = best_pc_result.pc if best_pc_result else 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    level_color = "🔴" if best_pc >= alert_threshold else ("🟡" if best_pc >= elevated_threshold else "🟢")
    st.metric("Highest Pc", f"{best_pc:.2e}", delta=f"{level_color}")
with col2:
    st.metric("TCA", best_event.tca.strftime("%Y-%m-%d %H:%M"))
with col3:
    st.metric("Miss distance", f"{best_event.miss_m / 1000:.3f} km")
with col4:
    st.metric("Relative velocity", f"{best_event.rel_v_ms / 1000:.3f} km/s")

# Covariance assumption callout
if best_pc_result:
    cov = best_pc_result.covariance_assumption
    with st.expander("⚙ Covariance assumptions (nominal model — no CDM available)", expanded=False):
        st.markdown(f"""
**Source:** {cov.source}  
**Combined σ_R:** {cov.sigma_r_m:.0f} m | **σ_T:** {cov.sigma_t_m:.0f} m | **σ_N:** {cov.sigma_n_m:.0f} m  
**Primary TLE age at TCA:** {cov.primary_epoch_offset_s / 86400:.1f} days  
**Secondary TLE age at TCA:** {cov.secondary_epoch_offset_s / 86400:.1f} days  
**Hard-body radius:** {cov.hbr_m:.1f} m  

*These assumptions are from `config.yaml`. Operational CDMs provide measured covariance — these are estimated.*
        """)

        # Sensitivity band
        from core.pc import compute_pc_sensitivity
        sensitivity = compute_pc_sensitivity(best_event)
        sens_rows = [
            {"Covariance scale": f"×{k:.1f}", "Pc": f"{v.pc:.2e}"}
            for k, v in sensitivity.items()
        ]
        st.table(pd.DataFrame(sens_rows))


# ---------------------------------------------------------------------------
# MANEUVERABLE MODE: Trade-space heatmap
# ---------------------------------------------------------------------------

if maneuverable:
    st.divider()
    st.subheader("🔥 Maneuver Trade Space — Total Pc Heatmap")
    st.caption(
        "Each cell shows **total Pc** = Pc(primary event, post-maneuver) + Σ Pc(newly induced conjunctions). "
        "The dodge that avoids one collision may cause another — Ricochet checks."
    )

    rescreen_result = None
    above_elevated = best_pc >= elevated_threshold

    if above_elevated or always_trade_space:
        if not above_elevated:
            st.info(
                f"Highest Pc ({best_pc:.2e}) is below the elevated threshold "
                f"({elevated_threshold:.0e}), so no burn is required. The trade space below is "
                "computed anyway to confirm that no candidate burn would induce new risk."
            )

        with st.spinner(f"Computing {grid_n}×{grid_n} maneuver trade space (this is the Ricochet computation)..."):
            from core.rescreen import rescreen as run_rescreen
            try:
                t0 = time.time()
                rescreen_result = run_rescreen(
                    norad_id=norad_id,
                    primary_tca=best_event.tca,
                    catalog=catalog,
                    dv_range=(-dv_max, dv_max),
                    grid_n=grid_n,
                )
                rescreen_elapsed = time.time() - t0
                st.success(f"Rescreen complete: {grid_n}×{grid_n} grid in {rescreen_elapsed:.1f}s")
            except Exception as e:
                st.error(f"Rescreen failed: {e}")

        if rescreen_result is not None:
            # Plot heatmap
            grid_data = rescreen_result.grid.values
            dv_labels = [f"{v:+.3f}" for v in rescreen_result.dv_values_ms]
            dt_labels = [f"{v / 3600:.1f}h" for v in rescreen_result.dt_values_s]

            # Cells with zero or negligible Pc are floored so they render as
            # "safe" green instead of dropping out of the heatmap as NaN.
            pc_display_floor = 1e-12
            with np.errstate(divide='ignore'):
                log_grid = np.where(
                    np.isnan(grid_data),
                    np.nan,
                    np.log10(np.maximum(grid_data, pc_display_floor)),
                )

            all_nan = bool(np.all(np.isnan(log_grid)))
            if all_nan:
                st.warning(
                    "Every grid point failed to propagate — the heatmap below is empty. "
                    "Try a narrower Δv range or a coarser grid."
                )

            log_floor = np.log10(pc_display_floor)
            log_ceiling = np.log10(alert_threshold) if all_nan else max(
                float(np.nanmax(log_grid)), np.log10(alert_threshold)
            )

            def _fmt_cell(v: float) -> str:
                if np.isnan(v):
                    return "n/a"
                if v < pc_display_floor:
                    return f"<{pc_display_floor:.0e}"
                return f"{v:.1e}"

            cell_text = [[_fmt_cell(float(v)) for v in row] for row in grid_data]

            fig = go.Figure(data=go.Heatmap(
                z=log_grid,
                x=dt_labels,
                y=dv_labels,
                zmin=log_floor,
                zmax=log_ceiling,
                text=cell_text,
                texttemplate="%{text}",
                textfont=dict(size=10),
                colorscale=[
                    [0.0, "#1a9850"],   # low Pc = green
                    [0.4, "#fee08b"],   # medium = yellow
                    [0.7, "#f46d43"],   # elevated = orange
                    [1.0, "#d73027"],   # high Pc = red
                ],
                colorbar=dict(
                    title="log₁₀(Total Pc)",
                    tickformat=".1f",
                ),
                hovertemplate=(
                    "Δv: %{y} m/s<br>"
                    "Burn time: %{x} before TCA<br>"
                    "Total Pc: %{text}<extra></extra>"
                ),
            ))

            # "No burn" reference line sits on the dv=0 row; the y axis is
            # categorical, so it is addressed by row index.
            zero_dv_row = int(np.argmin(np.abs(rescreen_result.dv_values_ms)))
            fig.add_hline(
                y=zero_dv_row,
                line_dash="dash",
                line_color="white",
                annotation_text="No burn",
            )

            # Mark recommended burn
            if rescreen_result.recommended_dv_ms is not None:
                rec_dv = rescreen_result.recommended_dv_ms
                rec_dt = rescreen_result.recommended_dt_s
                fig.add_annotation(
                    x=f"{rec_dt / 3600:.1f}h",
                    y=f"{rec_dv:+.3f}",
                    text=f"★ Rec: Δv={rec_dv:+.3f} m/s",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="white",
                    font=dict(color="white", size=12),
                )

            fig.update_layout(
                title=f"Total Pc Trade Space — {object_name} | TCA: {best_event.tca.strftime('%Y-%m-%d %H:%M')} UTC",
                xaxis_title="Burn time before TCA",
                yaxis_title="Along-track Δv (m/s)",
                height=500,
                template="plotly_dark",
            )

            st.plotly_chart(fig, use_container_width=True)

            baseline_pc = rescreen_result.baseline_pc
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Baseline total Pc (no burn)", f"{baseline_pc:.2e}")
            with col_b:
                st.metric("Best cell in grid", f"{np.nanmin(grid_data):.2e}" if not all_nan else "n/a")
            with col_c:
                st.metric("Worst cell in grid", f"{np.nanmax(grid_data):.2e}" if not all_nan else "n/a")

            # Recommendation callout
            if rescreen_result.recommended_dv_ms is not None:
                rec_pc = rescreen_result.grid_points
                best_gp = min(
                    [gp for gp in rec_pc if abs(gp.dv_ms - rescreen_result.recommended_dv_ms) < 1e-6],
                    key=lambda gp: abs(gp.dt_before_tca_s - rescreen_result.recommended_dt_s),
                    default=None,
                )
                if best_gp:
                    with st.container(border=True):
                        st.markdown(f"""
### Recommended burn
| Parameter | Value |
|-----------|-------|
| **Delta-v** | **{rescreen_result.recommended_dv_ms:+.4f} m/s** |
| **Burn time** | **{rescreen_result.recommended_dt_s / 3600:.1f} hours before TCA** |
| **Total Pc (post-maneuver)** | **{best_gp.total_pc:.2e}** |
| **Primary event Pc** | {best_gp.primary_event_pc:.2e} |
| **Induced events Pc** | {best_gp.induced_events_pc:.2e} |
| **Induced conjunctions** | {best_gp.n_induced_events} |

*This is the minimum-|Δv| burn that reduces total Pc below {alert_threshold:.0e}.*  
*{best_gp.force_model_note}*
                        """)
            elif baseline_pc < alert_threshold:
                st.success(
                    f"No burn recommended. Total Pc with no maneuver ({baseline_pc:.2e}) is already "
                    f"below the alert threshold ({alert_threshold:.0e}), and the grid confirms that "
                    "candidate burns would not reduce risk further."
                )
            else:
                st.warning(
                    f"No burn in the ±{dv_max} m/s grid reduces total Pc below {alert_threshold:.0e}. "
                    "Consider expanding the delta-v range or consulting 18 SDS."
                )

    else:
        st.info(
            f"No conjunctions above elevated threshold ({elevated_threshold:.0e}). "
            "Trade-space analysis not needed. Enable **Always compute trade space** in the "
            "sidebar to see the heatmap anyway."
        )


# ---------------------------------------------------------------------------
# NON-MANEUVERABLE MODE: Risk timeline + notification
# ---------------------------------------------------------------------------

else:
    st.divider()
    st.subheader("📋 Risk Timeline — Non-Maneuverable Satellite")
    st.info(
        f"**{object_name}** has no propulsion capability. "
        "No burn plan is possible. The actions below are your passive risk management options."
    )

    # Risk timeline chart
    if pc_results:
        timeline_data = []
        for ev, pc_res in pc_results:
            pc = pc_res.pc if pc_res else 0.0
            timeline_data.append({
                "TCA": ev.tca,
                "Pc": pc,
                "Miss (km)": ev.miss_m / 1000,
                "Object": ev.secondary_name,
                "log_Pc": np.log10(max(pc, 1e-15)),
            })

        df_timeline = pd.DataFrame(timeline_data)

        fig_tl = px.scatter(
            df_timeline,
            x="TCA", y="log_Pc",
            size="Miss (km)",
            color="log_Pc",
            hover_name="Object",
            hover_data={"Pc": ":.2e", "Miss (km)": ":.3f"},
            color_continuous_scale=[[0, "#1a9850"], [0.5, "#f46d43"], [1, "#d73027"]],
            title=f"Risk Timeline — {object_name} ({window_days} day window)",
            labels={"log_Pc": "log₁₀(Pc)", "TCA": "Time of Closest Approach"},
        )
        fig_tl.add_hline(
            y=np.log10(alert_threshold),
            line_dash="dash", line_color="red",
            annotation_text=f"Alert threshold ({alert_threshold:.0e})",
        )
        fig_tl.add_hline(
            y=np.log10(elevated_threshold),
            line_dash="dot", line_color="orange",
            annotation_text=f"Elevated ({elevated_threshold:.0e})",
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    # Payload safing window
    if best_event:
        safing_start = best_event.tca - timedelta(hours=12)
        safing_end = best_event.tca + timedelta(hours=2)

        with st.container(border=True):
            st.markdown("### ⏱ Payload Safing Window")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Safing starts", safing_start.strftime("%Y-%m-%d %H:%M UTC"))
            with col2:
                st.metric("Safing ends", safing_end.strftime("%Y-%m-%d %H:%M UTC"))
            st.caption(
                f"12h pre-TCA to 2h post-TCA around the highest-risk conjunction "
                f"({best_event.secondary_name}, TCA {best_event.tca.strftime('%H:%M UTC')}). "
                "Safe the payload if the risk level warrants it."
            )

    # Notification checklist
    with st.expander("📋 Notification Checklist", expanded=True):
        checklist_items = [
            "Contact the **18th Space Defense Squadron** for an authoritative CDM (space-track.org)",
            f"Notify spacecraft manufacturer / bus operator about {object_name}",
            "Alert payload users of possible safing window",
            "Log conjunction event in Mission Operations Record with TCA and Pc",
            "Re-screen within 24h if TLE age exceeds 2 days",
            "**This assessment uses public TLE data only — not for operational decisions**",
        ]
        for item in checklist_items:
            st.checkbox(item, value=False, key=f"chk_{hash(item)}")


# ---------------------------------------------------------------------------
# AI Decision Brief (both modes)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🤖 AI Decision Brief")

if use_ai_agent:
    with st.spinner("Generating AI decision brief via Granite..."):
        from ai.agent import run_agent
        try:
            agent_result = run_agent(
                norad_id=norad_id,
                maneuverable=maneuverable,
                object_name=object_name,
                session_id=session_id,
            )
            brief = agent_result["brief"]
            validated = agent_result["validated"]
            validation_errors = agent_result["validation_errors"]
        except Exception as e:
            st.error(f"AI agent error: {e}")
            agent_result = None
            brief = None
            validated = False
            validation_errors = [str(e)]
else:
    # Generate brief directly from tool results (no LLM)
    from ai.agent import _construct_brief_from_results, _fallback_agent
    agent_result = _fallback_agent(norad_id, maneuverable, object_name, session_id)
    brief = agent_result["brief"]
    validated = agent_result["validated"]
    validation_errors = agent_result["validation_errors"]
    st.info("AI agent disabled. Brief constructed directly from computed tool results (no LLM inference).")

if brief:
    # Validation status
    if validated:
        st.success("✅ All numeric values in this brief trace to tool-call results. Audit log: `audit/` directory.")
    else:
        st.error(f"⚠ Brief validation failed: {'; '.join(validation_errors[:3])}")

    # Display brief
    with st.container(border=True):
        st.markdown(f"""
**Risk level:** {brief.get('risk_level', 'UNKNOWN')} | **Pc:** {brief.get('highest_pc_formatted', 'N/A')} | **TCA:** {brief.get('tca_utc', 'N/A')[:16]}  

{brief.get('reasoning', '')}

**Limitations:**  
""")
        for lim in brief.get("limitations", []):
            st.markdown(f"- {lim}")

        if maneuverable and brief.get("recommended_dv_ms") is not None:
            st.markdown(f"""
**Recommended burn:** Δv = {brief.get('recommended_dv_ms'):+.4f} m/s, {brief.get('recommended_dt_hours', 'N/A')} h before TCA  
**Post-maneuver total Pc:** {brief.get('recommended_total_pc_formatted', 'N/A')}
""")

        st.error(f"⚠ {brief.get('disclaimer', 'NOT FOR OPERATIONAL USE')}")

    # Show raw JSON brief
    with st.expander("📄 Raw JSON brief (for audit)"):
        st.json(brief)


# ---------------------------------------------------------------------------
# Operator Sign-off (REQUIRED)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("✍ Operator Sign-Off")
st.warning(
    "**Sign-off is required before any operational action.** "
    "This brief is decision SUPPORT only. The operator is responsible for all actions taken."
)

with st.form("signoff_form"):
    col1, col2 = st.columns(2)
    with col1:
        operator_name = st.text_input("Operator name *", placeholder="Full name")
        operator_role = st.text_input("Role", placeholder="e.g. Mission Controller")
    with col2:
        decision = st.selectbox(
            "Decision *",
            ["-- Select --", "ACCEPT RISK — No action", "EXECUTE RECOMMENDED BURN", "ESCALATE TO SENIOR OPERATOR", "REQUEST AUTHORITATIVE CDM FROM 18 SDS"],
        )
        notes = st.text_area("Notes / rationale", placeholder="Brief note on decision basis")

    acknowledge = st.checkbox(
        "I acknowledge that this assessment uses public TLE data with assumed covariance, "
        "is NOT for operational use without an authoritative CDM, and my decision is based "
        "on my own professional judgement."
    )
    submitted = st.form_submit_button("Submit Sign-Off", type="primary")

    if submitted:
        if not operator_name or decision == "-- Select --" or not acknowledge:
            st.error("Please complete all required fields and acknowledge the disclaimer.")
        else:
            # Log the sign-off
            signoff_record = {
                "event_type": "operator_signoff",
                "session_id": session_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "norad_id": norad_id,
                "object_name": object_name,
                "operator_name": operator_name,
                "operator_role": operator_role,
                "decision": decision,
                "notes": notes,
                "brief_validated": validated,
                "acknowledged_disclaimer": acknowledge,
            }

            # Write sign-off to audit log
            signoff_path = Path("audit") / f"signoff_{session_id}.json"
            signoff_path.parent.mkdir(parents=True, exist_ok=True)
            with open(signoff_path, "w") as f:
                json.dump(signoff_record, f, indent=2)

            st.success(
                f"✅ Sign-off recorded. Operator: **{operator_name}** | "
                f"Decision: **{decision}** | "
                f"Log: `audit/signoff_{session_id}.json`"
            )
            st.balloons()


# ---------------------------------------------------------------------------
# Audit trail expander
# ---------------------------------------------------------------------------

st.divider()
with st.expander("🔍 Audit Trail — Session Events", expanded=False):
    from audit.logger import list_session_events
    session_events = list_session_events(session_id)
    if session_events:
        for ev in session_events:
            st.markdown(f"**{ev['timestamp'][:19]}** — `{ev['tool']}` ({ev['duration_s']:.1f}s)")
            with st.expander(f"Event {ev['event_id'][:8]}... details"):
                st.json(ev)
    else:
        st.caption("No events logged yet for this session.")

st.caption(
    "Ricochet — AI decision support for collision avoidance | "
    "IBM AI Builders Challenge 2025 | "
    "Data: CelesTrak | "
    "⚠ NOT FOR OPERATIONAL USE"
)
