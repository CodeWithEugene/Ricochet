# Ricochet — AI Decision Support for Satellite Collision Avoidance

🚀 **Live App:** [https://ricochet.streamlit.app/](https://ricochet.streamlit.app/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ricochet.streamlit.app/)

> **⚠ NON-NEGOTIABLE DISCLAIMER:** Public TLEs must not be used for operational conjunction assessment. Operators should contact the [18th Space Defense Squadron](https://www.space-track.org) for authoritative Conjunction Data Messages (CDMs).

---

## The problem

A satellite operator receives a conjunction warning. Standard practice is to compute a maneuver that avoids the threat. But that maneuver changes the satellite's trajectory for the next several days — and can fly it directly into *another* object.

This second-order problem — the maneuver that avoids one collision causing another — is documented in published NASA literature. Re-screening the post-maneuver ephemeris before executing a burn is standard practice at agencies with dedicated ops teams. For university cubesat programs, smallsat startups, and emerging national space agencies, there is no tooling for this at all.

**Ricochet** makes it accessible. It computes the *total probability of collision* — the primary event plus all maneuver-induced events — across a grid of candidate burns, and finds the minimum-Δv burn that genuinely clears your risk threshold.

---

## The core insight (one sentence)

> The maneuver that avoids one collision can cause another. Ricochet checks.

---

## Two operating modes

### Maneuverable primary
For satellites with propulsion:
- Screen against the full public catalog (7-day window)
- Compute Pc for each conjunction using Foster/Alfano 2D formulation
- For a grid of candidate burns (Δv × burn time), re-screen the post-maneuver trajectory against the full catalog
- Return **Total Pc = Pc(primary event, post-maneuver) + Σ Pc(newly induced events)**
- Visualise as a heatmap; recommend the minimum-|Δv| burn below threshold

### Non-maneuverable primary (e.g. Taifa-1, NORAD 56212)
Kenya's first satellite is a 3U cubesat with no propulsion. If it receives a conjunction warning, the operator cannot maneuver. Ricochet produces:
- Risk timeline of all approaching objects with Pc and TCA
- Payload-safing window (12h pre-TCA)
- Notification checklist (18 SDS, manufacturer, payload users)
- List of co-located objects that *could* maneuver

---

## Architecture

```
ricochet/
├── data/fetch_catalog.py   # CelesTrak GP/OMM → Parquet cache (4-hour TTL)
├── core/screen.py          # SGP4 screening: apogee/perigee filter → SatrecArray coarse → TCA refine
├── core/pc.py              # Foster/Alfano 2D Pc in encounter B-plane
├── core/maneuver.py        # Along-track Δv → two-body+J2 numerical propagation
├── core/rescreen.py        # ★ THE DIFFERENTIATOR: total-Pc trade space grid
├── ai/tools.py             # Tool wrappers (logged, JSON-serializable)
├── ai/agent.py             # Granite 4.2 tool-calling agent + brief validator
├── audit/logger.py         # Every tool call logged with inputs/outputs/timestamps
├── app.py                  # Streamlit UI: queue → event → heatmap → brief → sign-off
├── config.yaml             # ALL physical assumptions (nothing hardcoded)
└── tests/test_physics.py   # Physics oracle tests (not code-path tests)
```

---

## AI approach

**IBM Granite 3.3 8B** via Ollama, with tool calling.

The single most important design decision:

> **The language model never computes physics.** It selects tools, reads their numeric output, and drafts the decision brief. Every number in any generated brief must trace to a logged tool call. The brief is unsigned until a human operator signs it.

Concrete implementation:
- `validate_brief()` rejects any brief containing a numeric value not present in tool results
- Every tool call writes `audit/{event_id}.json` with inputs, outputs, model name, timestamp
- Operator sign-off is recorded to `audit/signoff_{session_id}.json`
- Covariance assumptions are printed on every Pc result (no silent assumptions)

---

## Covariance model

Public TLEs carry no covariance. Pc requires uncertainty estimates. Ricochet uses a nominal RTN growth model from `config.yaml`:

| Parameter | Value | Physical basis |
|-----------|-------|----------------|
| σ_R at epoch | 100 m | Radial TLE accuracy (Flohrer et al. 2008) |
| σ_T at epoch | 300 m | Along-track TLE timing error |
| σ_N at epoch | 50 m | Inclination/RAAN error, smaller |
| σ_T growth | 12 mm/s (~1 km/day) | B* drag coefficient uncertainty |

Both primary and secondary receive the same model. Combined covariance = C_primary + C_secondary.

**Every Pc result displays these assumptions.** Sensitivity bands (½×, 1×, 2×, 3× covariance) are computed automatically.

---

## Data sources

- **CelesTrak GP/OMM**: `celestrak.org/NORAD/elements/gp.php` — active objects, Starlink, debris. 4-hour TTL cache. User-Agent header set per their usage policy.
- **18th Space Defense Squadron** (space-track.org): authoritative CDMs. Not used by Ricochet — referenced as the correct operational resource.

---

## Prior art

This capability exists in operational form at NASA CARA, ESA CREAM, and commercial services (Kayhan Space, Slingshot, LeoLabs, COMSPOC). Ricochet does not claim to replace them. It is an open, explainable, free implementation for the long tail of operators — university cubesat programs, smallsat startups, emerging national agencies — who currently receive a CDM by email and have no tooling at all.

---

## Assumptions and limitations

- **Not for operational use.** Public TLE accuracy is ~1 km radial, ~10 km along-track at 1 day age.
- **Covariance is assumed, not measured.** Operational CDMs provide measured covariance; this tool uses a nominal model.
- **Force-model mismatch.** Post-maneuver primary is propagated with two-body + J2. Secondaries use SGP4 (drag, tesseral harmonics). Mismatch grows ~1-2 km/day at 500 km altitude over the 7-day window.
- **Independent-event Pc summation.** Total Pc = Σ Pc is valid only when individual Pc values are small (< ~0.01). Stated on every output.
- **B-plane approximation.** Covariance is propagated analytically (not via state transition matrix). Accurate for short prediction horizons.

---

## How IBM Bob was used

1. **Architect Mode** — fed `SPEC.md`, received critique of frame conventions, covariance model gaps, SGP4 vectorisation pattern, and silent unit-conversion bugs. Revised the spec before writing any code.
2. **Plan Mode** — module-by-module implementation planning with function signatures and external-reference test cases.
3. **Agent Mode** — implemented each module with physics oracle tests that verify against known-correct external values.
4. **Code Review** — agentic review + Semgrep scan before final commit.

---

## Live demo

🔗 **[https://ricochet.streamlit.app/](https://ricochet.streamlit.app/)**

Start with NORAD 56212 (Taifa-1) in Non-Maneuverable mode, then switch to NORAD 25544 (ISS) in Maneuverable mode to see the trade-space heatmap.

---

## Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Pull Granite for AI agent
ollama pull granite3.3:8b

# Run physics tests first — if these fail, do not trust the numbers
pytest tests/test_physics.py -v

# Launch the app
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Selected challenge theme

**Advance Space Exploration with AI** — IBM AI Builders Challenge, August 2025.

---

*Ricochet — the maneuver that avoids one collision can cause another. Ricochet checks.*
