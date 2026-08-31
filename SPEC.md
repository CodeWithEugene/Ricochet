# Ricochet — Technical Spec

## What this is
Decision support for satellite collision avoidance, aimed at operators without a dedicated ops team (university cubesats, smallsat startups, emerging national agencies).

## The core insight
An avoidance maneuver changes the primary's trajectory for days afterward. That new trajectory can create NEW conjunctions with other catalog objects. Agencies re-screen post-maneuver ephemerides as standard practice; small operators have no tooling for it at all. Ricochet computes TOTAL probability of collision across the primary event plus all maneuver-induced events, over a grid of candidate burns.

## Hard constraints
- Python 3.11+. Deps: sgp4, numpy, scipy, pandas, requests, streamlit, plotly, pyyaml, ollama, pytest. Do not add others without asking.
- All frames in TEME. Never mix TEME and ECI without an explicit, commented conversion.
- All units SI internally (metres, m/s, seconds). Display units convert at the presentation layer only.
- Every physical assumption lives in config.yaml. Nothing hardcoded.
- The LLM computes NO physics. It selects tools, reads their numeric output, and drafts prose. Every number in any generated brief must trace to a logged tool call.

## Modules
1. data/fetch_catalog.py — CelesTrak GP/OMM JSON -> local parquet cache. Respect their usage policy. Cache with timestamp; never refetch within 4 hours.
2. core/screen.py — given a primary NORAD ID and a window (default 7d), find close approaches against the cached catalog. Pipeline: apogee/perigee pre-filter -> coarse SGP4 propagation (60s steps, SatrecArray, vectorised) -> keep range < 50 km -> refine to 1s around each local minimum -> return TCA, miss distance, relative velocity, relative state vectors.
3. core/pc.py — Foster/Alfano 2D probability of collision in the encounter B-plane. Public TLEs carry NO covariance, so covariance comes from a nominal model in config.yaml (RTN sigmas growing with time since epoch). Every Pc result must carry its covariance assumption in the return object.
4. core/maneuver.py — apply an along-track delta-v at time T before TCA; return the post-maneuver trajectory by numerical propagation (two-body + J2). The first-order analytic result (downtrack drift ~ 3 * dv * dt) is a TEST ORACLE ONLY, never the implementation.
5. core/rescreen.py — for a grid of (delta-v, burn time) candidates, re-run screening of the post-maneuver trajectory against the full catalog for 7 days after the burn. Return total_pc = Pc(primary event, post-maneuver) + sum(Pc of all induced events). Output a 2D grid suitable for a heatmap.
6. ai/tools.py + ai/agent.py — Granite 4.2 via Ollama, tool calling. Tools wrap modules 2-5. Agent emits a JSON decision brief validated against a schema. A validator rejects any brief containing a numeric value not present in the tool results.
7. audit/ — every tool call logged with inputs, outputs, timestamps, model name and version. One replayable JSON per event.
8. app.py — Streamlit. Alert queue -> event detail -> trade-space heatmap -> decision brief -> operator sign-off (recorded, required).

## Two operating modes
- MANEUVERABLE primary: full delta-v trade space + total-Pc re-screen.
- NON-MANEUVERABLE primary (e.g. Taifa-1, NORAD 56212, 3U cubesat, ~508 km SSO, no propulsion): no burn plan. Instead produce a risk timeline, payload-safing window, notification checklist, and a list of co-located maneuverable objects in the same shell.

## Non-negotiable disclaimer
Public TLEs must not be used for operational conjunction assessment. Operators should contact the 18th Space Defense Squadron. This string appears in the UI and the README.