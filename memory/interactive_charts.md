---
name: interactive-charts
description: "index.html interactive SVG charts — generalized renderChart with method-toggle + Γ-selector dropdowns, and the per-Γ data pipeline."
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

The index.html interactive charts (1-D 2-arm experiment, both subtabs) use a generalized JS renderer
**`renderChart(id, data)`** (evolved from the old policy-only `renderPolicyChart`). One self-contained SVG line
chart supporting: `ymin/ymax`, `ylabel`, `xticks` (explicit Γ labels), `hlines` (ceiling ref-lines, drawn only if
in-range), `gammaLine` (vertical matched-Γ marker), a **"Show methods ▾" toggle dropdown**, and an **optional
"Γ ▾" selector** (when `data.gammas` + `data.seriesByGamma` present). State in `CHART_REG[id]={data,hidden}`;
`setChartGamma(id,gKey,btn)` swaps `series=seriesByGamma[gKey]`, re-renders the `.pchart-wrap` svg, and re-applies
hidden methods. `toggleSeries` records `hidden` so visibility persists across Γ switches.

**Charts** (all via renderChart): `ichart-bern`/`ichart-uni` = policy π(treat|x) WITH Γ-selector; `ichart-kalcate`
= Kallus-DGP policy (no Γ); `ichart-rz-{bern,uni}-{train,test}` = realised E[Y] vs Γ (focused y, oracle in note,
best-means hline); `ichart-obj-{bern,uni}` = worst-case objective vs Γ. All have the method dropdown.

**Data pipeline** (`assets/exp_1d/`):
- `exp_1d_capture_grid.py {bern|uni} {METHOD}` — re-solves ONE method over the 6-Γ sweep, captures the deployed
  π(treat) on the 21-X grid per Γ, GATES the matched-Γ grid vs stored (Δ=0), writes `_bygamma/{mode}_{method}.json`.
  Ran all 9×2=18 in parallel via a Workflow (18/18 ok, all gates 0).
- `exp_1d_build_chartdata.py` — reads NPZ (realised/objective/ceilings) + `_bygamma/*.json` (per-Γ grids) →
  `_chartdata.json` (POLICY/REALIZED/OBJECTIVE) → spliced into index.html as `POLICY_DATA/REALIZED_DATA/OBJECTIVE_DATA`.

**CRITICAL GOTCHA:** `seriesByGamma` keys MUST equal JS `String(Number)` — JS `String(1.0)==="1"` but Python
`str(1.0)=="1.0"`. Use `gkey(g)=str(int(g)) if g==int(g) else str(g)` so keys are `["1","3","6","9","12.1825","16"]`,
matching `String(data.gammas[i])`. Mismatch → Γ-selector silently no-ops.

To verify charts: headless probe counts `g[data-method]` per chart-id (policy/realised/objective=9 series, kalcate=8),
checks `.gamma-dd-btn` presence (policy only), and tests `setChartGamma` changes a polyline (compare FULL points
string — the first 2 grid points are often identical, a short slice gives false negatives). See [[keep_html_in_sync]],
[[html_design_system]], [[wasserstein_2arm_cap]].
