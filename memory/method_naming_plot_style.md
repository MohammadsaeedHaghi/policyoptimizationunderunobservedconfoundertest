---
name: method_naming_plot_style
description: Canonical code-1.1 method names (X-X-X convention) after the 2026-06-18 rename + the color-by-family / shape-by-uncertainty-set plot scheme.
metadata: 
  node_type: memory
  type: project
  originSessionId: 58aeb24f-db15-4091-9dfb-a7f743c17425
---

**Canonical method names (code 1.1, as of 2026-06-18).** The `X-X-X` convention: 1st slot = objective/estimator,
2nd = `O` if the Tan/Rosenbaum odds-box is used else `X`, 3rd = `W` if the Wasserstein ball is used else `X`.

| current name | folder/fn stem | was (older shorthands) |
|---|---|---|
| `IPW-O-W` | `ipw_o_w` / `solve_ipw_o_w` | R-OW / RW |
| `IPW-O-X` | `ipw_o_x` | R-O / RW-noW |
| `IPW-X-X` | `ipw_x_x` | IPW |
| `Hajek-O-W` | `hajek_o_w` | **Regret-O-W** / Regret-OW / Kallus-W |
| `Hajek-O-X` | `hajek_o_x` | **Regret-O-X** / Regret-O / Kallus-odds |
| `DoublyRobust-X-X` | `doublyrobust_x_x` | AIPW / plain DoublyRobust |
| `DoublyRobust-O-W` | `doublyrobust_o_w` | R-OW-DR |
| `DoublyRobust-O-X` | `doublyrobust_o_x` | R-O-DR |
| `Direct-X-X` | `direct_x_x` | **Policy-Optimization** / Direct-Opt / Direct-Optimization |
| `Kallus` | `kallus` (flat, parametric) | — |
| `Oracle` | `oracle` | — |

**2026-06-18 rename** (`/tmp/rename_methods.py`): `Regret-O-{X,W}→Hajek-O-{X,W}` (the regret methods ARE the
self-normalised Hájek estimator — see [[regret_selfnorm_fix]]; the OBJECTIVE is still "regret", the name now reflects
the estimator), `Policy-Optimization→Direct-X-X`. Folders, `*_capped/_uncapped.py` files, classes
(`RegretOResult→HajekOResult`, `RegretOWResult→HajekOWResult`, `PolicyOptimizationResult→DirectResult`), funcs,
`run_experiment.REGISTRY`, `common/config.py` (KNOWN_METHODS/WASSERSTEIN_METHODS/MAXIMIZE_METHODS), tests, app, and
index.html all updated. 54 tests + 11/11 smoke PASS. Kallus untouched (user). Internal smoke-test display labels +
the exp_second heatmap `id`/`gridBy` keys still use old shorthand (invisible; displayed labels are new).

**Plot styling (index.html `renderChart`/`_csvg`, function `_mstyle`).** COLOUR = estimator family, SHAPE = uncertainty set:
- colour by 1st slot — `FAMILY_COLOR={IPW:#1f77b4, Hajek:#9467bd, DoublyRobust:#2ca02c, Direct:#ff7f0e, Kallus:#d62728, Oracle:#444}`.
- shape by suffix — `-X-X`→dashed line (no marker); `-O-X`→solid line + **squares**; `-O-W`→solid line + **circles**;
  Kallus→**triangles**; Oracle→dotted. Helpers `_mstyle(name,fb)`, `_mk(marker,..)`, `_mswatch(st)` (legend uses
  class `.msw`, NOT `.sw`). `_mstyle` keys off `se.label` (so chart-data series need the new name in `"label"`; the
  short `"id"` toggle-keys can stay). This SUPERSEDES the old "one canonical colour per method" — see [[html_design_system]].
- A "How to read every figure" legend box (`.lbx`) sits at the top of the methods-tab Overview subpanel explaining this.
