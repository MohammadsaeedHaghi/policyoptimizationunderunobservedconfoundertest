---
name: secondexp-2d
description: "'Second experiment' = 2-arm 2-D discrete-X (R-OW wins) rebuilt in the exp_first structure; new interactive 2-D heatmap component."
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

**`assets/exp_second/`** — HTML Experiment subtab **"★ Second experiment · 2-D (R-OW wins)"** (`sub-exp_second`, pink #be185d,
2nd subtab button, after exp_first). The user asked to reproduce the OLD "2-arm 2-D" experiment (whose only artifact was
`assets/exp_2arm2d/` aggregate npz + 4 pngs, NO pipeline code) in the EXACT exp_first structure, named "exp second". User chose
(AskUserQuestion): **interactive 2-D heatmap** for the policy chart + **9 methods, n_train=500, 5 seeds, both regimes**.

**DGP (`dgp.py`, from the old tab's eqcard):** X=(X1,X2)~Unif 11×11 grid [-1,1]²; S~Bern(½) UNOBSERVED. proj=X1+0.6X2.
P(Y⁰=1)=σ(0.5) (FLAT clean control, no X/S); P(Y¹=1|X,S)=σ(-1.2+4·proj+5S); logit P(T=1|X,S)=-proj+γ(S-½) (MIS-targeted:
treats where proj LOW; +S channel). γ=5, matched Γ=e^{2.5}≈12.18, sweep {1,3,6,9,12.18,16}. Optimal: treat proj>0.04 (~47% of cells).
E[Y]: never=0.622, treat-all=0.615, optimal=0.732. GRID=(121,2). Regimes: UNCAP (1,1) + CAP (1,0.5) (cap "loose" — just above 47%).

**GOTCHA (z-score ↔ epsilon unboundedness):** the 2-D Wasserstein ground cost is z-scored (`zscore=True`). Do NOT pass an external
`epsilon=` computed from raw-X distances — it mismatches the solver's z-scored D and makes the inner LP UNBOUNDED (Gurobi status 5).
Fix: omit `epsilon`, let each Wasserstein solver compute its tight ε internally from its own z-scored D (c_eps default=1.0). (exp_first
passed epsilon externally because zscore=False there.)

**RESULT (5 seeds, realised TEST @matched Γ):** UNCAP — DR strongest (R-O-DR 0.683, AIPW 0.681, R-OW-DR 0.674), R-OW 0.670 > IPW
0.657 > R-O 0.655 (Wasserstein edge +0.015). **CAP — R-OW BEST 0.684** > R-OW-DR 0.677 > AIPW 0.673 > R-O-DR/IPW 0.658 > R-O 0.645
(**Wasserstein edge R-OW−R-O = +0.039**, ≈ the original 5-method finding's +0.042). best-means 0.733, full-info 0.750. Faithfully
reproduces "R-OW > IPW > R-O" + the Wasserstein edge; AIPW (new vs original 5 methods) competitive but R-OW tops it capped.

**Pipeline:** `dgp.py` · `ground_truth_plots.py` → 5 2-D-heatmap DGP plots (potential / average_potential / assignment_rule /
inverse_propensity / whats_seen; matplotlib mathtext: use `\frac{1}{2}` NOT `\tfrac`) · `run_multiseed.py {seed}` (both regimes,
9 methods incl Kallus + constrained-oracle ceilings, nearest-train deploy via argmin Euclidean, per-seed 121-grid policies ALL seeds) ·
`perf_plots.py` → per-regime realized_vs_gamma / realized_bar / treat_fraction / objective_vs_gamma + exp_second_results.json ·
`build_heatmap_chart.py` → HEATMAP_SECOND_{UNCAP,CAP} + splice + regenerate `_exp_second_view.html`. Run via `xargs -P 3`.

**NEW interactive 2-D heatmap component in index.html (reusable):** `renderHeatmap(id,D)` + `_hmsvg` (SVG 11×11 colored rects +
colorbar, purple colormap `_hmcolor`) + `_hmDD` (dropdown builder) + `setHmMethod/setHmGamma/setHmSeed` + `_hmupd/_hmSel`. D fields:
{ax1,ax2, methods:[{id,label}], gammas, defaultGamma, matchedGamma, seeds:[{key,label}] (Seed 0..4 + Average, defaultSeed:"avg"),
gridBy:{seedkey:{gkey:{method:[121 vals]}}}, note}. THREE dropdowns (method · Γ · seed) swap ONE heatmap; reuses .chart-dd styling +
toggleDD; scoped per-id so no clash with the line charts' gamma/seed dropdowns. gkey(g)=str(int(g)) if int else str(g). Heatmap vals
rounded to 2 dec (data is big: 11×6×6×121×2 ≈ index.html +360KB, viewer ~900KB — acceptable).

**HTML:** exp_first structure exactly — §1 DGP (eqcard + 5 heatmaps) shared; §2 Performance split via `showInner` into
**Uncapped/Capped inner-subtabs** (`second-uncap`/`second-cap`), each = interactive heatmap chart + realized_vs_gamma + realized_bar +
treat_fraction + objective_vs_gamma; finding card shared below. 0 mjx-merror; all 5 chart families on the site coexist + render.
See [[firstexp_case4]] (the structure template + showInner + seed-selector), [[keep_html_in_sync]] (build_heatmap_chart regenerates the viewer).
