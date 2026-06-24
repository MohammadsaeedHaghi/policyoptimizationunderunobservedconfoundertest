---
name: discrete_multiarm_study
description: Discrete-X multi-arm study where RW beats RW−Wasserstein and Direct IPW; new K-arm core.
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

**MAJOR UPDATE 2026-06-09 — ported to the conti_binary plot design (user request).** The three settings —
**★ Unconstrained / ★ Capacity-constrained / ★ Capacity·stronger-γ=5** — are now ★-STARRED and follow the
binary study's design for ALL THREE cases (exp_a/b/c):
- **Kallus integrated into `discrete/run.py`** (solve_kallus_w_multiarm + solve_kallus_multiarm in the Γ loop) →
  the main CSV now carries every method; `kallus_run.py` is LEGACY (the dedicated "Kallus (regret)" view still
  reads the old `kallus_*.csv`, left intact).
- **Oracle renamed → "Full info"** (it was ALREADY the clairvoyant: `oracle_policy`/`solve_oracle_capacity` on
  Ypot, per-X-group) + NEW **"Best (true means)"** reference (best deployable policy on true μ via
  `dgps.true_arm_means`); both emitted as rows. Black `#111` / green `#2ca02c`.
- **E[Y] + usage columns** added to every row (exp_train/exp_test on μ; realized_train_dep; train_use_k/test_use_k).
- **New plots** (mirror binary): `plot_bands` is now **1×3** (realised train/test + worst-case obj on BASES only;
  dropped the old panel-(d)); NEW `plot_deploy` (2×2 realised + E[Y]); NEW `plot_policy_strip` (the deployed-policy
  plot = a **1-D per-method "chosen arm vs X" strip grid**, the analogue of the binary heatmap, with reference
  rows Best-arm/Best-true-means/Full-info). Artifacts `{name}_{sfx}_deploy.png` / `_policy.png`.
- **build_html**: COL/ORDER + CCOL/CORDER expanded & renamed; the three variant views + Compare-all READ THE MAIN
  CSV (dropped the kallus merge); `_deploy_card`+`_policy_card` added to `_variant_panel`; the three views starred.
- Re-ran at **5 reps** (was 10) — Kallus-W's Wasserstein LP roughly doubles per-rep cost.
- **exp_c N=1000 variant DROPPED**: at n=1000 the RW/Kallus-W Wasserstein LP (1000×1000 distance matrix) is
  prohibitively slow once Kallus is inline (~20 min/rep) — `hi-n1000` removed from `_variants_for`. Re-add only
  with a grid-snapped LP support. Tests: `tests/test_discrete_port.py` (2 pass; 60 total). Everything below this
  block predates the port — numbers/2×2-figure/10-seed/Oracle-naming/n1000 references are SUPERSEDED.

---

Restructured experiment study (2026-06-04): **discrete X, no policy extension**, three
**multi-arm** DGPs with **Rosenbaum/Tan S-channel** confounding, comparing **RW (Wass+odds)**
vs **RW−Wasserstein (odds only)** vs **Direct IPW**, with a test-set **Oracle** ceiling.
Each experiment now runs in **two modes** — unconstrained and **capacity-constrained** — giving
6 sub-sub-tabs in the HTML Discrete-X study.

**γ_true bumped 2.0 → 3.0 (2026-06-04):** trends are OPPOSITE for the two modes — the no-cap
Wasserstein gap shrinks with γ_true, the cap gap grows with it. γ_true=3.0 is the common level
where **RW > RW−noW > Direct IPW in BOTH modes**. c_ε=1 stays best (bigger radius washes out the
gain). Matched sensitivity is now Γ=e^{γ/2}=e^{1.5}≈**4.48** (the figure marker); e^{γ}=e^3≈20 is
no longer auto-added to the grid (too far out — run.py only adds it when ≤12).

**Headline (N_train=500, N_test=2000, 10 seeds, realised test outcome at the matched Γ≈4.48):**
- NO-CAP: exp_a RW 0.684 > noW 0.672 (+0.012) > DI 0.655; exp_b 0.705 > 0.687 (+0.018) > 0.668;
  exp_c 0.698 > 0.676 (+0.022) > 0.656.
- CAP:    exp_a RW 0.663 > noW 0.651 (+0.012) > DI 0.640; exp_b 0.672 > 0.661 (+0.011) > 0.647;
  exp_c 0.666 > 0.655 (+0.011) > 0.637.
RW wins in all 6 panels. Past the matched Γ (≥7) RW−noW catches up as the odds box keeps widening
and RW becomes over-conservative — RW's edge is sharpest at the correctly-specified sensitivity.

**Bigger-γ_true cap variant added (2026-06-04, user request):** a THIRD view per experiment — cap at
**γ_true=5.0** (matched Γ=e^2.5≈12.18) — kept ALONGSIDE the γ=3 ones (NOT replacing). 9 sub-sub-tabs
total now (exp×{no-cap, cap, cap-γ5}). At RW's operating peak (Γ≈3–5) RW wins big: exp_a +0.042 vs
RW−noW (Γ3), exp_b +0.024 (Γ5), exp_c +0.022 (Γ5). KEY: at strong confounding RW's realised peak sits
WELL BELOW the (large) matched Γ — at Γ=12.18 RW is past peak and RW−noW nearly catches up, so the
γ=5 HTML headline is anchored at RW's **best Γ** (not matched), and the γ=5 figures show a **red
dashed line at that selected operating Γ** (Γ=3/5/5 for exp_a/b/c) in addition to the dotted matched
line — `plot_bands(..., select_gamma=)` draws it (default = RW's peak; skipped when ≈matched, so the
γ=3 figures keep only the dotted line). `build_html.GAMMA_HI=5.0`,
`build_html` panels carry per-tab (cap, γ_true, anchor∈{matched,best}); `run_all.py --gamma-true`
overrides OPERATING_GAMMA_TRUE; artifacts `out/<dgp>_cap/<dgp>_gt5_00.*`. (γ=5 cap run took 38min —
the grid to 13 incl. matched 12.18 makes the large-Γ Wasserstein LPs heavy.)

**Capacity setting (user-requested):** caps come from the **historical train shares** — control
(arm 0) is UNCONSTRAINED (cap=1.0), each active arm k is capped at P̂(T=k) (e.g. exp_a caps ≈
[1, 0.17, 0.41]). Constraint `(1/n)Σ_i π_k(X_i) ≤ cap_k`; overflow routes to control (always
feasible since Σcap≥1). All three methods already accept `cap`; the **Oracle becomes the
capacity-constrained ceiling** via new `srpo.multiarm.solve_oracle_capacity` (LP: max realised
outcome with full Ypot s.t. caps). At Γ=1 the three still coincide in OBJECTIVE VALUE (the policy
can differ — a binding cap makes the LP degenerate). Run with `--cap-mode data`; artifacts go to
`out/<dgp>_cap/`. Tests: `test_capacity_constraint_binds_and_nests_at_gamma1`,
`test_oracle_capacity_respects_caps_and_dominates` (37 multiarm tests pass).

**Key empirical finding:** the Wasserstein covariate-balance edge over the odds-box is a
**multi-arm / hard-estimation effect** — it grows with the number of arms and vanishes in
data-rich **2-arm** settings (with 2 arms the per-arm Hájek calibration already does the
balancing; confirmed RW≈RW-noW at N=500 for every 2-arm DGP and grid resolution tried). So the
study was made all-multi-arm. Also: in this **discrete multi-arm** setting RW **is Γ-responsive**
(rises toward the oracle as Γ grows) — unlike the continuous binary cases ([[rw_gamma_invariance]]).
The DGP needs **asymmetric** confounding (S inflates some arms more than others, aligned with the
S→treatment coupling) or Direct IPW is unaffected and even wins.

**Direct IPW definition (important for the Γ=1 nesting):** all three methods are fed the SAME
per-arm-normalised weights (Σŵ=n). Direct IPW uses the **simple IPW value** objective
`max_π (1/n)Σ_i π_{T_i}(X_i) Y_i ŵ_i` — NOT the self-normalised Hájek ratio. That makes Direct IPW
**exactly RW (and RW−W) at Γ=1**, so at Γ=1 all three coincide and RW/RW−W improve as Γ grows (the
expected nesting). `solve_direct_ipw_multiarm` is the plain value LP; `tests/test_multiarm.py::
test_direct_ipw_nests_rw_at_gamma1` asserts it. (RW stays the paper's calibrated-value formulation;
RW's "normalisation" is the Σ_{I_t}w=n calibration, i.e. divide by n, not the policy-weighted Σπŵ.)

**Replot, never re-solve:** figures/CSVs are regenerated from the saved NPZ in ~3s via
`experiments/discrete/replot.py` (recomputes only the trivial Direct IPW LP when its definition
changed; RW/RW−W pulled straight from saved). The figure is 2×2: (a) train outcome, (b) test outcome,
(c) worst-case objective, (d) learned policy (chosen arm vs X). Use replot for any plot/metric tweak.

**New infrastructure (clean, legacy binary pipeline untouched):**
- `rw_implementation/srpo/multiarm.py` — K-arm solvers `solve_rw_multiarm`,
  `solve_rw_no_wasserstein_multiarm`, `solve_direct_ipw_multiarm` + `inverse_weights_multiarm`,
  `propensity_matrix`, `tight_epsilon_multiarm`. `MultiArmResult.pi` is (K,n). K=2 reproduces the
  binary solvers (tested in `tests/test_multiarm.py`).
- `rw_implementation/experiments/discrete/{dgps,metrics,run,search,run_all}.py` — DGPs (exp_a 3-arm,
  exp_b/exp_c 4-arm, coeffs in `dgps.DEFAULTS`, operating γ_true in `OPERATING_GAMMA_TRUE`),
  shaded-band plots (fill_between ±1 SE, not error bars), save-all CSV/NPZ/PNG in `discrete/out/<dgp>/`.
  `run.py`/`run_all.py` take `--cap-mode {none,data}`; cap run writes `discrete/out/<dgp>_cap/`.
  `run.cap_from_train` builds the per-arm caps; `search.evaluate(_bestG)` take `cap_mode=`.
- `experiments/discrete/build_html.py` — rebuilds the whole 6-sub-sub-tab Discrete-X HTML block from
  saved CSV/NPZ (no re-solve); headline anchored at the matched Γ. Run after any plot/data refresh.
- DGP specs saved to `DGP/DGP - discrete exp_{a,b,c}.rtf` + `DGP/discrete_dgps.py`.
**"★ Compare all" overview tab added (2026-06-04):** the FIRST/active variant tab in each experiment —
overlays ALL six methods (RW, RW−noW, Direct IPW, Kallus-odds, Kallus-W, Oracle) on shared realised
train+test plots vs Γ, one combined figure per setting (unconstrained γ=3, capacity γ=3, capacity γ=5;
N=1000 omitted — no Kallus there). Pure replot from saved CSVs (RW family from `<dgp>_<sfx>.csv`,
Kallus from `kallus_<sfx>.csv`), NO solving. Generator `experiments/discrete/compare_plot.py`
(`out/<setting>/compare_<sfx>.png` + `.npz`); HTML `build_html._compare_panel`/`_compare_read`. Reading
the figure: value-maximisers cluster/coincide at Γ=1, RW rises toward Oracle, Kallus regret-minimisers
fall toward control as Γ grows.

**Kallus (regret) view added (2026-06-04, user request):** a "Kallus (regret)" variant tab in each
experiment — the Kallus & Zhou worst-case-**regret** minimiser (vs all-control baseline π₀), the
regret counterpart of RW's worst-case-value, with BOTH uncertainty sets: **Kallus-odds** (box) and
**Kallus-W** (box+Wasserstein). Compared ONLY with Direct IPW + Oracle, on all settings except N=1000
(unconstrained γ=3, capacity γ=3, capacity γ=5 → 3 stacked figures per experiment). New free-π
discrete solvers `srpo.multiarm.solve_kallus_multiarm` / `solve_kallus_w_multiarm` (regret-min LPs
mirroring the RW solvers; MIN over the box dual). KEY derivation gotcha: the Kallus-W LP dualises the
inner **max** (regret) — opposite sign to RW's inner-min — so gd term is +(1/n)Σgd, metric is
`βD+gd+θ≥0`, factual row is an **equality** `p−q−(1/n)θ+(1/n)πY=(1/n)1[k=0]Y` (got UNBOUNDED first
with RW's signs). Properties (tested, 41 pass): **Kallus-odds≡Direct IPW at Γ=1** (box collapses ⇒
regret-min = value-max), Kallus-W≡odds at Γ=1, **do-no-harm obj≤0** at all Γ (π₀ feasible), regret
monotone→0 as Γ grows. Behaviour: at Γ=1 = Direct IPW; as Γ grows Kallus reverts to control
(realised ~0.51) — conservative, falls BELOW Direct IPW (flat ~0.65) and Oracle (~0.72). Runner
`experiments/discrete/kallus_run.py` (parallel over reps; RECOMPUTES Direct IPW+Oracle on the saved
draws so Γ=1 coincidence is exact — saved RW-run DI vertex differs under LP degeneracy); 45 min for
9 settings. Artifacts `out/<setting>/kallus_{gt3_00,gt5_00}.csv|npz` + `kallus_cmp_*.png`. HTML:
`build_html._kallus_panel`; colours Kallus-odds `#1f77b4`, Kallus-W `#17becf`.

- HTML: "★ Discrete-X study" sub-tab in `code/README.html` Experiments has **3 sub-sub-tabs**
  (exp_a/b/c), and **each contains a 4th-level "variant" switcher** with 3 buttons —
  {Unconstrained, Capacity-constrained, Capacity·stronger γ=5} (restructured 2026-06-04 from the
  earlier 9 flat sub-sub-tabs, at user request: "one tab per experiment, three sub-tabs each").
  The 4th level uses `nav.varianttabs` + `.variant-panel` CSS and a scoped JS handler
  (`nav.varianttabs > button`, toggles within the owning `.subsubtab`). Oracle box documents the
  constrained ceiling. `build_html._experiment_subsub` builds the per-experiment tab + variant nav.
  Each capacity view also has a **capacity-usage figure** (`run.plot_capacity_usage`, shown via
  `build_html._capacity_usage_card`): result-style **line plot vs Γ**, a **2×K grid** (rows=train/test,
  cols=treatments). Each panel: realised allocation (1/n)Σπ_k for RW / RW−W / Direct IPW vs Γ with shaded
  ±1 SE bands, a black dashed horizontal line at cap_k, and a dotted vertical matched-Γ marker (Direct IPW
  flat, Γ-independent; arm 0 = control, uncapped). A curve above the dashed line = violation at that Γ.
  Saved `out/<dgp>_cap/<dgp>_cap_usage_gt{X}_00.png` (+ `.npz`). Evolution at user request: green/red
  TABLE → one-Γ bar plot → **Γ-sweep line plot** ("show across every big Γ to see the violation").
  Train always ≤ cap (LP-enforced); test mostly ≤ cap with small overflows on the binding arm at some Γ.

**exp_c has a 4th variant: cap γ=5 at N_train=1000** (added 2026-06-04, user request "see what happens at
N=1000"; the other experiments keep 3 variants). Finding: more data lifts ALL methods toward the oracle
(RW 0.681→0.688, noW 0.659→0.672, DI 0.653→0.669 at Γ=5) and SHRINKS RW's edge (RW−noW +0.022→+0.016,
RW−DI +0.027→+0.019, SE 0.004→0.003) — confirms RW's advantage is a robustness-to-estimation-error effect
that more data erodes; RW still wins. N=1000 is ~12× heavier per RW solve (n² Wasserstein transport):
330s/rep-pair, so it runs via `experiments/discrete/run_parallel_reps.py` (fans reps over processes;
3 workers → 88 min for 10 reps). Artifacts in `out/exp_c_cap_n1000/`. build_html: `_variants_for(name)`
returns the per-experiment variant list (dicts with vk/cap/gt/anchor/sub/ntrain/label); exp_c gets the
extra `hi-n1000` entry → the 4th `.varianttabs` button.
