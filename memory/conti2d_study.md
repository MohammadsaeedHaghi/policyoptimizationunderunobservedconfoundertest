---
name: conti2d_study
description: 2-feature continuous + capacity study (radial DGP) with empirical Rosenbaum verification.
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

**conti2d study** (`experiments/conti2d/`, 2026-06-07): a TWO-feature continuous-X capacity experiment,
built creatively around making the **Rosenbaum sensitivity model** correct (user: "the only important
thing is rosenbaum"). New package: `dgps.py`, `rosenbaum.py`, `run.py`, `__init__.py`; test
`tests/test_conti2d.py` (4 tests, 50 total pass). 4th Experiments sub-tab `exp-conti2d` ("2-feature conti").

**Radial DGP** (4-arm, X=(X1,X2)~Uniform[-1,1]²): a **control** arm wins the centre; three specialist
arms win outer angular sectors (φ=0, 2π/3, 4π/3). Outcomes σ(θ_base + A·r·cos(φ−φ_k) + c_k·S); assignment
softmax with the S-channel `γ(S−0.5)·d_k`. **A=6.0** (was 2.0 — see Wasserstein-edge note below): outcomes
are SHARPLY X-heterogeneous, so control is best in only ~0.7% of the space (wins only the immediate centre,
r<0.15); specialists ~33% each. **Capacity binds all 3 specialists.**

**WASSERSTEIN EDGE — RW > RW-noW** (user asked 2026-06-07 "push until RW beats RW-noW, change DGP as
needed"). Scouted the levers (scratch scout/verify scripts, since deleted): the edge GROWS with #arms AND
with the OUTCOME-X-slope A (sharper heterogeneity ⇒ covariate balancing matters more); strong selection B
HURTS it. Chose the minimal change: **A: 2→6** (keep K=4, B=0.8, cap (1,.12,.12,.2), radial structure).
Real-pipeline result (realised test, matched Γ): **RW > RW-noW > Direct IPW** — γ=3: RW .692 / RW-noW .678
/ DI .665; γ=5: RW .692 / RW-noW .677 / DI .656; gap RW−RW-noW ≈ +0.014/+0.016 (clear, >2σ over 5 seeds),
Oracle .786.

**RW > RW-noW on BOTH train & test** (user asked 2026-06-07). On the **realised in-sample train** metric
RW-noW slightly wins/ties — it OVERFITS the noisy training Ypot (overfitting raises realised-on-fitted-data
by construction); that's the regularization signature and the realised tabs document it (KEEP them). Swept
n_train/grid/B/A: realised-train gap stays ~0; even noise-free train_μ of the LP policy is ~0 (Wasserstein
regularization is ~free in-sample). The fix is to evaluate the **DEPLOYED (Shapley-extended) policy** — the one you'd actually use — on each
distribution, instead of the raw in-sample LP fit. On the deployed policy RW > RW-noW on BOTH train and
test, on the **realised** outcome (binary Y: train_dep +.012/+.014, test +.014/+.016) AND, more cleanly, on
the noise-free **expected** E[Y] (μ: train +.012/+.013, test +.012/+.015). User explicitly wanted realised-
train RW>RW-noW (not just E[Y]) — the deployed-realised-train metric delivers it. Added runner cols
`realized_train_dep` (deployed policy realised on train), `exp_train`, `exp_test`; `plot_deploy` is a 2×2
(realised train/test + E[Y] train/test); NEW conti2d tab **"RW wins train+test (E[Y])"** with realised &
expected tables (existing realised-LP tabs untouched). Key lesson: RW's advantage is GENERALIZATION
(deployed/extended policy), NOT the raw in-sample LP fit — which RW-noW wins purely by overfitting; compare
the deployed policy, not realised-on-the-fitted-LP. TRADE-OFF: A=6 shrank the control "core" (control best in only 0.7% area) — flagged to user;
can dial A back (≈4–5) or raise theta0 / use K=8 if a larger core is wanted.

**KEY Rosenbaum finding** (the centrepiece — first place in the repo that VERIFIES matched Γ=e^(γ/2)):
`rosenbaum.py` measures the realised marginal-sensitivity Λ̂ = max_i OR(e_cond,e_marg) (factual
P(T|X,S) vs S-marginal P(T|X)). With **single-signed** `d=[0,.34,.67,1]` (max|d|=1) the matched Γ holds:
**Λ̂=4.82 vs e^(γ/2)=4.48 (γ=3, ratio 1.08); 15.4 vs 12.2 (γ=5, ratio 1.26)**. CRITICAL: the exp_c-style
**opposite-sign** `d=[-1,-.3,.3,1]` instead blows Λ̂ up to ~10× e^(γ/2) (cross-arm softmax coupling) — so
the matched-Γ=e^(γ/2) heuristic the whole project relies on is only clean for single-signed confounding /
binary treatment. Use single-signed d when matched Γ must be exact.

**Binning lesson:** 2-D continuous X rounded to 21² cells = all singletons → degenerate (all methods
coincide, see [[conti_x_study]]). Fix: snap train X to a **6×6 grid** (36 cells, ~17 units/cell) for the
LP support; extend to continuous test via Shapley/KNN. POLICY_GRID=6, `_snap()` in run.py.

**Methods**: RW + RW−Wasserstein + Direct IPW + **Kallus-W & Kallus-odds (regret-minimisers)** + Oracle,
all Shapley+KNN extended. (Kallus added 2026-06-07 — user: "combine with kallus regret minimization";
`solve_kallus_w_multiarm`/`solve_kallus_multiarm` added to run.py's emit loop, same grid-snap/CAP/rounding=6
as RW; STYLE Kallus-W #17becf, Kallus-odds #1f77b4; in ORDER/C2_ORDER but NOT BASES so they stay out of the
worst-case-obj panel & policy heatmap.) Kallus-W≈Kallus-odds here.

**Direct Optimization baseline** (user asked 2026-06-08): naive max Σ_i π_{T_i}(X_i)·Y_i (raw factual
reward, NO IPW, NO confounding adjustment) s.t. caps. = `solve_direct_ipw_multiarm` with UNIT weights;
added `srpo.multiarm.solve_direct_opt_multiarm` (thin wrapper), STYLE/ORDER/C2_* entry colour #ff7f0e
(orange, marker X), solved once/rep like DI. In outcome/deploy/usage plots + tables, NOT obj-panel/heatmap.

**DGP RETUNED so Direct Opt is the WORST** (user: "with unobserved confounder, Direct Opt shouldn't do well;
tune the DGP", 2026-06-08). KEY LESSON from an extensive sweep: stronger c does NOT fool Direct Opt — a
prognostic c boosts all arms (no reordering), aligned c makes inflated arms genuinely good, and the TIGHT
CAPACITY shields Direct Opt by bounding over-prescription of any one arm (even a decoy arm). What works:
a **MIS-TARGETED historical policy** — the assignment X-feature ROTATED by π (`assign_rot=π` in dgps.py;
`_radial_feat(X, rot)`, outcome rot=0 / assignment rot=π) so units were historically assigned to the
WRONG-sector arm (observed confounding, spread across cells ⇒ cap can't shield). PARAMS: A 6→5, B 0.8→1.0
(B=1.5 over-inflated Λ̂ to 2.3×; B=1.0 keeps Λ̂ ratio 1.2/1.4). Direct Opt is clearly WORST (ignores both
confounders); Direct IPW corrects the observed mis-targeting (beats DirectOpt) but not unobserved S. Lesson:
capacity can hide a naive method's bias; observed confounding (mis-targeting), not just stronger unobserved
c, is what exposes it. (The first retune used prognostic c=[2,2,2,2] which made CONTROL the strongest arm —
superseded by the control-philosophy fix below.)

**CONTROL = clean SOLID fallback, never the strongest (user 2026-06-08: "I dont want control to be the
strongest arm — it has no cap constraint").** The user rejected control-strongest, then (after I surfaced
the tensions) chose **option C "Clear RW best only (control is a solid fallback)"** + "Kallus mid-pack".
FINAL PARAMS in `dgps.py`: **theta0=1.10** (control μ≈0.75, a CLEAN solid uncapped fallback, optimal in only
~2.2% of X — tiny centre — but NOT weak), **theta_base=−0.30, A=5.0, c=[0.0,2.0,2.0,2.0]** (control c_0=0:
the fallback is a clean baseline, NOT itself confounded; specialists S-confounded c=2), alpha=[0.55,0,0,0],
B=1.0, assign_rot=π, d=[0,.34,.67,1]. **FINAL ordering (realised TEST, deployed Shapley, matched Γ):**
N=600 γ=3 Oracle .852 > RW .751 > DI .724 > Kallus-W .718 > RW-noW .712 > DirectOpt .689; γ=5 RW .745 >
Kallus-W .734 > DI .731 > RW-noW .706 > DirectOpt .695. **N=1000 (cleaner):** γ=3 Oracle .867 > RW .777 >
Kallus-W .766 > Kallus-odds .758 > DI .738 > RW-noW .726 > DirectOpt .709; γ=5 RW .776 > Kallus-W .768 >
Kallus-odds .764 > DI .748 > RW-noW .713 > DirectOpt .712. So **RW clearly best (+.01–.04), Direct Opt worst,
Kallus mid-pack (2nd–3rd), RW>RW-noW (+.05/.06 Wasserstein edge), control never optimal**. KEY TENSION
(documented): control-weakest CONFLICTS with both RW-best AND Kallus-mid — robust RW/Kallus hedge toward
control under high Γ, so a WEAK control penalises them and lets the committing Direct IPW beat RW. RW-clearly-
best therefore REQUIRES control to be a SOLID (non-weak) arm; the resolution is control clean-but-never-
optimal (solid value, tiny optimal region). See [[control_arm_philosophy]].

**TAB RESTRUCTURED 2026-06-09 (user: "make it clean like discrete-X", chose per-γ self-contained + drop
Overview/Compare):** the conti2d tab now mirrors the discrete-X / binary clean layout — a SHARED always-visible
header (`_conti2d_shared_cards` = intro + DGP + data-sizes + methods; the old `_conti2d_overview` mega-page and
its both-γ headline table were REMOVED), then per-γ self-contained views **★ Capacity γ=3 / ★ Capacity γ=5 /
★ N=1000 / Rosenbaum check**. Each γ view = `_conti2d_block(gt)` (results 1×3 + policy heatmap + capacity-usage)
+ `_conti2d_deploy_block(gt)` (deploy 2×2 E[Y]); the deploy block was trimmed to FIGURE-ONLY (tables dropped,
`_conti2d_deploy_read` removed). The standalone "Compare all" and "deploy" variant views were dropped (folded
into each γ). N=1000 is one starred view showing both γ stacked. No re-run — all figures already existed.

**DGP card → LaTeX, 2026-06-09** (user: "the DGP explanation isn't showing in latex; fix"). `_conti2d_shared_cards`
was a plain f-string with Unicode/HTML math (σ, θ₀, <sup>); rewrote the "1 · Data-generating process" card in
MathJax LaTeX ($$…$$ display eqs for X/r/φ/S, potential outcomes σ(θ_base+A·r·cos(φ−φ_k)+c·S), mis-targeted
assignment, μ_k & Γ=e^{γ/2}, capacity with \underbrace) — same style as [[conti_binary_study]]'s DGP card. REQUIRED
converting the function to a **raw f-string `rf'''`** and **doubling every literal brace** `{{`/`}}` (the
[[keep_html_in_sync]] gotcha). Also fixed a pre-existing bug: `γ_true ∈ {3, 5}` was a single-brace f-expr that
rendered as the tuple "(3, 5)" — doubled to `{{3, 5}}`. `{600 // 36}` stays a real f-expr (→16 units/cell).

**N=1000 ALIGNED to the binary design — Full info + Best (true means), STARRED 2026-06-09** (user: "align
conti2d to Full info + Best means (just N=1000); star only the N=1000 tab"). Scope decision (AskUserQuestion):
**ONLY the N=1000 view** — γ=3/γ=5 keep their existing **"Oracle (test)"** (means-based) ceiling unchanged.
KEY INSIGHT: conti2d's "Oracle (test)" IS the means-based capacity oracle (== "Best (true means)" by
construction), so aligning = (1) RENAME "Oracle (test)"→"Best (true means)" (identical values) + (2) ADD a
"Full info" CLAIRVOYANT ceiling (per-unit, knows BOTH counterfactuals; `solve_oracle_capacity(X, Ypot, cap)`).
CHEAP patch (no heavy RW/Kallus re-solve, per the no-rerun rule): a one-off script regenerated the
deterministic N=1000 samples per rep (seed0=9000+r, n_train=1000/n_test=2000), renamed the Oracle rows in
`exp_radial_n1000_{gt3,gt5}_00.csv`, appended Full info rows (one flat row/Γ; Γ-free clairvoyant), and patched
the seed0 panel NPZ with `fi_arm`=argmax clairvoyant train decisions; then `run.py --replot --tag _n1000`.
(Patch script deleted after — it is NOT idempotent: re-running would duplicate the Full info rows.) `run.py`
edits: STYLE/ORDER gained "Full info"(#111,--,*) + "Best (true means)"(#2ca02c,-.,*) KEEPING "Oracle (test)"
(plot fns iterate ORDER and draw whichever methods are present per CSV → γ=3/5 unaffected); `plot_policy_2d`
renamed the "Oracle (capacity)" panel→**"Best (true means, cap)"** and adds a **Full-info per-unit SCATTER**
panel gated on `panel["fi_arm"]` (clairvoyant has no smooth X→arm map — mirrors conti_binary). build_html
edits: C2_COL/C2_ORDER gained Full info+Best means (keep Oracle); `_conti2d_block` is now **data-driven**
(`is_n1000 = "Full info" in by`) — ceiling usage-row, prose, policy caption + Full-info-scatter note all adapt.
`build_conti2d` variants UNSTARRED g3/g5 ("Capacity γ=3"/"Capacity γ=5"); only "★ N=1000" starred. RESULT
(realised, N=1000): Full info ~0.98 (clairvoyant overfits noise → only ~0.78 on E[Y]), Best (true means) ~0.86
(top on E[Y]), then methods (RW best, per CONTROL section). 60 tests pass, HTML validates (259 div, 1370 $).

**(pre-2026-06-09) Naming** (user 2026-06-08): the conti2d study nav button is
**"★ Continuous 2D constraint experiment"** (star-marked). The policy heatmap
(`plot_policy_2d`, HEAT_BASES) shows ALL methods incl. **Kallus-W & Kallus-odds**.
KALLUS BEHAVIOUR (clarified): Kallus reverts to control as Γ grows — at Γ=1 it uses specialists like DI
(control .56), but at the matched Γ (4.5/12) it is **~70–89% control** (do-no-harm regret-min; usage from the
final CSVs). Its heatmap is mostly grey (control). It's mid-pack (2nd–3rd) in realised test (~.72–.77)
because control is now a CLEAN SOLID fallback (c_0=0, θ0=1.1, μ≈0.75 — never optimal but NOT weak), so
hedging toward it isn't penalised. RW is the best METHOD in the realised-outcome plots across the whole
Γ-sweep (train & test); the green dashed top line is the Oracle ceiling, not a method. 52 tests pass; HTML
text synced to the final config (2026-06-08).

**N=1000 variant** (user asked 2026-06-07): `run.py` now takes `--tag`; run with
`--n-train 1000 --tag _n1000` → separate `exp_radial_n1000_*` files (the N=600 study is preserved). Added
as a 4th conti2d variant tab "N=1000" in build_html (`_conti2d_block(..., tag)`, guarded). ~28 units/cell
vs 17. The RW>RW-noW edge GROWS with N (the in-between A=6 numbers RW−RW-noW ≈ +.03/+.02 were historical;
FINAL A=5 config N=1000 numbers are in the CONTROL section above — RW .777/.776, RW-noW .726/.713, edge
+.05/.06). Both N variants use the SAME final config (A=5). Artefacts: `out/exp_radial_{gt3,gt5}_00.{csv,npz}` +
`_results/_policy2d/_usage_*.png` + `rosenbaum_{gt}.{csv,png}`. `--replot` re-renders from CSV.

**Capacity = FIXED vector `CAP=(1, 0.12, 0.12, 0.2)`** in `run.py` (control uncapped; user set 2026-06-07,
tightened from the earlier (1,.5,.5,.29); replaced the per-seed `cap_from_train`). TIGHT — all 3 specialists
bind hard (unconstrained oracle wants ~0.29/0.31/0.34; total specialist budget only 0.44, ~56% routed to
control). NOTE: test usage exceeds the caps (~0.18/0.19/0.23) because caps are LP-enforced on the train
support only; the Shapley/KNN-extended test policy overflows (expected, flagged in the usage plot).

**Constrained best-policy ORACLE (user-requested fix).** The old "Oracle (test)" used `solve_oracle_capacity`
on the *realised binary Ypot* → a near-trivial NOISE ceiling (~0.97: with 4 arms each ~0.4–0.86, almost
every unit has some arm drawing Y=1, so caps barely move it). FIX: feed `solve_oracle_capacity` the TRUE
outcome MEANS μ_k(X)=E[Yᵏ|X] (new `dgps.outcome_means`, marginal over S) instead of Ypot ⇒ the best
DEPLOYABLE policy under the caps = the fair achievable ceiling (**0.775**). `run._oracle_cap` now does this.
NOTE for tests: the realised-Y clairvoyant max is still the valid UPPER bound for a method's *realised*
outcome (a μ-oracle is not), so test_conti2d's ceiling-assertion keeps using the Ypot clairvoyant.

**Caps ARE enforced — and never violated by the learned policy** (user flagged 2026-06-07). The cap LP
`(1/n)Σ_i π_k(X_i) ≤ cap_k` binds the policy on the binned support; the LP solution respects it exactly
(asserted in test_solvers_caps_and_simplex). Oracle usage = [0.56, 0.12, 0.12, 0.2] (≤cap, solved per-unit).
**BUG FIXED:** `run.emit` was reporting the Shapley/KNN-EXTENDED policy's usage even on TRAIN (which
overflows, since extending a binned policy to raw X re-weights past the caps) — making it LOOK like a train
violation. Now TRAIN reports the actual LP policy (`pi_lp`, ≤cap; identical for both extensions, no
extension on-support); only TEST is extended (and may overflow — an unavoidable deployment artifact, not a
constraint violation). HTML usage table now has 4 rows: cap / Oracle / RW(train·LP, green ≤cap) /
RW(test·extended, red overflow); `plot_usage` includes the Oracle and labels train=LP / test=extended.
`_oracle_cap` returns (value, usage). RW train usage e.g. [0.569, 0.111, 0.12, 0.2] — all ≤ cap.

**Policy heatmap fix** (user flagged 2026-06-07): `plot_policy_2d`'s Oracle panel used to show
`true_best_arm` = the UNCONSTRAINED argmax (specialists fill ~94% of area) — inconsistent with the
capacity. Now it shows TWO reference panels: "Best arm (no cap)" (unconstrained ideal) AND "Oracle
(capacity)" = `solve_oracle_capacity(grid, μ(grid), CAP)` argmax, which respects the caps (control fills
56% of the area, specialists [0.12,0.12,0.2]) — matching the reported oracle usage. Then RW/RW−W/DI panels.

**Headline (matched Γ, cap=(1,.12,.12,.2)):** control 0.626, constrained oracle **0.723** (tighter cap ⇒
lower ceiling than the 0.775 at cap=.5/.5/.29 — the oracle responds correctly to the cap). **RW peaks near
the matched Γ** and beats DI (γ=3: RW .619 > RW−noW .615 > DI flat .597; γ=5: RW .622 / RW−noW .625 / DI
.591). DI over-assigns specialists (confounded signal) → ≈/below control; RW conservative, robustly recovers
to ≈control. Neither reaches the oracle — strong confounding + coarse binning leave real headroom (the loose
0.97 Ypot-max oracle had hidden this). Wasserstein edge marginal. See [[conti_x_study]], [[kallus_paper]].
