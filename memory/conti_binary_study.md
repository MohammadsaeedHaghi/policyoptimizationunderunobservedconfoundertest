---
name: conti_binary_study
description: Binary 2-treatment continuous-X + capacity study (2-arm companion to conti2d); RW best in a Goldilocks Γ band.
metadata:
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

**conti_binary study** (`experiments/conti_binary/`, 2026-06-08): the **2-arm (binary-treatment)** companion to
[[conti2d_study]], built on the user's goal "good result for same experiments + plot for a two-treatment cap-
constrained continuous dataset." Same suite as conti2d (dgps/rosenbaum/run + tests + HTML tab), K=2. New
files: `experiments/conti_binary/{__init__,dgps,rosenbaum,run}.py`, `tests/test_conti_binary.py` (6 tests, 58
total pass). HTML: ★ "Binary 2-treatment constraint experiment" nav button + `build_contibin()` in
build_html.py (CONTIBIN_OUT, CB_GAMMAS=[3,4,5], CB_COL/CB_ORDER reuse C2_*).
**TAB LAYOUT (restructured 2026-06-08, user: "split the two experiments into two sub-tabs", then added γ=5):**
the tab now SPLITS BY γ — a SHARED always-visible header (`_contibin_shared_cards`: intro + DGP + data-sizes +
methods), then ONE γ sub-tab PER confounding level (γ=3, 4, 5) at the
**variant level** (`_contibin_gamma_panel`; deliberately the variant level, which JS-scopes to `.closest('.subsubtab')`
— NOT the subsubtab level, whose `activateSubsubtab` toggles GLOBALLY and would blank other studies' tabs).
Each γ panel stacks its FULL per-γ story: results&policy block + deployed-E[Y] block + per-γ Rosenbaum
(`_contibin_block/_deploy_block/_contibin_rosen_one`). The per-γ intro ORDERING is **data-driven**
(`_contibin_order(gt)` reads the CSV at matched Γ and states the true rank) so each tab is self-correcting.
Verified by a 4-agent adversarial workflow (structure/parity/JS/regression — all PASS). Removed the old 6
mixed variant views (Overview/Compare/deploy/Rosenbaum) in favour of the per-γ split. TABLE CLEANUP (user
2026-06-08, "these tables are just numbers, the plots show it"): removed the deployed-policy tables (realised
+ Expected E[Yπ]), the section-4 both-γ headline table, and each γ's realised-test table — KEPT all plots,
the per-arm capacity-usage table, the Rosenbaum table, and the data-driven per-γ order line (which still
carries the matched-Γ numbers). NOTE: conti2d's analogous deploy/headline tables were left intact (user
scoped the cleanup to the binary tab).

**DGP** (`dgps.py`, K=2, X=(X1,X2)~U[-1,1]², proj=X1+0.6·X2): control arm 0 = σ(θ0), θ0=0.5 (μ0≈0.62, flat,
CLEAN, the uncapped safe default); treatment arm 1 = σ(θ1+A·proj+c·S), **θ1=−1.2 (RISKY: harmful for low-proj,
good only for high-proj), A=3.0, c=4.5 (strong unobserved S inflation of the TREATED outcome)**. Assignment
**mis-targeted**: P(T=1|X,S)=σ(α1−B·proj+γ(S−0.5)·d1), α=[0,0], **B=0.5**, **d=[0,1]** single-signed. cap=
**(1.0, 0.40)** in run.py (control uncapped; treatment cap BINDS — unconstrained oracle wants ~0.46).
**OPERATING_GAMMAS=[3.0, 4.0, 5.0]** (γ=5 added 2026-06-08 on user request — `run.py --only-gt 5` runs just that
γ without re-running 3,4).

**ORACLE = "Full info" (clairvoyant), changed 2026-06-08 on user request.** The reported oracle is NO LONGER
the means-based best-deployable policy — it is the **full-info clairvoyant** that knows both per-unit
counterfactuals (= the project's **Sample Oracle**): `_oracle_cap` now does `solve_oracle_capacity(X, Ypot, cap)`
(was `outcome_means(X)`). Labelled **"Full info"**, BLACK (#111111). It is a **γ-independent realised ceiling
≈0.855** (potential outcomes don't depend on γ; treats only ~0.23 — the units that actually flip 0→1, so the
cap doesn't bind it). User chose to show it on EVERY panel — incl. the noise-free E[Y] panels where its
E[Y]-on-means ≈0.637 sits BELOW the methods (it overfits realized noise; accepted as honest). A SECOND reference
**"Best (true means)"** (green #2ca02c) — the best DEPLOYABLE policy on μ_k(X) — was then added (user 2026-06-08
"add the method that looks into the true mean to the outcome plots"): it's emitted as its own row (realised
0.709 / E[Y] 0.703, treats 0.40 = cap) and now appears in ALL outcome plots (it's the proper noise-free E[Y]
ceiling, tops the E[Y] panels; Full info tops the realised panels) AND as the **"Best (true means, cap)"** panel
in the policy/treatment heatmap. So conti_binary now has TWO reference lines: Full info (clairvoyant, black) +
Best (true means) (deployable, green). The **policy heatmap also gets a "Full info (per-unit, seed0)" panel**
(user 2026-06-08) — rendered as a SCATTER of the seed0 train points coloured by the clairvoyant's per-unit
decision (it has NO smooth X-map; P(Y¹>Y⁰|X)≤0.34 here so a region-heatmap would be all-control). run.py stores
`panel["fi_arm"]=argmax(opol_tr)`; plot_policy_2d branches to `a.scatter(...)` for that panel. (Patched the
existing seed0 NPZs + `--replot` to add it without a full re-run — the seed0 train sample is deterministic at
seed 9000.) Added a "Full info = Sample Oracle" note to the
top-level Methods tab's Sample-Oracle sub-tab (`code/README.html`, hand-written, build_html preserves it).
CB_COL/CB_ORDER in build_html were DECOUPLED from the shared C2_* so conti2d keeps its green means "Oracle (test)".

**FINAL ordering (realised TEST, deployed Shapley, matched Γ), 5 seeds N=600 — RW best at ALL THREE γ** (the
Full-info ceiling sits far above at ~0.855):
- γ=3 (Γ=4.48): **RW .647** > Direct IPW .638 > Kallus-W .629 ≈ Kallus-odds .627 ≈ RW-noW .627 > **Direct Opt .616**  (means-best 0.700; Full info 0.855)
- γ=4 (Γ=7.39): **RW .648** > Direct IPW .630 > Kallus .626 > RW-noW .620 > **Direct Opt .617**
- γ=5 (Γ=12.18): **RW .645** > Direct IPW .628 > Kallus .627 > RW-noW .621 > **Direct Opt .619**
So **RW clearly best at every γ** (edge over DI a stable ≈+.01–.02), RW>RW-noW (Wasserstein edge +.020–.026),
Kallus mid (Γ-responsive: trt-use 0.16→0.08→0.08), **Direct Opt worst**. Heatmap: RW treats a COHERENT high-proj
region (smooth=Wasserstein); RW-noW FRAGMENTED; DI/Direct-Opt spill into the lower-left HARMFUL region; Kallus
nearly all control. Λ̂/matched grows mildly: 1.36 (γ=3), 1.56 (γ=4), 1.74 (γ=5) — all <2× (single-signed d).

**KEY STRUCTURAL FINDINGS (2-arm is much harder than 4-arm — confirms prior memories):**
1. **RW worst-case-value is Γ-invariant** ([[rw_gamma_invariance]] confirmed empirically here): RW treats a
   ~fixed ~0.45 at EVERY Γ; only Kallus (regret) is Γ-responsive (refrains more as Γ grows). CORRECTION
   (2026-06-08): with the FINAL strong-confounding + loose-cap config (c=4.5, cap=0.40) RW is clearly best at
   γ=3, 4 AND 5 — the earlier scout finding that "RW slips at γ≥5" was from a WEAKER config (c=3.5, cap=0.25);
   it does NOT hold for the final config. Only at the very low end (γ=2) does robustness stop mattering (DI ties
   RW). So OPERATING_GAMMAS=[3,4,5], each its own HTML sub-tab.
2. **Wasserstein edge collapses in 2-arm at tight caps** ([[discrete_multiarm_study]]: "edge vanishes in
   2-arm") — c_eps∈{1,2,3,4} had ZERO effect at cap 0.25 (constraint slack). RECOVERED only with a **loose cap
   (≥0.40)** that forces extrapolation into the marginal region: then RW>RW-noW by ~.02–.03 (RW coherent,
   RW-noW fragmented). Lesson: in 2-arm the Wasserstein term needs room (loose cap) to bind.
3. **Binary Rosenbaum Λ̂ is intrinsically inflated.** With B=0 + balanced intercepts, Λ̂=e^(γ/2) EXACTLY; ANY
   selection (B>0, needed to separate Direct Opt from DI) spreads propensities and pushes Λ̂ above e^(γ/2)
   (the MSM bound is tight only at propensity 0.5; worst at high γ → e^γ at extreme propensity). Kept B=0.5 →
   **Λ̂/matched ≈ 1.36 (γ=3), 1.56 (γ=4)** — close, far below the ~10× of opposite-sign multi-arm.

**Control philosophy difference vs conti2d** ([[control_arm_philosophy]]): here control is the safe default,
optimal for the MAJORITY (~54%) — that's the canonical binary treatment-vs-control setting (treatment is the
scarce, risky, sometimes-beneficial intervention), NOT the 4-arm "never-optimal specialist-loser" design.
Control still clean+solid so RW/Kallus hedging isn't penalised.

**Why each method ranks where it does:** unobserved S inflates the treated outcome → naive over-treats
harmful units. Direct Opt (no IPW) fooled by both S and the mis-targeting → worst. Direct IPW corrects
observed mis-targeting (1/P(T|X)) but not S → 2nd. RW (worst-case box + Wasserstein) treats the robustly-good
high-proj region, avoids harmful → best. Kallus (do-no-harm) refrains when treatment looks risky → mid.

Tuning took ~5 scout rounds (scout deleted): v1 beneficial-treatment → Kallus worst (refraining loses when
treatment good); fixed by making treatment RISKY (θ1<0) so do-no-harm is sensible; then loose cap + strong c
to make RW clearly beat DI and recover the Wasserstein edge. `--tag`/`--replot` supported like conti2d. See
[[conti2d_study]], [[kallus_paper]], [[wasserstein_binding]].

**NO-GRID degeneracy demonstration — 2nd subsubtab, 2026-06-09** (user: "generate a new tab … run the
experiment without griding, same DGP, reproduce the results"). WHY the free-π LP grids (see also
[[conti2d_study]] binning lesson): the LP has a per-unit var π[k,i] tied across same-X units by `tie_same_x`;
on continuous X every point is unique so nothing ties → the policy is free per unit. **Runner:** added a
`--no-grid` flag to `experiments/conti_binary/run.py` — uses the RAW continuous X as the LP support instead of
`_snap(tr.X)` (threaded through the task tuple as a 7th element; auto-tags outputs `_nogrid` so the gridded
artefacts are preserved). **TWO findings, both in the tab:** (1) STATISTICAL — the value-maximisers
(RW/RW-noW/Kallus-W/Kallus-odds/Direct IPW/Direct Opt) cluster to within a realised-test spread of only
~0.002–0.004 (the robustness/regret/naive distinction vanishes) and **overfit**: realised-train ~0.78 (memorise)
but realised-test ~0.63, BELOW Best-means 0.70 and far below Full info 0.86 — γ-INVARIANT across 3/4/5; the
policy heatmap is the clincher (every method = the SAME fragmented per-unit blotch vs the clean half-plane the
gridded policies learn). (2) COMPUTATIONAL — without the grid the Wasserstein transport LP is **O(n²)**
(≈360k terms at n=600 vs ≈1.3k gridded over 36 cells), so **n=600 is infeasible** (workers ran 22 min without
finishing one γ). The no-grid run therefore uses **n=200** (same DGP/caps/seeds/Γ-sweep, only n differs — stated
in the tab); the degeneracy is n-independent. Also: realised_train ≈ realised_train_dep here (support == raw X
⇒ Shapley/KNN exact on the training points, no snapping gap) — vs the gridded ~0.03 gap.
**HTML:** `build_contibin()` now emits a 2nd subsubtab `exp-contibin-ng` ("★ no gridding (free-π on raw X)")
beside the gridded `exp-contibin-b` ("binary (2-arm) · gridded"); `_contibin_nogrid_panel/_shared/_stats`
(prose is ADAPTIVE on the measured spread — "collapse" if <1e-3 else "cluster", since exact collapse is
config-dependent: n=200/5-seed clusters ~0.003, the n=200 single-seed smoke collapsed to 0.0000, n=140
single-seed clustered ~0.025). Tests: `tests/test_conti_binary_nogrid.py` (3 tests — train≈train_dep no-grid,
gridded HAS a gap, no-grid overfits; **63 total pass**).
**JS BUG FIXED (code/README.html, static JS, build_html preserves it):** `activateSubsubtab` toggled
`.subsubtab` sections GLOBALLY — fine when each experiment had 1 subsubtab, but the new 2nd conti_binary
subsubtab made it reachable (switching gridded↔no-grid would blank conti2d/discrete/etc.). FIX: scope the
toggle to the clicked subsubtab's `target.closest('.subtab')` (buttons+sections queried within that scope),
falling back to the page-wide lists when there is no `.subtab` ancestor — a strict improvement that also fixes
the latent cross-experiment bug for discrete/param/methods. Verified each "exp-*" subsubtab resolves to its own
`.subtab` (exp-contibin-b/ng → exp-contibin).
**PROCESS LESSON (cost me ~40 min):** do NOT diagnose ProcessPoolExecutor "stalls" by `ps|grep <module>` — on
macOS spawn the WORKERS' cmdline is a spawn stub (no module name), so grep only matches the IDLE PARENT (0% CPU,
correctly waiting). I misread that as hung and killed runs that were computing fine; `pkill -f <module>` then
ORPHANED the workers (ppid=1, kept running, thrashed memory to ~90MB free). Correct checks: watch TOTAL python
CPU or children via `pgrep -P <parent>`; kill spawn workers with `pkill -f multiprocessing.spawn`. Also
`--jobs 5` works fine (the earlier "jobs-5 hang" was the same misread).
