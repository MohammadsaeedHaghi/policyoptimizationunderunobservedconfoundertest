---
name: owgap-dgp
description: "exp_owgap DGP -- discrete-X K=2 where BOTH O-W methods beat ALL others by a GOOD GAP in both regimes at small Γ; the decoupling insight V(π)=mean[π·CATE]; ε is the collapse knob; value-O-W is Γ-stable (only Hajek-O-X collapses)."
metadata:
  node_type: memory
  type: project
  originSessionId: 16b87e39-c6c8-4604-955a-f4b218ddc989
---

**Goal (2026-06-24):** a discrete-X, binary-T DGP where BOTH IPW-O-W and DoublyRobust-O-W beat EVERY other
method (incl plain DoublyRobust-X-X) by a GOOD GAP, in BOTH capped & uncapped, peaking at small Γ (2-3), with
the large-Γ collapse onset ~Γ=4 controlled by 3 ε values. Successor to [[owwin3-owbest-dgp]] (which had only a
modest margin). Built as the standalone `assets/exp_owgap/` (dgp.py, run_owgap.sh, build_owgap.py, owgap.html).

**Winning DGP "owgap"** (search `assets/exp_dgp_search/{search_owgap.py,sweep_eps.py}`): discrete X 7-level grid,
S∈{-1,+1} UNOBSERVED:
- P(S=+1|X)=σ(**10**X) (strong S-X coupling -> the Wasserstein lever)
- e(X,S)=σ(**0.8**S − 2X), clip[.02,.98] (MODERATE selection -> small matched Γ)
- μ0=**8**S, μ1=**9**S+**1.5**X (d0=8,d1=9: HUGE common confounding, δ=d1−d0=**1** small non-cancelling, b1=1.5)
- noise 0.6; CAP=(1,0.5); methods FIT logistic P(T|X). CATE=δ·E[S|X]+1.5X = treat iff X>0. oracle≈0.847.

**THE KEY DECOUPLING INSIGHT (cracked the "big gap" problem):** because X is symmetric and S~σ(αX) is odd,
mean_X[eY0]=0, so the realised value is EXACTLY **V(π)=mean_X[π(X)·CATE(X)]** — it depends ONLY on CATE and the
policy, NOT on d0. Therefore d0 (common confounding magnitude) controls ONLY the outcome-model BIAS = how wrong
naive is (=> the gap), while δ and b control CATE (=> the value scale and the collapse speed). They DECOUPLE:
crank d0 LARGE (d0=8 -> strong bias -> naive over-treats the harmful X<0 region -> big gap) while keeping δ,b
SMALL (CATE max ~2.5, moderate -> collapsible / erasable by a Γ≈4 box). owwin3 missed this (it raised d AND δ
together). Large-d + small-differential = big gap; verified the gap is ~2-3x owwin3's.

**THE HONEST COLLAPSE FINDING (matches [[discrete-ow-wins-dgp]]):** the worst-case-VALUE robust methods
(IPW-O-X, DR-O-X, and BOTH O-W) keep a BOUNDED worst case via the self-normalised odds-box, so they are largely
**Γ-stable** — they do NOT cleanly collapse to never-treat as Γ grows. The clean collapse-to-never-treat is the
REGRET method **Hajek-O-X** (drops to ~0 by Γ≈2). For the O-W family, **ε is the controllable conservatism knob**:
larger c_ε makes IPW-O-W soften earlier after its Γ=2-3 peak and pushes DR-O-W's level down (DR-O-W is much more
ε-sensitive than IPW-O-W). So requirement "collapse onset at Γ≈4" is honestly framed as: Hajek-O-X collapses
cleanly; value-O-W softens post-peak under ε; X-X is Γ-free (flat, cannot collapse). Headline c_ε=**1.0** (both
O-W clearly beat all); 3 ε shown = {1.0, 1.5, 2.0}.

**Full study:** N=600 (the O(n^2) Wasserstein LP makes N=700×5seed×9Γ×3ε impractical; N=600≈700, result robust),
5 seeds, Γ=[1,1.5,2,2.5,3,4,5,6,8], both regimes, ONE run per c_ε via `assets/run_experiment_parallel.py`
(--workers 8; the Gurobi WLS licence caps concurrency near 8). Results JSON: owgap_results_ce{1.0,1.5,2.0}.json.

**RESULTS (N=600 5-seed, headline c_ε=1.0), oracle 0.847:**
- UNCAPPED, Γ=2-3: IPW-O-W 0.77, DR-O-W 0.75-0.82 vs best competitor DR-X-X 0.642 (IPW-O-X peaks 0.50, DR-O-X
  0.57-0.70, Direct 0.12, IPW-X-X 0.39). BOTH O-W beat ALL by +0.11..+0.18. Hajek-O-X collapses 0.39->0 by Γ=2.5. ✓
- CAPPED (treat<=50%), Γ=2-3: IPW-O-W 0.72-0.77 beats best competitor DR-X-X 0.709; but DR-O-W ≈ 0.705-0.712
  **TIES DR-X-X** at small Γ (pulls ahead only at Γ>=5: 0.755-0.79). HONEST CAVEAT for (a)/(b).
- WHY the capped DR-O-W gap is thin: under cap<=50% the cap RESCUES naive DR-X-X (it can only treat its top-50%
  by CATE_hat; with MONOTONE confounding the level RANKING is preserved, so capped DR-X-X stays near the right 50%
  -> hard to beat). Beating naive UNDER A CAP needs NON-MONOTONE confounding (the [[nonmono_experiment]] regime).
  Uncapped, naive over-treats freely -> big clean gap.

**Properties (a)-(e) status:** (a) gap: ✓ uncapped both O-W (+0.11..0.18); capped IPW-O-W ✓, DR-O-W TIES DR-X-X
at small Γ (honest caveat). (b) both regimes: ✓ uncapped; capped IPW-O-W ✓, DR-O-W weak at small Γ. (c) peak at
Γ=2-3: ✓. (d) collapse onset ~Γ=4: at c_ε=1 the value-O-W is roughly FLAT past its peak (Γ-stable, bounded
worst case); at **c_ε=2 IPW-O-W shows the clean peak@Γ=2.5-3 then decline 0.69->0.65(Γ4)->0.62(Γ8)** = onset@~4;
Hajek-O-X collapses to never-treat by Γ=2.5 in every panel. So value-O-W collapse is ε-GATED, not purely Γ-driven.
(e) 3 ε control: ✓ c_ε=1 flat-high (both O-W win) -> c_ε=2 peak+collapse-onset; larger c_ε = earlier/steeper, DR-O-W
much more ε-sensitive than IPW-O-W. **Gap-vs-collapse trade-off is real:** the big-CATE variant (d0=8,d1=9.5,b=2,
oracle 1.199) gives a CLEAN capped gap (+0.32) BUT its O-W keeps RISING through Γ=4 (no peak@2-3, no collapse).
owgap (this, oracle 0.847) prioritises (c)+(d)+(e); the big-CATE variant prioritises (a)+(b). Deliverable uses owgap.

Story (clinical): aggressive therapy under hidden vitality S; S tracks fitness X and dominates outcomes in both
arms (so the outcome model is badly biased by S-selection); the therapy's true effect is subtle and partly
harmful for the unfit (helps X>0, harms X<0); clinicians escalate the vital -> naive over-treats; covariate
(Wasserstein) balance reaches the hidden S -> O-W recovers "treat only the fit." See [[owwin3-owbest-dgp]],
[[owwin2-nocap-dgp]], [[discrete-ow-wins-dgp]], [[no_true_propensities]], [[save-all-artifacts]],
[[method_naming_plot_style]], [[html-raw-lessthan-bug]].
