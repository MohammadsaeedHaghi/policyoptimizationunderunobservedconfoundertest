---
name: discrete-ow-wins-dgp
description: "The simple discrete-X DGP (v3) where OW family clearly beats OX beats XX, capped — found via /goal search"
metadata: 
  node_type: memory
  type: project
  originSessionId: 58aeb24f-db15-4091-9dfb-a7f743c17425
---

**Goal (2026-06-21):** find a SIMPLE discrete-X DGP where OW family clearly > OX family clearly > XX family, multi-seed, low N. Harness: `assets/exp_dgp_search/{search.py,verify.py}` — capped solvers, EXACT analytic evaluation (X uniform on grid → value = mean over levels of π·eY1+(1-π)·eY0, zero test noise), only training-sample noise averaged over seeds.

**Winning DGP "v3"** (discrete X, 7 levels, S∈{-1,+1}, cap=(1,0.5) treat≤50%):
- X ~ Uniform on linspace(-1,1,**7**)
- P(S=+1|X) = σ(**6**X)   (strong S-X correlation = the OW lever)
- e(X,S) = σ(**2**S − **2**X), clipped [0.02,0.98]   (mis-targeted low-X + S-confounded → ε>0 on logistic fit)
- μ0(X,S)=**2.5**S, μ1(X,S)=2.5S+X  ⇒ CATE=X, oracle treat iff X>0; NOISE=**0.6** Gaussian
- methods FIT logistic P(T|X) (never true e); matched Γ huge (e^{2cs}=e^4≈55) but methods peak/plateau by Γ≈8-12.

**FINAL N=700, 8 seeds, cap 0.5 — BOTH families 8/8 per-seed (goal met):** oracle 0.286. per-seed win-rate DR(O-W>O-X>X-X)=**8/8**, IPW(O-W>O-X>X-X)=**8/8**. DR monotone at every Γ∈{2,3,5,8,12}: Γ=12 → DR-O-W 0.230 > DR-O-X 0.187 > DR-X-X 0.064 (gaps +0.043, +0.123); Γ=8 → 0.224 > 0.169 > 0.064. IPW-O-W byΓ=[.096,.149,.188,.212,.224], IPW-O-X byΓ=[.05,.111,.107,.089,.094] (its lucky peak), IPW-X-X .05. At N=500 the DR family was already 8/8 but IPW only 4/8 (IPW-O-X lucky-peak variance sd .067); N=700 averages it out → 8/8. cap=0.4 shrinks all values, no help — cap 0.5 is best. Run via `assets/exp_dgp_search/verify.py` (logs verify_v3/verify2/verify_n700).

**Built into index.html as the ★ "Sixth experiment · discrete-X (clear OW>OX>XX)" tab** (`assets/exp_owwin/{dgp,run_owwin,build_owwin}.py`, var OWWIN_CHARTS, charts ichart-ow-*): data-explanation plots + value-vs-Γ + π-policy (Γ dropdown) + mean±SD table, all 8 methods, N=700 8-seed. Plus a **"Larger N_train" section** (N=1500, 5 seeds, `run_owwin_bigN.py`/`build_owwin_bigN.py`, owwin_results_N1500.json) — win holds (Γ=8: DR-O-W 0.211>DR-O-X 0.148>DR-X-X 0.067).

**Large-Γ collapse panel (2026-06-22, `collapse_data.py`/`build_collapse.py`, owwin_collapse.json):** swept Γ→1000, capped, 3 seeds. KEY FINDING refining the user's "large Γ → collapse" intuition: only the REGRET method **Hajek-O-X collapses to never-treat** (treat-frac→0 by Γ≈5, value→0); the worst-case-VALUE methods (IPW-O-X, DoublyRobust-O-X) and OW do NOT collapse — they stay pinned at the 50% cap (DR-O-X value even RISES to 0.225, OW to ~0.265) because the self-normalised MSM box + per-arm calibration keep the worst case BOUNDED (adversary can only redistribute a fixed weight budget, not send value to −∞). Connects to [[rw_gamma_invariance]] (value+calibration optimum is Γ-stable; a collapse/Γ-responsive figure needs the regret objective). Two charts in the tab (value & treat-frac vs log10 Γ).

**ε-sweep + the cap/collapse tension (2026-06-22, `eps_sweep_data.py`/`build_epspanel.py`, owwin_epssweep.json):** user wanted ONE DGP giving BOTH (a) XX<OX<OW and (b) methods bad at large Γ. KEY TENSION found: the capacity cap is REQUIRED for the ordering (it forces the naive XX to commit to a wrong 50%) but the SAME cap PREVENTS collapse — it pins treat-frac at the cap so the value methods can't hedge to never-treat (capped DR-O-X/DR-O-W even RISE with Γ). Uncapped allows collapse but XX becomes good (0.206) → ordering breaks. Resolution = the **Wasserstein radius ε is the clean collapse knob**: uncapped ε-sweep (c_eps∈{0.5,1,2,4}, Γ→1000), larger ε → over-conservative → value falls at large Γ. DR-O-W: c_eps=0.5 flat ~0.24, c_eps=1 ~0.22, c_eps=2 →0.18, **c_eps=4 → 0.085** (0.238→0.085 collapse). IPW-O-W: c_eps=0.5 flat 0.25 → c_eps=4 ~0.08. So tight ε (c_eps=1) is the right operating choice; cranking ε AND Γ = assume-impossible-confounder-AND-balance-budget = give up. Tab now has THREE deep-dive panels: large-Γ collapse (Hajek→never-treat, value methods Γ-stable under cap), and the ε-sweep (DR-O-W & IPW-O-W value vs log10Γ per c_eps). Honest takeaway in HTML: ordering holds in the sensible regime; collapse appears when ε/Γ pushed unrealistic.

KEY: this confirms the [[wass_wins_experiments]] recipe on DISCRETE X — strong S~X correlation + mis-specified propensity (ε>0) + binding cap + non-deterministic (Gaussian) outcomes + confounding strong enough that XX is fooled. DR-family is the clean headline; IPW-O-X lucky-peak variance is the thing to tame. Next: lock the cleanest config, then build an index.html experiment tab. Don't use the counting propensity (ε=0). See [[no_true_propensities]], [[wasserstein_2arm_cap]].
