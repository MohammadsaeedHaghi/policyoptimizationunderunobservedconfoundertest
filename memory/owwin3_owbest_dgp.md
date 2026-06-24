---
name: owwin3-owbest-dgp
description: "I3 DGP — OW family (IPW-O-W & DR-O-W) beats EVERY other method (incl plain DR-X-X) at SMALL Γ, both regimes, discrete"
metadata: 
  node_type: memory
  type: project
  originSessionId: 58aeb24f-db15-4091-9dfb-a7f743c17425
---

**Goal (2026-06-24):** user wanted a discrete DGP where BOTH IPW-O-W and DoublyRobust-O-W beat ALL other methods (incl DR-O-X AND plain DR-X-X, Direct, IPW-O-X, IPW-X-X, Hajek), in BOTH capped & uncapped, at a SMALL operating Γ (E3's Γ=20 was too large), and try different ε.

**Winning DGP "I3"** (search `assets/exp_dgp_search/search_owbest.py`, verify `verify_I.py`): discrete X 7-level grid, S∈{-1,+1}:
- P(S=+1|X)=σ(**10**X) (strong S-X correlation)
- e(X,S)=σ(**0.8**·S − 2X) — MODERATE selection (cs=0.8 → matched Γ=e^1.6≈5, so the operating Γ is SMALL)
- **μ0=5S, μ1=6S + X** — S boosts BOTH arms strongly (5 & 6 → badly biases the outcome model) with a SMALL differential d1−d0=1 (subtle treatment×vitality synergy → CATE=S+X stays subtle so plain DR is beatable). noise=0.6.
- methods FIT logistic P(T|X); **tight ε (c_eps=1) is best** (larger ε hurt DR-O-W).

**THE KEY INSIGHT that cracked "OW beats plain DR-X-X":** earlier DGPs had additive confounding identical in both arms (μ0=μ1=dS+...) which CANCELS in the treat-vs-control comparison → plain DR-X-X stays strong/unbeatable. Fix = NON-cancelling confounding via a treatment×S interaction (d1≠d0), but with d0,d1 LARGE (strong model bias) and d1−d0 SMALL (subtle CATE, else CATE is huge & X-aligned → trivial, all methods tie at oracle). Large-d-same-arms and big-differential both FAIL; large-d + small-differential is the sweet spot.

**N=500, 5 seeds, BOTH regimes (verify_I.log):** oracle 0.70.
- UNCAPPED: comp max = DR-O-X 0.560; IPW-O-W=0.611, DR-O-W=0.595 at **Γ=1.5** → both beat all. (Hajek 0.277, Direct 0.108 weakest.)
- CAPPED (≤50%): comp max = DR-O-X 0.580; IPW-O-W=0.617, DR-O-W=0.607 at **Γ=2** → both beat all.
Margins modest (+0.03..0.05) but consistent at small Γ in both regimes. I1 (d0=4,d1=5,cs=1) also works (Γ=2/3) with bigger uncap margin; I3 preferred for smaller Γ.

Story (clinical): aggressive treatment with a mild synergy with hidden vitality S (helps the vital a bit more, d1>d0); vitality dominates outcomes & tracks fitness X; clinicians escalate the vital → naive over-escalates; covariate balance (OW) recovers the right policy. To build as the Eighth experiment tab. See [[owwin2-nocap-dgp]], [[discrete-ow-wins-dgp]], [[wass_wins_experiments]].
