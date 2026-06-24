---
name: owwin2-nocap-dgp
description: "E3 DGP — discrete-X, XX<OX<OW AND large-Γ collapse WITHOUT a capacity cap (works both capped & uncapped)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 58aeb24f-db15-4091-9dfb-a7f743c17425
---

**Goal (2026-06-22):** user wanted a discrete DGP giving BOTH (XX<OX<OW) AND (methods bad at large Γ) **without** a capacity cap, ideally working with AND without cap. The [[discrete-ow-wins-dgp]] owwin needed the cap for the ordering (cap forces naive XX into a wrong 50%); uncapped owwin's XX was too good. Fix: make the confounding LARGE+STEEP so naive XX over-treats into the harmful region on its own.

**Winning DGP "E3"** (`assets/exp_owwin2/`, search `assets/exp_dgp_search/search_uncap{,2}.py`, verify `verify_E3.py`): discrete X 7-level grid, S∈{-1,+1}:
- P(S=+1|X)=σ(**7**X) (strong S-X correlation)
- e(X,S)=σ(**4**S − **2**X), clipped [0.02,0.98] (STRONG selection on S = big confounding)
- μ0=**5**S, μ1=5S + **3**X ⇒ CATE=3X, oracle treat iff X>0; noise=**0.9** Gaussian
- methods FIT logistic P(T|X); oracle≈0.857.

**N=500, 5 seeds, BOTH regimes (verify_E3.log):**
- UNCAPPED: XX=0.286 < OX=0.695 < OW=0.769 (clean big margins). Large-Γ COLLAPSE of both robust families: IPW-O-X 0.686(Γ8)→0.131(Γ1000), DR-O-X 0.695→0.211, DR-O-W 0.769→0.66, IPW-O-W 0.727→0.671. So XX<OX<OW AND methods-bad-at-large-Γ, no cap.
- CAPPED (≤50%): XX=0.196 < OX=0.662 < OW=0.664 (ordering holds, OW/OX margin thin capped); DR-O-X collapses 0.662→0.245.

WHY uncapped works here (vs owwin): strong confounding (d=5, cs=4) makes naive XX over-treat into harmful X<0 region on its own → XX bad (0.286) even uncapped → room for OX/OW. WHY collapse uncapped: no cap → robust methods can hedge to never-treat (value 0) as the box widens → value falls. Connects to the cap/collapse tension in [[discrete-ow-wins-dgp]] (cap pins treat-frac, blocks collapse; uncapped allows it). To be built as a new experiment tab (Seventh) with both regimes + value-vs-Γ to 1000 showing collapse. See [[wass_wins_experiments]], [[no_true_propensities]].
