---
name: saeed-paper
description: "Saeed's own method (notes in paper/saeed/policy_learning_unobserved_confounding__notes.pdf) — robust policy learning combining marginal-sensitivity bounds on Z with Wasserstein covariate-balance constraints, plus a Shapley extension rule."
metadata: 
  node_type: memory
  type: project
  originSessionId: 85317083-7452-4dfa-ba5f-16d73019ebb5
---

User's own research method, to be implemented under `code/saeed_implementation/` (parallel to [[kallus-paper]] and [[kallus-repo]]).

**Population problem (P).** sup_π inf_Z E[1{T=0}(1−π(X))Y/Z + 1{T=1}π(X)Y/(1−Z)] over policies π and latent confounded propensities Z ∈ S_Γ. The set S_Γ requires:
- (C1) Marginal sensitivity: Γ⁻¹ ≤ [e(X)/(1−e(X))]·[(1−Z)/Z] ≤ Γ — same as Tan/Rosenbaum/Kallus.
- (C2) Calibration: E[1/Z·1{T=0}]=1 and E[1/(1−Z)·1{T=1}]=1.
- (C3) **Wasserstein covariate balance** — W_d(P_X, Q_X^{t,Z})=0 for t∈{0,1}, where Q^{t,Z} are arm-specific re-weighted distributions. **This is novel vs. Kallus.**

**Empirical problem (P̂).** Decision variables: in-sample policy values p_i∈[0,1], inverse-weight candidates w_i, transport plans ζ^t_{ij}≥0. Box constraints encode (C1); transport block (with marginals 1/n on the source side, w_j/n on the target side, budget Σd(X_i,X_j)ζ^t_{ij}≤ε_t) encodes (C3) AND implies Σ_{j∈I_t} w_j = n (i.e. (C2)). It's a bilinear saddle-point: outer max in π, inner LP in (w,ζ).

**Shapley extension (Lemma 1).** Given the in-sample optimum π̂(x̂_k), extend to all x ∈ X via π^S(x)=min_k max_j A_{jk}(x), where A_{jk}(x) is the inverse-distance convex combination of π̂(x̂_j) and π̂(x̂_k). Optimal for the DRO extension problem (Wasserstein ambiguity on (X,Y) per arm) **for any reference distribution Q^t and any radius ρ_t** — that's the strong robustness claim.

**Key differences from [[kallus-paper]]:**
1. Adds Wasserstein covariate balance — Kallus has only the marginal-sensitivity box.
2. Sign convention is **utility, not loss** (outer is sup_π over E[π(X)Y/...]) — flip internally when comparing.
3. Empirical objective is **unnormalized Horvitz-Thompson** ((1/n)Σ), not Hájek; calibration comes from transport marginals.
4. No parametric policy class — n free decision variables p_i∈[0,1], then **Shapley** for extension. Kallus uses logistic/tree classes directly.

**Why:** User is building this as their own research contribution, layering Wasserstein-DRO on top of the Kallus-style sensitivity model.

**How to apply:** When implementing in `code/saeed_implementation/`, default to Gurobi for the LP (user offered a license), Euclidean distance on standardized X, n free in-sample p_i variables, single-shot LP via inner-duality. Code Dataset interface mirrors `crpo.Dataset` so synthetic DGPs (e.g. [[kallus-paper]] §7.1.1 or DGP Case 2) can be reused.
