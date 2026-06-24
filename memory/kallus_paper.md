---
name: kallus_paper
description: "Core method, formulas and algorithm from Kallus & Zhou 2021 \"Minimax-Optimal Policy Learning Under Unobserved Confounding\"."
metadata: 
  node_type: memory
  type: project
  originSessionId: 85317083-7452-4dfa-ba5f-16d73019ebb5
---

**Paper:** Kallus & Zhou, "Minimax-Optimal Policy Learning Under Unobserved Confounding," *Management Science* 67(5), 2021, pp. 2870–2890. PDF lives at `paper/kallus/kallus.pdf`.

**Problem.** Learn a personalized treatment policy `π: X → Δ^m` from observational tuples `(X_i, T_i, Y_i)` (Y is loss; lower is better; T ∈ {0,…,m-1}). Standard unconfoundedness (`ê_t(x) = P(T=t|X=x)` equals true `e_t(x,y) = P(T=t|X=x,Y(t)=y)`) often fails. Without it, point identification of policy value is impossible, so optimize **worst-case regret against a baseline π₀** over an uncertainty set on inverse propensity weights.

**Marginal Sensitivity Model (Tan 2012), parameterized by Γ ≥ 1.** Per-unit odds-ratio bound between true and nominal propensity ⇒ uncertainty set on true inverse weights `W_i* = 1/e_T(X_i, Y_i)`:
- `a_i^Γ = 1 + Γ^{-1}(W̃_i − 1)`,   `b_i^Γ = 1 + Γ(W̃_i − 1)`,   where `W̃_i = 1/ê_T_i(X_i)` is the nominal inverse propensity.
- Empirical set: `W^Γ_n = {W ∈ R^n : a_i^Γ ≤ W_i ≤ b_i^Γ, ∀i}`.
- Γ=1 → unconfoundedness; Γ=∞ → no restriction.

**Per-treatment Hájek regret estimator** (this is the key vs. their 2018 paper — normalization is *per t*, not global):
`R̂_π₀(π; W) = Σ_t  E_n[(π(t|X) − π₀(t|X)) · 𝟙[T=t] · Y · W] / E_n[W · 𝟙[T=t]]`.

**Worst-case (over W) regret** and **minimax-optimal policy**:
- `R̂̄_π₀(π; W^Γ_n) = sup_{W ∈ W^Γ_n} R̂_π₀(π; W)`.
- `π̂ = argmin_{π ∈ Π} R̂̄_π₀(π; W^Γ_n)`.

**Theorem 3 (sort-based inner solver).** For a fixed policy, the inner sup over W is a linear-fractional program. Sort r-values `r_i = (π(T_i|X_i) − π₀(T_i|X_i)) Y_i` (per treatment t). The optimum has a threshold k*: indices below k* use `a_i^Γ`, above use `b_i^Γ`. Concretely
`λ(k) = (Σ_{i<k} a_(i) r_(i) + Σ_{i≥k} b_(i) r_(i)) / (Σ_{i<k} a_(i) + Σ_{i≥k} b_(i))`
is discrete-concave, unimodal — solvable by linear scan or ternary search.

**Budgeted set `W^{Γ,Λ}`.** Adds total-variation budget per treatment `(1/|J_t|) Σ_{i∈J_t} |W_i − W̃_i| ≤ Λ_t`. Practitioner sets `Λ_t = ρ · (1/|J_t|) Σ max(W̃ − a^Γ, b^Γ − W̃)` for ρ ∈ (0,1). Inner subproblem becomes an LP (Charnes–Cooper transformation), still tractable.

**Algorithm 1 (subgradient descent, parameterized policy θ).**
For k = 0,…,N−1:
  1. step size `η_k = η_0 (k+1)^{−κ}`,  κ ∈ (0,1] (paper uses κ=0.5).
  2. Inner: find worst-case W given current θ via Theorem 3 (or budgeted LP).
  3. Subgradient on θ: `g(θ; W) = Σ_i W_i / Σ_j W_j · (∇_θ π_θ(T_i|X_i) − ∇_θ π₀(T_i|X_i)) · Y_i` (per-treatment normalization).
  4. `θ_{k+1} = Proj_Θ(θ_k − η_k · g)`.
Return mean of `θ_t` across iterations; run multiple restarts and take best robust objective; if best objective is positive, return baseline (do-no-harm fallback).

**Theoretical guarantees.**
- *Improvement (Theorem 1):* worst-case empirical regret asymptotically upper-bounds the true population regret, at rate `O(n^{−1/2})` with sub-Gaussian tails, dependent on VC-major dim of Π. So if `π₀ ∈ Π` and objective ≤ 0, no harm (up to vanishing terms).
- *Minimax optimality (Theorem 2 / Corollary 2):* uniform convergence over Π and the *monotone* sub-class of W^Γ (Proposition 2/3 — monotone weight solutions form a VC-hull class with finite VC-major dim). So π̂ attains the population minimax-optimal regret.
- *Estimated propensities (Proposition 6):* additive bound in terms of `1/ê_T_i − 1/ẽ_T_i` error.

**Empirics in paper.**
- Sec 7.1: synthetic binary (n=200, d=5, true Γ=1.5 created by setting `e(X,U) = (4 + 5U + ẽ(X)(2 − 5U))/(6ẽ(X))`); plots out-of-sample policy regret vs log(Γ).
- Sec 7.1.2: synthetic with 3 treatments (multinomial logistic).
- Sec 7.2: WHI semisynthetic — train on observational (with logistic-regression nominal propensities), evaluate on actual randomized clinical-trial data via Horvitz–Thompson. Outcome `Y = S + Tλ` (blood-pressure + treatment-effect scalarization).
- Sec 8: calibration plots (train policy under Γ_k, evaluate worst-case regret under Γ_k′ for all k,k′) to choose Γ.

**Key notation gotcha.** In code, treatments are often *signed* `t ∈ {−1, +1}` for binary (rather than `{0,1}`); the function `get_sgn_0_1` converts. Multi-treatment uses integer-coded T.

See [[kallus_repo]] for reference implementation file map and pitfalls.
