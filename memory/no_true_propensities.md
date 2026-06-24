---
name: no_true_propensities
description: Never feed true/oracle propensities (e_true) or the confounder S to any method — always ESTIMATE P(T|X) from data.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

The user requires that **no method ever uses the TRUE propensities** (`e_true`) or the unobserved confounder
`S` as an input. Propensities must always be **estimated from the observed (X, T)** (logistic regression in
`code 1.1/common/propensity.py`; the existing `srpo`/runners already do this via `propensity_matrix(tr.X, tr.T)`).

**Why:** the whole problem is policy optimization under **UNOBSERVED** confounding — the analyst only has the
nominal (estimated, S-ignorant) propensity, and the MSM/Rosenbaum sensitivity model bounds how wrong it is.
Feeding oracle propensities would be cheating and would invalidate every result.

**How to apply:** the `common` propensity API takes only `(X, T)` — it is *structurally* unable to access
`e_true`. `e_true` and `S` are saved as raw-draw provenance and are used ONLY in the Rosenbaum DGP-validation
diagnostic (to confirm matched Γ = e^{γ/2}) — never to weight, fit, or optimize any policy. When wiring any
method, verify its weights come from `common.ipw_weights_from_data(X, T, …)`, not from the DGP's `e_true`.
See [[saeed_paper]], [[code_1_1_overhaul]].
