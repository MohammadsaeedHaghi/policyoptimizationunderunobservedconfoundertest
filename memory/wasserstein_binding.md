---
name: wasserstein_binding
description: When the RW Wasserstein constraint actually binds and helps vs odds-only (RW-noW).
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

The RW uncertainty set is (odds-ratio box) ∩ (Wasserstein ball, radius ε_t = c_ε·good_epsilon(t)).
Whether the Wasserstein term does anything depends on **Γ**, not just c_ε.

Diagnostic on case4 (γ_true=1, `ceps_diagnostic.py`, mean of 3 reps):
- **Γ=2**: at c_ε≥0.75 the transport dual β=0 (non-binding) → RW ≡ RW-noW (test 0.789=0.789). Only c_ε≤0.6 makes it bind, and the gain is marginal.
- **Γ=4**: at the **default c_ε=1** β≈1.55 (**binds**) → RW test **0.799 vs RW-noW 0.770** (~3-pt win for Wass+odds). Tightening to c_ε=0.75 nudges to 0.8015.

Takeaways:
- **c_ε=1 is the right default for the full batch**: it is guaranteed feasible (ε=good_epsilon is exactly achievable at the nominal weights), and the Wasserstein term already binds and helps once Γ is large enough (≥4) that the box is wide. c_ε<1 is NOT guaranteed feasible (the full LP can still find a feasible w, but it can also fail).
- So "RW (Wass+odds) > RW (odds only)" shows up at the **high-Γ end** of the sweep, not at Γ≈1–2. Earlier confusion came from the smoke test using only Γ∈{1,3}, which sits in the non-binding regime.
- This is the empirical answer to "does the Wasserstein constraint matter" — it does, where confounding is strong. See [[saeed_paper]]; diagnostic artifacts saved as `out/case4/ceps_diagnostic_*`.
