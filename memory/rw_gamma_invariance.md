---
name: rw_gamma_invariance
description: Key finding — the RW (worst-case value + per-arm calibration) optimal policy is Γ-invariant.
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

**The RW realised-outcome curve is flat in Γ, and this is structural, not a bug or a tuning issue.** Established 2026-06-02 over ~16 DGP configurations (case4.1, CATE-crossing-zero, poor-overlap ×2, continuous-Y, extreme two-region, controlled heterogeneous ŵ from 1.1–14.4, strongS, 5D; γ_true up to 50; Γ up to 30; ε from tight to very loose; free-policy and parametric). In every one the in-sample RW policy was **byte-identical across Γ** (`max|Δπ| = 0`); only the worst-case *objective* moved.

**Why:** RW maximises worst-case **value** with the per-arm Hájek calibration `Σ_{i∈I_t} w_i = n`. That restricts the adversary to a zero-sum redistribution (`Σδ=0`); widening the box with Γ lowers every policy's worst-case value by an amount that preserves their ranking ⇒ the Γ=1-optimal policy stays optimal for all Γ. Removing the calibration restores Γ-dependence but collapses to the degenerate `w=aᵢ` "memorise" policy (which is why the calibration was added).

**Two non-obvious corollaries:**
- Raising γ_true (true confounding) does NOT help: the *estimated* ê(X)=P(T=1|X) marginalises out the unobserved S, so observed ŵ stays moderate (≈2–8) no matter how strong the true confounding. Confirmed ê∈[0.32,0.97] even at γ_true=50.
- Even forcing extreme/heterogeneous observed ŵ by hand was still flat ⇒ it's the calibration, not overlap magnitude.

**Implication for the figure the user wants:** a Γ-responsive policy requires changing the objective to **worst-case regret / improvement-over-control** (where the worst case can go negative so the treat threshold moves with Γ) — like Kallus, which does respond (via its do-no-harm truncation cliff). DGP/ε cannot produce it. User has not yet greenlit building the regret variant. See [[saeed_paper]], [[wasserstein_binding]].

New DGPs `case8` (strongS) and `case9` (5D) were added (common/dgps/, DGP/ folder, HTML Experiments sub-tabs) as documented Γ-flat stress cases.
