---
name: regret_selfnorm_fix
description: "Regret-style methods (Regret-O, Regret-OW, Kallus) must use the SELF-NORMALISED Hájek regret (floating denominator), not a fixed Σw=n calibration."
metadata: 
  node_type: memory
  type: project
  originSessionId: 58aeb24f-db15-4091-9dfb-a7f743c17425
---

The Kallus & Zhou regret estimator is the **per-treatment self-normalised Hájek ratio** (their Eq. 4, 10):
`R̂^(t)(π;W) = [Σ_{T_i=t}(π(t|x_i)−π₀(t|x_i))Y_i W_i] / [Σ_{T_i=t} W_i]` — the denominator **floats with W** (a linear-FRACTIONAL program; Charnes–Cooper / Thm 3). π₀ = all-control.

**The bug (fixed 2026-06-16).** The original `code 1.1` regret solvers used a *fixed* normalisation instead:
- **Regret-O** (box-only) imposed a hard `Σ_{T_i=k}W_i=n` calibration constraint + `(1/n)ΣrW` objective — WRONG (pins the denominator).
- **Kallus** (box-only, the only variant the runners use) did the same `Σ_{I_k}w=n` calibration — WRONG.
- **Regret-OW** was **already correct**: its Wasserstein transport mass-balance (demand 1/n, supply W_j/n) *forces* `Σ_{I_k}W=n` per arm, so its `(1/n)ΣrW` value already equals the Hájek ratio. **Do NOT "fix" Regret-OW** — reverting it was necessary (a CC rewrite conflicts with the transport mass-balance and makes Γ=1 infeasible).

**The fix.** New `common/hajek_regret.py`: `selfnorm_box_dinkelbach` (fast root-find) + `selfnorm_box_lp` (Charnes–Cooper, cross-checked to machine precision) + `selfnorm_wasserstein_dinkelbach`.
- Regret-O capped/uncapped: replaced the calibration dual with the CC dual (Eq. 12) embedded in the free-π min: `min_{π,u,v,λ} Σ_t λ_t s.t. v_i−u_i+λ_{T_i} ≥ (1[t=0]−π_t)Y_i, Σ_{I_t}(u_i a_i − v_i b_i) ≥ 0`. Verified: embedded objective == independent inner sup for the returned π; Γ=1 ⇒ exactly the IPW maximiser.
- Kallus inner: returns the per-arm **normalised** worst-case w (Σ_{I_t}=1); the subgradient drops the `/n` (uses normalised w directly). Box via Dinkelbach (fast); Wasserstein path exists but is never called (runners use `wasserstein=False`).

**KEY box-weight subtlety.** The self-normalised box MUST be built on the **RAW** inverse weights `W̃=1/ê ≥ 1` (so `a_i = 1+Γ⁻¹(W̃−1) ≥ 1 > 0` and the floating denominator stays positive). The codebase's per-arm **Hájek-normalised** weights can dip below 1 → `a_i < 0` at Γ≈12 → zero/negative denominator → ratio blowup. So: Regret-O & Kallus get `w_raw = 1/P[arange,T]` (from the `P` matrix of `ipw_weights_from_data`); Regret-OW keeps Hájek weights (its transport pins ΣW=n, denom = n > 0 always). The VALUE methods (R-OW/R-O/IPW/AIPW/DR) are unchanged (Hájek + their own calibration) — do NOT touch them. See [[no_true_propensities]], [[kallus_paper]].

**Reward sign (Y high = better, opposite to the paper's loss).** The coefficient is unchanged: `r_i=(1[T_i=0]−π_{T_i})Y_i` with `Yc=Y` when `maximize=True`. Only the normalisation changed.

**Re-run.** Surgical regret-only rerun scripts merge ONLY Regret-O/Regret-OW/Kallus (both regimes, all Γ, + per-Γ grids) into the existing `_multiseed/*.json`, leaving the other 6 methods intact: `assets/exp_1d/rerun_regret.py` (n=500, 3 seeds), `assets/exp_first/rerun_regret.py` (5 seeds, ZS=False), `assets/exp_second/rerun_regret.py` (5 seeds, ZS=True, 2-D grid). Then each dataset's agg + perf_plots + build_*chart regenerate plots/index.html. Regret-OW (Wasserstein LP, ~n² constraints at n=500) is the runtime bottleneck (~6–8 min/dataset). Empirically the self-norm fix barely moved exp_1d (Regret-O/Kallus still ≈ never-treat at the wide Γ≈12 box — the inherent do-no-harm conservativeness). READMEs (Regret-O/Kallus/Regret-OW .html) updated to the self-normalised definition.
