---
name: save-all-artifacts
description: Every experiment script must persist its underlying numeric data alongside the PNG plots. User strongly dislikes re-running anything.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 85317083-7452-4dfa-ba5f-16d73019ebb5
---

Every experiment script must save the underlying numeric arrays / dataframes
as artifacts (NPZ for arrays, CSV for tabular, JSON for small dicts) alongside
the PNG plot it produces. PNGs alone aren't enough — the user must be able to
re-render, re-analyse, or rebuild any plot/table from disk *without re-running*
any fit.

**Why:** the user has explicitly said multiple times that they don't like to
re-run experiments. The cost of a missing NPZ shows up downstream: any
follow-up question that needs the raw numbers (per-training-point policies,
per-rep IPW values, per-method grid curves) forces a fresh fit if the data
isn't on disk.

**How to apply:**

* In every new experiment driver, save an artifact with the same stem as the
  PNG (e.g. `policy_curves_gt1_00.png` ↔ `policy_curves_gt1_00.npz`).
* For RW LP fits: persist `pi0`, `pi1`, `mu`, `nu`, `beta`, `gamma_dual`,
  `theta` from `SolverResult` per (rep, Γ).
* For Kallus fits: persist `θ`, the chosen `Γ`, the truncation outcome.
* For RW-parametric fits: persist `θ` and the realised objective.
* For Direct IPW: persist `θ`.
* For Sample Oracle: derivable from the seed, but persist the per-i decisions
  anyway so a one-shot reader doesn't need to regenerate from the DGP.
* For policy-curve / heatmap plots: persist the X-grid and each method's
  evaluated policy values.
* CSVs that aggregate across reps should keep one row per (rep, method, Γ)
  rather than just the mean — variance / dispersion info is otherwise lost.

When introducing a new script, the saved-artifact step is **not optional** —
it should be the same priority as saving the PNG itself.
