# Saving experiment results — instruction (code 1.1)

This document is the **authoritative spec** for how every experiment in `code 1.1` persists its results.
The rule of the project is *save everything*: a result is not "done" until its raw numeric artefacts are on
disk next to (or instead of) any plot. A reader should be able to recover **every policy, every number, and the
exact DGP** from this folder **without opening the code**.

> Re-running experiments is expensive and to be avoided. If a later question needs a number, it must already be
> on disk. When in doubt, save it.

---

## 0. Principle

For each experiment we sweep these dimensions:

| dimension | values | where it lives |
|---|---|---|
| **experiment** (DGP scenario) | conti2d-radial-K4, conti-binary-K2, discrete-expA-K3, … | top-level folder |
| **method** | R-OW, R-O, Hajek-OW, Regret-O, IPW, Direct-X-X, DoublyRobust, Oracle, Kallus | folder |
| **capacity** | Capped / Uncapped (Kallus = no cap, parametric) | folder |
| **γ_true** | true confounding level (3, 4, 5, …) | folder |
| **Γ** (`\bigGamma`) | the sensitivity-model sweep {1, 2, 3, 5, 7, 10} ∪ {e^{γ/2}} | **key inside files** |
| **seed** / replication | seed₀ … seed₀+R−1 | **key inside files** |
| **extension** | Shapley, KNN (off-support deployment) | **key/column inside files** |

Folders for the slow-moving dimensions (experiment → method → cap → γ_true); Γ and seed are **keys inside the
files** (a folder per Γ×seed would explode into thousands of dirs).

---

## 1. Directory layout & naming

```
results/
├── SAVING.md                                   ← this file
└── <experiment>/                               one folder per DGP scenario  (§2 for the slug)
    ├── config.json                             full provenance: DGP params + protocol + Γ-grid + seeds + caps
    ├── data/
    │   ├── seed-9000.npz                        raw DGP draws for this seed (train + test)
    │   └── seed-9001.npz                        …
    ├── reference/
    │   ├── full-info.npz                        clairvoyant Sample-Oracle (knows both counterfactuals)
    │   ├── best-true-means.npz                  best deployable policy on the true means μ_k(X)
    │   └── best-arm-nocap.npz                   unconstrained argmax-μ ideal (the "no-cap" ideal map)
    ├── rosenbaum.csv                            matched-Γ diagnostic, per γ_true (+ pooled large-N row)
    └── <Method>/<Capped|Uncapped>/gtrue-<γ>/    e.g.  R-OW/Capped/gtrue-3/
        ├── meta.json                            method, cap vector, γ_true, matched Γ, ε, knn_k, support size
        ├── policy.npz                           OPTIMAL on-support policy + support_X + ALL duals  (keyed by Γ, seed)
        ├── extended.npz                         DEPLOYED policy: Shapley & KNN, on train / test / grid (keyed by Γ, seed)
        ├── outcomes.csv                         one row per (Γ, seed, extension): realised + expected, usage, objective
        └── figures/
            ├── results.png   + results.npz       Γ-sweep outcome curves (+ its plotted data)
            ├── deploy.png    + deploy.npz         2×2 deployed train/test, realised + E[Y]
            ├── policy.png    + policy.npz         arm-choice map (2-D heatmap or 1-D strip) at matched Γ
            └── usage.png     + usage.npz          per-arm capacity usage vs Γ
```

- **Kallus** is parametric (no capacity), so it has **no `Capped|Uncapped` level**:
  `Kallus/gtrue-<γ>/…`.
- **Method / cap folder names** reuse the hyphenated labels from `methods/` exactly: `R-OW`, `R-O`, `Hajek-OW`,
  `Regret-O`, `IPW`, `Direct-X-X`, `DoublyRobust`, `Oracle`, `Kallus`; `Capped`, `Uncapped`.
- **γ_true folder**: `gtrue-3`, `gtrue-5` (integer γ). If a study varies sample size or another knob, append it
  to the **experiment** slug, not the γ folder (e.g. `conti2d-radial-K4-n1000`).

---

## 2. The experiment slug

`<family>-<structure>-K<arms>[-<variant>]`, lower-case, hyphenated. Examples:

| slug | meaning |
|---|---|
| `conti2d-radial-K4` | 2-D continuous X, radial 4-arm DGP, default N |
| `conti2d-radial-K4-n1000` | same, N_train = 1000 variant |
| `conti-binary-K2` | 2-D continuous X, binary treatment (2-arm) |
| `conti-binary-K2-nogrid` | same, free-π LP on raw X (no 6×6 snapping) |
| `discrete-expA-K3` | 1-D discrete-X grid, case exp_a, 3 arms |

The slug must be unique per DGP+protocol so two experiments never collide.

---

## 3. What to save — exhaustive catalog

### A. Config / provenance — `config.json` (one per experiment)
Everything needed to regenerate the experiment without the code:
- `experiment` slug, `dgp_module` (e.g. `experiments.conti_binary.dgps`), `created` (ISO timestamp), code ref/commit if available.
- **`dgp_params`**: the full `PARAMS` dict verbatim (θ₀, θ₁/θ_base, A, c, W2/proj weights, α, B, d, …) + `n_arms`.
- **`protocol`**: `n_train`, `n_test`, `seed0`, `n_reps`, `gamma_true_grid`, `Gamma_sweep` (the swept Γ list per γ),
  `matched_Gamma` = e^{γ/2} per γ_true, `cap` vector(s), `POLICY_GRID`/binning, `C_EPS`, `knn_k`, the tight per-arm
  Wasserstein radius `epsilon`.
- Human-readable **DGP formulas** (the LaTeX/text of potential-outcome, assignment, μ, capacity) so the spec is
  self-describing.

### B. Raw DGP draws — `data/seed-<NNNN>.npz` (one per seed)
The actual sampled data, train **and** test, so nothing must be re-drawn:
- `train_X, train_T, train_Y, train_Ypot, train_S, train_e_true, train_mu`
- `test_X,  test_T,  test_Y,  test_Ypot,  test_S,  test_e_true,  test_mu`

(`Ypot` = per-unit potential outcomes (n,K); `mu` = true outcome means μ_k(X) = E[Yᵏ|X], S-marginal; `e_true` =
true propensities; `S` = the unobserved confounder.) These come from each DGP's `generate()` / `outcome_means()`.

### C. Learned optimal policy + duals — `policy.npz` (per method/cap/γ_true)
The **optimal policy for each method and each Γ** on its support, plus everything to reconstruct it:
- `support_X` — the covariate support the LP was solved on (6×6-snapped grid, or raw X for `-nogrid`).
- `cap` — the per-arm capacity vector used (all-ones for Uncapped).
- For each Γ in the sweep and each seed:
  - `pi__G<Γ>__seed<s>` — the (K, n_support) optimal policy (the LP solution).
  - `obj__G<Γ>__seed<s>` — `objective_value` (worst-case / IPW value the method optimised).
  - `usage__G<Γ>__seed<s>` — per-arm realised usage `(1/n)Σ π_k` (≤ cap by construction when Capped).
  - **duals** (so the worst-case weights are recoverable): `beta__…`, `mu__…`, `nu__…`, `gamma_dual__…`,
    `theta__…` — the `MultiArmResult` multipliers. **These fields are overloaded across methods — you MUST also
    store a semantics label (see C′ below), or the stored numbers are uninterpretable.**
  - `obj_kind__G<Γ>__seed<s>` — `worstcase_value` (R-OW, R-O) / `worstcase_regret` (Hajek-OW, Regret-O; ≤0) /
    `hajek_ipw` (IPW) / `factual` (Direct-X-X) / `oracle`.
  - `status__G<Γ>__seed<s>` — solver status (optimal / fallback).
- **Parametric methods (Kallus)**: instead of an on-support `pi`, save `theta__G<Γ>__seed<s>` (the learned
  softmax parameters), the Γ it was fit at, and `fallback__…` (whether do-no-harm/revert-to-control fired).
- **DoublyRobust**: save the learned policy/`theta`, plus the **nuisance fits** it relies on — the outcome-model
  parameters and the propensity model — and the DR value estimate, so the doubly-robust correction is reproducible.

### C′. Dual semantics & uncertainty-set inputs (so duals are interpretable + worst-case weights reconstructible)

> **⚠ The dual fields are field-overloaded.** `MultiArmResult` reuses `mu, nu, beta, gamma_dual, theta` across
> solvers with **different meanings**. Store the semantics (per method) alongside the numbers — otherwise a saved
> `beta = 2.15` is ambiguous (a Wasserstein radius? a calibration multiplier?).

| method | `obj_kind` | `mu, nu` | `beta` | `gamma_dual`, `theta` |
|---|---|---|---|---|
| **R-OW** | worstcase_value | box duals | Wasserstein radius (≥0) | **present** (transport duals) |
| **R-O** | worstcase_value | box duals | Hájek calibration (free sign) | — (None) |
| **Hajek-OW** | worstcase_regret (≤0) | Kallus regret-box `p, q` | Wasserstein radius (≥0) | **present** |
| **Regret-O** | worstcase_regret (≤0) | Kallus regret-box `p, q` | Kallus calibration (free sign) | — |
| **IPW / Direct-X-X** | hajek_ipw / factual | NaN | NaN | — |
| **Oracle** | oracle | — | — | — |

**Uncertainty-set inputs** (save per seed; the duals above + these reproduce the exact adversarial reweighting):
- `ips_weights` (ŵ, shape n), `P` (propensity matrix n×K from `propensity_matrix`), `D` (n×n distance matrix
  used by the Wasserstein term), and per (method, Γ): `Gamma`, `epsilon` (K — R-OW/Hajek-OW only),
  `a_box`, `b_box` (shape n, = `marginal_sensitivity_box(ŵ, Γ)`).

**Do-no-harm (Kallus family) is DERIVED — there is no stored boolean.** Record `dnh_active = (objective_value ≥ −tol)`,
`worst_case_regret = objective_value`, and `arm0_mass = pi[0,:].mean()`.

### D. Extended / deployed policy — `extended.npz` (per method/cap/γ_true)
The policy you actually deploy off-support, via **both** extensions, on every evaluation set:
- `ext_<shapley|knn>__<train|test|grid>__G<Γ>__seed<s>` — the (K, n) extended policy.
  - `train` = re-evaluated on that seed's train X; `test` = on the test X; `grid` = on a dense (X₁,X₂) grid /
    1-D X grid for the policy-map figure.
- `grid_X` — the dense grid coordinates the `…__grid__…` arrays are evaluated on (save once).
- **On-support note:** Shapley and KNN agree exactly *on the support* (renormalisation is a no-op); they differ
  only off-support (test X, dense grid), and the extended `test_use_k` may overflow the cap (the extension is
  unconstrained). Save **both** so this is visible.
- **Policy-map reference layers** (for the arm-choice figure; save once per γ_true at the matched Γ, seed0):
  `argmax_mu` (best arm, no cap), `oracle_cap_arm` (best-true-means under cap), `fi_arm` (Full-info per-unit) —
  on `grid_X` (or the seed0 support for `fi_arm`).

### E. Outcomes — `outcomes.csv` (per method/cap/γ_true)
**One row per `(seed, Gamma, extension)`** (keep per-rep rows — never only the mean):

| column | meaning |
|---|---|
| `seed`, `Gamma`, `extension` | the cell (extension ∈ {shapley, knn}) |
| `realized_train` | raw on-support LP policy, realised on train Ypot (in-sample fit) |
| `realized_train_dep` | **deployed** (extended) policy, realised on train |
| `realized_test` | deployed policy, realised on test |
| `exp_train`, `exp_test` | **expected** E[Yᵖ] on the true means μ (noise-free), train / test |
| `obj_ipw` | the method's worst-case / IPW objective value |
| `train_use_0..K-1`, `test_use_0..K-1` | per-arm usage (train = LP ≤ cap; test = extended, may overflow) |
| `cap_0..K-1` | the capacity vector (for convenience) |

Also write `outcomes-summary.csv` (across-seed mean ± SE per (Γ, extension)) **in addition to**, never instead
of, the per-rep rows.

### F. Reference ceilings — `reference/*.npz` (per experiment, cap-aware)
- `full-info` — clairvoyant Sample-Oracle on realised Ypot (`solve_oracle_capacity(X, Ypot, cap)`): policy +
  realised value + usage, per seed. The realised ceiling.
- `best-true-means` — best deployable policy on μ_k(X) under the cap: policy + realised + **E[Y]** value. The
  noise-free deployable ceiling.
- `best-arm-nocap` — unconstrained argmax_k μ_k(X) (the ideal arm map, no capacity).
  Save the seed0 per-unit decisions (`fi_arm`) for the policy-map scatter.

### G. Diagnostics — `rosenbaum.csv` (per experiment)
Per γ_true (and a pooled large-N row): `lambda_hat` (realised marginal-sensitivity Λ̂), `matched_gamma` =
e^{γ/2}, `p95_OR`, `median_OR`, `ratio_to_matched`, plus the extra fields `diagnostics()` returns but that are
otherwise dropped: `log_lambda_hat`, `per_arm_lambda` (arm → max OR in arm). Save the per-unit odds-ratio array
`OR` (NPZ) with its companions `e_marg` (S-marginal nominal propensity at T_i) and `e_cond` (= e_true) whenever
the histogram is produced.

### H. Figures — `figures/*.png` + same-stem data
Every PNG gets a same-stem `.npz`/`.csv` holding exactly the plotted series (so the plot can be re-rendered or a
number re-read without re-fitting): `results`, `deploy`, `policy`, `usage` (+ `rosenbaum` at experiment level).

---

## 4. File-format & key conventions

- **NPZ** for arrays (policies, duals, draws, extended policies, grids): `np.savez_compressed`.
- **CSV** for tabular sweeps (outcomes, rosenbaum) — one row per replication cell, never pre-aggregated only.
- **JSON** for config/meta (human-readable provenance).
- **Key template** inside NPZ: `<quantity>__G<Γ>__seed<seed>` — e.g. `pi__G4.4817__seed9000`,
  `mu__G12.1825__seed9003`, `ext_shapley__test__G3.0__seed9000`. Use the rounded Γ (4 d.p.) consistently with the
  CSV `Gamma` column so files cross-reference.
- **Γ in names**: `G` + value (`G4.4817`). **γ_true in folders**: `gtrue-<int>`.
- Numbers in CSV at full precision (don't round away variance).

---

## 5. Concrete example

```
results/conti-binary-K2/
├── config.json
├── data/{seed-9000.npz, …, seed-9004.npz}
├── reference/{full-info.npz, best-true-means.npz, best-arm-nocap.npz}
├── rosenbaum.csv
├── R-OW/Capped/gtrue-3/{meta.json, policy.npz, extended.npz, outcomes.csv, figures/…}
├── R-OW/Uncapped/gtrue-3/…
├── IPW/Capped/gtrue-5/…
└── Kallus/gtrue-3/…                          ← no Capped/Uncapped (parametric)
```
`outcomes.csv` in `R-OW/Capped/gtrue-3/` has 5 seeds × |Γ-grid| × 2 extensions rows; `policy.npz` has a
`pi__G<Γ>__seed<s>` (+ duals) for every (Γ, seed); `extended.npz` the Shapley/KNN deployment of each.

---

## 6. Per-experiment checklist (a run is "done" only when all are on disk)

- [ ] `config.json` with full `dgp_params` + `protocol`
- [ ] `data/seed-*.npz` for every seed (train + test draws)
- [ ] per (method, cap, γ_true): `meta.json`, `policy.npz` (pi + duals/θ for every Γ, seed),
      `extended.npz` (Shapley+KNN on train/test/grid), `outcomes.csv` (per-rep rows) + `outcomes-summary.csv`
- [ ] `reference/` ceilings (full-info, best-true-means, best-arm-nocap)
- [ ] `rosenbaum.csv` (per γ_true + pooled)
- [ ] every `figures/*.png` has a same-stem data file
