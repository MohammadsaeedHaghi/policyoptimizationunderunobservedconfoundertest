# Semi-synthetic UCI benchmarks from the risk-score paper, adapted for policy learning

Replicates the data-generation procedure of *"Learning Risk Scores Robust to Unobserved
Confounders"* (`../42350_MainPaper.pdf`, anonymous submission), and extends it with a synthetic
heterogeneous CATE so that **both** potential outcomes exist and off-policy learning can be
evaluated.

## What is theirs and what is ours

| piece | source |
|---|---|
| `X` real UCI covariates, standardised | theirs |
| `U = Y⁰ = ` the UCI label, relabelled to {−1, +1} | theirs (the label plays both roles) |
| `logit π⁰(x,u) = λᵀx + γu`, `λⱼ ~ U(−0.1, 0.1)`, **no clipping**, `T ~ Bern(1 − π⁰)` | theirs, Eq. (6) |
| dataset screen: CV logistic log-loss ∈ [0.35, ln 2] | theirs |
| `τ(x) = a·(σ(bᵀx − c) − ½)`, `Y¹ = Y⁰ + τ(x) + ε` | **ours** |

`π⁰` is the probability of **no** treatment. `T = 1` means treated.

### Why τ is centred

The specification this was written from used `τ(x) = a·σ(bᵀx − c)`, which is **strictly positive**
(`a > 0`, `σ ∈ (0,1)`). Under that form the true CATE never changes sign, so the optimal policy is
"treat everyone", the oracle value equals the all-treat value, and no policy learner can be
distinguished from a constant. Measured over `d ∈ {5,10,20,50}`: `P(τ < 0) = 0.0000` in every case.
On `bank_marketing` the uncentred form gives headroom of exactly **+0.00000**.

Subtracting ½ makes the sign flip on about half the population. Pass `--uncentred-tau` (CLI) or
`centre_tau=False` (API) to reproduce the degenerate version as a negative control;
`example_generate_and_check.py` asserts both behaviours.

### Γ is exact here

Because `π⁰` is never clipped, the odds ratio between the two hidden states is **exactly**
`e^{2γ}` at every `x` — verified to ~1e-13. So the matched MSM parameter is `Γ = e^{2γ}`, not the
`[e^γ, e^{2γ}]` interval the paper has to settle for.

| γ | 0.0 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|
| Γ = e^{2γ} | 1.000 | 7.389 | 20.086 | 54.598 |

## Files

| file | role |
|---|---|
| `semisynthetic_dgp.py` | `load_uci_dataset`, `generate_semisynthetic`, `run_replications`, CLI |
| `screen_datasets.py` | applies the paper's log-loss screen to 15 UCI candidates → `screen.json` |
| `example_generate_and_check.py` | every invariant from the spec, plus the negative control |
| `prepare_semisynth.py` | population arrays per γ → `prepared/*_pop.npz` |
| `run_semisynth.py` | one (γ, seed) cell: all methods × c_ε × L × rationality, matched Γ only |
| `patch_rationality.py` | adds the C4 constraint to 4 solvers (backups in `solver_backups/`) |
| `test_rationality.py` | C4 invariants: monotone, inert at Γ=1, Γ=1 identity preserved |
| `aggregate_semisynth.py`, `build_semisynth_report.py` | summary JSON and HTML |

## Reproducing

```bash
sbatch scripts/uci10/ss_prep.sh          # screen + prepare + invariant checks
sbatch scripts/uci10/ss_array.sh         # 40 jobs = 4 gamma x 10 seeds
python3 aggregate_semisynth.py
python3 build_semisynth_report.py
```

Standalone generation, matching the spec's CLI:

```bash
python3 semisynthetic_dgp.py --dataset bank_marketing --gamma 1.0 \
        --n-reps 3 --seed 42 --out-dir example_outputs
```

## Output fields

`D_obs` — what an estimator may see: `X (n,d)`, `T ∈ {0,1}`, `Y` real.

`D_oracle` — evaluation only, **never** passed to an estimator: `U`, `Y0`, `Y1`, `tau`, `pi0`,
`e_star` (= 1 − π⁰, the true propensity for treatment).

`metadata` — `gamma`, `Gamma_MSM = e^{2γ}`, `lambda_coefs`, `tau_params (a, b, c)`,
`noise_sigma`, `centre_tau`, `seed`, `dataset_name`, `n_train`, `n_test`, plus a `diagnostics`
block carrying the measured odds-ratio error, propensity range, `P(T=1)`, `frac_tau_pos`,
`corr(x,U)`, and the oracle / never / all-treat values with their implied `headroom`.

`generate_semisynthetic` asserts before returning: `U == Y0`, both ⊂ {−1,+1}, `Y == Y0` where
`T = 0`, `Y == Y1` where `T = 1`, and that no oracle field appears in `D_obs`.

## Two deviations from the paper, both deliberate

1. **n_train = 300, n_test = 4000** (they use 1500/500). Our O-W solvers carry n² transport
   variables per arm; n = 1500 is 4.5M variables against the 180k we have actually run. A larger
   test set only sharpens evaluation, since both potential outcomes are known for every row.
2. **The solver's covariate is the scalar index `bᵀx`,** because the Lipschitz block in our LPs is
   exact only in one dimension. Since `τ` depends on `x` only through `bᵀx`, that index is a
   *sufficient statistic for the optimal policy* — the projection loses nothing about which
   decision is correct. Note the assignment depends on `λᵀx`, which is **not** a function of
   `bᵀx`, so the nominal propensity estimated from the index carries extra unexplained variation
   and the operative Γ can exceed the declared `e^{2γ}`.
