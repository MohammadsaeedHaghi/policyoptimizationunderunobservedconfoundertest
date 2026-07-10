# Candidate DGP frameworks for the "O-W beats everyone" study

Ten DISTINCT unobserved-confounder data-generating frameworks (discrete X, binary treatment, K=2),
each a self-contained module with the standard repo interface
(`LEVELS, CAP, K, generate(n,seed), grid_truth(), exact_value(pi_grid), oracle_policy(X)`), plus a
`naive_policy_grid()` diagnostic. Any of them can be driven directly by `assets/run_experiment_parallel.py`:

```
python3 assets/run_experiment_parallel.py --dgp assets/dgp_ideas/dgp03_strong_proxy.py \
    --out assets/dgp_ideas/_out_dgp03.json --n 700 --seeds 5 \
    --gammas 1,1.5,2,2.5,3,4,5,6,8 --regimes uncap,cap --ceps 1.0 --workers 8 --threads 1
```

## The ideas (each places the hidden confounder differently)

| # | module | where the confounder lives | one-line idea |
|---|--------|----------------------------|----------------|
| 01 | `dgp01_binaryS_selection` | selection + both outcomes (binary S) | I3 baseline: vitality type S boosts both arms, helps treated slightly more |
| 02 | `dgp02_gaussianU_outcome` | OUTCOME (continuous U, Kallus-style) | unrecorded biomarker U raises both arms, drives T; U correlated with X |
| 03 | `dgp03_strong_proxy` | OUTCOME, X is near-perfect proxy of U | tight U\|X -> balancing X almost fully balances U (O-W at its strongest) |
| 04 | `dgp04_signflip_indication` | effect-modification SIGN FLIP | confounding by indication / Simpson reversal: treatment helps robust, harms frail |
| 05 | `dgp05_multiplicative_effect` | MULTIPLICATIVE on the effect | hidden vitality U scales the benefit: effect = (a+bX)*U |
| 06 | `dgp06_saturation_doseresponse` | latent index, SATURATING response | logistic dose-response + treatment cost; treat only on the steep part |
| 07 | `dgp07_latent_class_mixture` | latent CLASS (responder/non-responder) | hidden subgroup determines benefit; responders also have better baseline |
| 08 | `dgp08_cancelling_baseline_CONTRAST` | baseline only, CANCELS in contrast | NEGATIVE CONTROL: U identical in both arms -> plain DR should suffice |
| 09 | `dgp09_two_confounders` | TWO correlated latents (U1 selects, U2 prognoses) | richer, realistic confounding; no single proxy explains assignment |
| 10 | `dgp10_interaction_treated_only` | TREATED arm only (clean control) | treatment efficacy depends on hidden fitness U; control arm is clean |

## Screening

`python3 assets/dgp_ideas/screen.py` prints, per idea (exact, noise-free, no Gurobi):
oracle / naive / gap / ctrl / all / oracle-treat-fraction / naive-vs-oracle agreement / raw confounding.

IMPORTANT: the screen is a ROUGH FILTER, not a verdict. It measures headroom over the raw confounded
contrast (what plain IPW chases). The known OW-winner `dgp01` screens with gap=0 because its O-W
advantage comes from worst-case robustness under the cap, which a no-solver screen cannot measure. Use
the screen only to confirm a DGP is non-degenerate (partial oracle, real confounding present); the true
O-W-vs-everyone ranking requires the actual solver runs (Gurobi) on the server.

## What to look for on the server

For the paper target (BOTH IPW-O-W and DR-O-W beat ALL methods incl. plain DR-X-X, peak at small Gamma
in BOTH capped and uncapped, visible large-Gamma collapse, 3 epsilon values), the structurally most
promising are the NON-CANCELLING frameworks where the hidden variable affects the arms DIFFERENTLY and
correlates with X: 02, 03, 09, 10 (and the 01 baseline). 04/05/06/07 are alternative mechanisms worth
testing. 08 is the deliberate contrast that should NOT favor O-W. Run the solver sweep over these, sweep
DGP params + 3 epsilons, and keep the cleanest large-gap, reproducible winner.
