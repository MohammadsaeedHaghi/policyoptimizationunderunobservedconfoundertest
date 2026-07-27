#!/bin/bash
#SBATCH --job-name=coupled_kappa
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=/home1/haghim/coupled_kappa_%j.out
# Coupled synthetic, confounder-leverage dial KAPPA (U-term amplitude x KAPPA).
# Gamma fixed at Gamma* = 4.95 (known by construction), n=200 quick protocol.
# Declared prediction: the O-W gap over box-only opens as KAPPA makes the confounder
# dominate the outcome scale (owgap mechanism), at high coupling beta.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
for SPEC in k2.5b10 k4b10 k4b5; do
  python3 assets/run_owgap_lip_gamma_2d.py \
    --dgp "assets/exp_coupled_synth/dgp_$SPEC.py" \
    --out "assets/exp_coupled_synth/coupled_$SPEC.json" \
    --n 200 --n-test 4000 --seeds 3 --ceps 1.0 --workers 2 --deploy shapley --gammas 4.95 \
  && echo "COUPLED_${SPEC}_DONE $(date)"
done
echo "COUPLED_KAPPA_ALL_DONE $(date)"
echo "SBATCH END $(date)"
