#!/bin/bash
#SBATCH --job-name=owgap_v2_capr
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/home1/haghim/owgap_v2_capr_%j.out
# exp_owgap_v2 cap-robustness: capped regime ONLY at budget 40% and 50% (base run has 30%).
# Preempts the reviewer objection "the 30% cap was chosen to flatter O-W". 8 seeds, ce=1.0.
# WORKERS=2 THREADS=1 -- only crash-free Gurobi WLS config. Chain after the continuous final.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
for C in 40 50; do
  python3 assets/run_experiment_parallel.py \
    --dgp "assets/exp_owgap_v2/dgp_cap${C}.py" \
    --out "assets/exp_owgap_v2/owgap_v2_cap${C}_ce1.0.json" \
    --n 600 --seeds 8 --gammas 1,1.5,2,2.5,3,4,5,6,8 --regimes cap \
    --ceps 1.0 --workers 2 --threads 1 \
  && echo "OWGAP_V2_CAP${C}_DONE $(date)"
done
echo "OWGAP_V2_CAPROBUST_ALL_DONE $(date)"
echo "SBATCH END $(date)"
