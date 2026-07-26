#!/bin/bash
#SBATCH --job-name=owgap_v2_20s
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/home1/haghim/owgap_v2_20seed_%j.out
# exp_owgap_v2 FINAL: 20 seeds, all three transport budgets, both regimes.
# Pilot (8 seeds) confirmed margins at matched Gamma=5: +0.155 uncap / +0.176 cap.
# WORKERS=2 THREADS=1 -- the ONLY crash-free Gurobi WLS config. Chain after the continuous sweep.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
for CE in 1.0 1.5 2.0; do
  python3 assets/run_experiment_parallel.py \
    --dgp "assets/exp_owgap_v2/dgp.py" \
    --out "assets/exp_owgap_v2/owgap_v2_20seed_ce${CE}.json" \
    --n 600 --seeds 20 --gammas 1,1.5,2,2.5,3,4,5,6,8 --regimes uncap,cap \
    --ceps "$CE" --workers 2 --threads 1 \
  && echo "OWGAP_V2_20SEED_CE${CE}_DONE $(date)"
done
echo "OWGAP_V2_20SEED_ALL_DONE $(date)"
echo "SBATCH END $(date)"
