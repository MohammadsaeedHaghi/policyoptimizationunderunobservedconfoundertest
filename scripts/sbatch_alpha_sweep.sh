#!/bin/bash
#SBATCH --job-name=owgap_alpha
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/haghim/owgap_alpha_%j.out
# ALPHA coupling-strength sweep: exp_owgap DGP with P(S=+1|X)=sigma(alpha*X), alpha in {1,2,4,6,10}.
# N=600, 8 seeds, ce=1.0, both regimes, 9 gammas. WORKERS=2 THREADS=1 -- the ONLY crash-free
# Gurobi WLS config (baseline = 2 sessions, account-wide; overage is cumulative server-side).
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
for a in 1 2 4 6 10; do
  python3 assets/run_experiment_parallel.py \
    --dgp "assets/exp_owgap_alpha/dgp_a${a}.py" \
    --out "assets/exp_owgap_alpha/owgap_alpha${a}_ce1.0.json" \
    --n 600 --seeds 8 --gammas 1,1.5,2,2.5,3,4,5,6,8 --regimes uncap,cap --ceps 1.0 \
    --workers 2 --threads 1 \
  && cp "assets/exp_owgap_alpha/owgap_alpha${a}_ce1.0.json" "selected experiment/exp_owgap_alpha/" \
  && echo "ALPHA_${a}_DONE $(date)"
done
echo "OWGAP_ALPHA_ALL_DONE $(date)"
echo "SBATCH END $(date)"
