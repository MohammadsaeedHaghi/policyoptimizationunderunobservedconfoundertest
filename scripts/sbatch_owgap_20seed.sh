#!/bin/bash
#SBATCH --job-name=owgap_20seed
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=/home1/haghim/owgap_20seed_%j.out
# Full 20-seed rerun of the exp_owgap N=600 grid (was 5 seeds): all 3 epsilons, both regimes,
# 9 gammas. Submitted with --dependency=afterany:<alpha job> so it NEVER overlaps the alpha
# sweep on the Gurobi WLS license (baseline = 2 sessions account-wide). WORKERS=2 THREADS=1.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
for ce in 1.0 1.5 2.0; do
  python3 assets/run_experiment_parallel.py \
    --dgp assets/exp_owgap/dgp.py \
    --out "assets/exp_owgap/owgap_results_20seed_ce${ce}.json" \
    --n 600 --seeds 20 --gammas 1,1.5,2,2.5,3,4,5,6,8 --regimes uncap,cap --ceps ${ce} \
    --workers 2 --threads 1 \
  && cp "assets/exp_owgap/owgap_results_20seed_ce${ce}.json" "selected experiment/exp_owgap/" \
  && echo "OWGAP20_CE${ce}_DONE $(date)"
done
echo "OWGAP_20SEED_ALL_DONE $(date)"
echo "SBATCH END $(date)"
