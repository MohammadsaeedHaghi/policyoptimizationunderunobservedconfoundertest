#!/bin/bash
#SBATCH --job-name=a4_uncap
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=5 --mem=16G --time=6:00:00
#SBATCH --output=/home1/haghim/a4_uncap_%j.out
# gstar CONTINUOUS at ALPHA=4 -- uncap. Split from the chained job so the two regimes run
# CONCURRENTLY: wall time is now max(uncap, cap) rather than their sum.
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py --dgp assets/exp_gstar/dgp_a4_cont.py \
  --out assets/exp_gstar/gstar_a4_cont_uncap_ce1.0.json --n 400 --n-test 4000 --seeds 5 \
  --gammas 1,2,3,4,5,6,8 --ceps 1.0  --workers 5 --deploy shapley \
  && echo "A4_uncap_DONE $(date)"
