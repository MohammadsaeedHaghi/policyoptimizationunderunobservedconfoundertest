#!/bin/bash
#SBATCH --job-name=v2c_shp_ceps
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/home1/haghim/v2cont_shapley_ceps_%j.out
# v2 continuous L x Gamma sweep at the two remaining transport budgets c_eps=1.5, 2.0
# (c_eps=1.0 landed as owgap_v2_lip_gamma_2d_shapley.json). 8 seeds, Shapley deployment,
# save-everything runner. Completes the 3-epsilon ablation to match the discrete suite.
# WORKERS=2 THREADS=1 -- only crash-free Gurobi WLS config.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_owgap_v2_cont/dgp.py" \
  --out "assets/exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_shapley_ce1.5.json" \
  --n 400 --n-test 4000 --seeds 8 --ceps 1.5 --workers 2 --deploy shapley \
&& echo "V2CONT_SHAPLEY_CE15_DONE $(date)"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_owgap_v2_cont/dgp.py" \
  --out "assets/exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_shapley_ce2.0.json" \
  --n 400 --n-test 4000 --seeds 8 --ceps 2.0 --workers 2 --deploy shapley \
&& echo "V2CONT_SHAPLEY_CE20_DONE $(date)"
echo "V2CONT_CEPS_ALL_DONE $(date)"
echo "SBATCH END $(date)"
