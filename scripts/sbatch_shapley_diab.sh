#!/bin/bash
#SBATCH --job-name=diab_shp
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/haghim/diab_shapley_%j.out
# Diabetes A (in-regime) + B (fully real) rerun with SHAPLEY deployment (replaces KNN k=50).
# Chained after the v2-cont shapley job: Gurobi WLS baseline = 2 sessions account-wide.
# WORKERS=2 THREADS=1 -- only crash-free Gurobi WLS config.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_diabetes/dgp_inregime.py" \
  --out "assets/exp_diabetes/diab_inregime_lip_gamma_2d_shapley.json" \
  --n 400 --n-test 4000 --seeds 3 --ceps 1.0 --workers 2 --deploy shapley \
&& echo "DIAB_INREGIME_SHAPLEY_DONE $(date)"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_diabetes/dgp.py" \
  --out "assets/exp_diabetes/diab_real_lip_gamma_2d_shapley.json" \
  --n 400 --n-test 4000 --seeds 3 --ceps 1.0 --workers 2 --deploy shapley \
&& echo "DIAB_REAL_SHAPLEY_DONE $(date)"
echo "DIAB_SHAPLEY_ALL_DONE $(date)"
echo "SBATCH END $(date)"
