#!/bin/bash
#SBATCH --job-name=diab_pilots
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/home1/haghim/diab_pilots_%j.out
# Diabetes-130 pilots, L x Gamma sweep (continuous X), 3 seeds each:
#  A) dgp_inregime.py -- real covariates + amplified confounding (matched Gamma=5, in-regime):
#     naive concludes "insulin harmful", treats nobody; O-W should recover ~oracle.
#  B) dgp.py -- fully real (X,S,T) (matched Gamma=1.26, corr=0.196, out-of-regime):
#     prediction = naive ~ oracle, robustness unnecessary; validates the scoping diagnostic.
# WORKERS=2 THREADS=1 -- only crash-free Gurobi WLS config. Chain after cap-robustness job.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_diabetes/dgp_inregime.py" \
  --out "assets/exp_diabetes/diab_inregime_lip_gamma_2d.json" \
  --n 400 --n-test 4000 --seeds 3 --k 50 --ceps 1.0 --workers 2 \
&& echo "DIAB_INREGIME_DONE $(date)"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_diabetes/dgp.py" \
  --out "assets/exp_diabetes/diab_real_lip_gamma_2d.json" \
  --n 400 --n-test 4000 --seeds 3 --k 50 --ceps 1.0 --workers 2 \
&& echo "DIAB_REAL_DONE $(date)"
echo "DIAB_PILOTS_ALL_DONE $(date)"
echo "SBATCH END $(date)"
