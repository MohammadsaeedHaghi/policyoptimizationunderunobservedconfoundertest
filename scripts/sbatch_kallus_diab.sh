#!/bin/bash
#SBATCH --job-name=kallus_diab
#SBATCH --partition=debug
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/home1/haghim/kallus_diab_%j.out
# Kallus & Zhou baseline for both Diabetes DGPs. Pure numpy (no Gurobi) -> runs in parallel
# with the solver chain; the WLS 2-session limit is untouched.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/exp_owgap_alpha/run_kallus.py \
  --dgp assets/exp_diabetes/dgp_inregime.py \
  --out assets/exp_diabetes/kallus_diab_inregime.json \
  --n 400 --seeds 20 --gammas 1,1.5,2,2.5,3,4,5,6,8 --eval draws --n-test 4000 \
&& echo "KALLUS_DIAB_INREGIME_DONE $(date)"
python3 assets/exp_owgap_alpha/run_kallus.py \
  --dgp assets/exp_diabetes/dgp.py \
  --out assets/exp_diabetes/kallus_diab_real.json \
  --n 400 --seeds 20 --gammas 1,1.5,2,2.5,3,4,5,6,8 --eval draws --n-test 4000 \
&& echo "KALLUS_DIAB_REAL_DONE $(date)"
echo "SBATCH END $(date)"
