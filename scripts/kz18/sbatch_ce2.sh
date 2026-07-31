#!/bin/bash
#SBATCH --job-name=kz_ce2
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/home1/haghim/kz18_ce2_%j.out
# Kallus-Zhou (Management Science; par.nsf.gov/servlets/purl/10168529) synthetic campaign.
# One job per experiment arm; each runs its 5 seeds in parallel on the CARC token license.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py --dgp assets/exp_kz18/dgp.py --out assets/exp_kz18/kz_main_ce2.0.json --n 200 --ceps 2.0 --gammas 1,2,3,4.4817,6,8 --seeds 5 --n-test 4000 --workers 5 --deploy shapley \
  && echo "KZ18_ce2_DONE $(date)"
echo "SBATCH END $(date)"
