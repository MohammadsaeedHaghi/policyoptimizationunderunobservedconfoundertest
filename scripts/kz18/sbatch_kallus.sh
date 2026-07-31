#!/bin/bash
#SBATCH --job-name=kz_kallus
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/haghim/kz18_kallus_%j.out
# Kallus-Zhou (Management Science; par.nsf.gov/servlets/purl/10168529) synthetic campaign.
# One job per experiment arm; each runs its 5 seeds in parallel on the CARC token license.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/exp_owgap_alpha/run_kallus.py --dgp assets/exp_kz18/dgp.py --out assets/exp_kz18/kz_kallus.json --n 400 --seeds 5 --gammas 1,2,3,4.4817,6,8 --eval draws --n-test 4000 \
  && echo "KZ18_kallus_DONE $(date)"
echo "SBATCH END $(date)"
