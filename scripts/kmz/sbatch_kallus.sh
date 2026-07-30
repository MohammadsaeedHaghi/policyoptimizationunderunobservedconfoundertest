#!/bin/bash
#SBATCH --job-name=kallus
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/haghim/kmz_kallus_%j.out
# KMZ'19 (arXiv 1810.02894) DGP campaign job -- cluster token license, parallel wave.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/exp_owgap_alpha/run_kallus.py --dgp assets/exp_msmbench/dgp_g15.py --out assets/exp_msmbench/kmz_kallus.json --n 400 --seeds 5 --gammas 1,2,3,4.4817,6,8 --eval draws --n-test 4000 \
  && echo "KMZ_KALLUS_DONE $(date)"
echo "SBATCH END $(date)"
