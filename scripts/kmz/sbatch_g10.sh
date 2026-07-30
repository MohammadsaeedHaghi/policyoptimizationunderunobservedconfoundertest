#!/bin/bash
#SBATCH --job-name=g10
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/home1/haghim/kmz_g10_%j.out
# KMZ'19 (arXiv 1810.02894) DGP campaign job -- cluster token license, parallel wave.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py --dgp assets/exp_msmbench/dgp_g10.py --out assets/exp_msmbench/kmz_g10.json --n 400 --n-test 4000 --seeds 5 --gammas 2.7183 --ceps 1.0 --workers 5 --deploy shapley \
  && echo "KMZ_G10_DONE $(date)"
echo "SBATCH END $(date)"
