#!/bin/bash
#SBATCH --job-name=kmLg10
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=16G --time=5:00:00
#SBATCH --output=/home1/haghim/grand_kmLg10_%j.out
echo "START $(date) on $(hostname) job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py --dgp assets/exp_msmbench/dgp_g10.py --out assets/grand/km_lad_g10.json --n 200 --n-test 4000 --seeds 10 --gammas 1,1.6487,2.7183,4.4817,7.3891,12,16,24,32 --ceps 1.0 --workers 10 --deploy shapley \
  && echo "GRAND_kmLg10_DONE $(date)"
echo "END $(date)"
