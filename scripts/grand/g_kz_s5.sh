#!/bin/bash
#SBATCH --job-name=kzs5
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6 --mem=16G --time=8:00:00
#SBATCH --output=/home1/haghim/grand_kzs5_%j.out
echo "START $(date) on $(hostname) job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/run_grid_seed.py --dgp assets/exp_kz18/dgp.py --out assets/grand/kz_s5.json --seed 5 --n 200 --n-test 4000 --gammas 1,1.6487,2.7183,4.4817,7.3891,12,16,24,32,40 --ceps 0.5,1.0,2.0 --caps none,0.3,0.4,0.5 --workers 6 --deploy shapley \
  && echo "GRAND_kzs5_DONE $(date)"
echo "END $(date)"
