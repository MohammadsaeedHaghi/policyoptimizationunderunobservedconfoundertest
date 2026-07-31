#!/bin/bash
#SBATCH --job-name=gss0
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6 --mem=16G --time=8:00:00
#SBATCH --output=/home1/haghim/grand_gss0_%j.out
echo "START $(date) on $(hostname) job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/run_grid_seed.py --dgp assets/exp_gstar/dgp_cont.py --out assets/grand/gs_s0.json --seed 0 --n 200 --n-test 4000 --gammas 1,2,3,5,8,12,16,24,32 --ceps 0.5,1.0,2.0 --caps none,0.3,0.4,0.5 --workers 6 --deploy shapley \
  && echo "GRAND_gss0_DONE $(date)"
echo "END $(date)"
