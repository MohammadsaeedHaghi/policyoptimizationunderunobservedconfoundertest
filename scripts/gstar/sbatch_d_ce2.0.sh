#!/bin/bash
#SBATCH --job-name=d_ce2.0
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/home1/haghim/gstar_d_ce2.0_%j.out
# exp_gstar campaign wave job (cluster token license -- no WLS, no chaining).
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/run_experiment_parallel.py --dgp assets/exp_gstar/dgp.py --out assets/exp_gstar/gstar_ce2.0.json --n 600 --seeds 5 --gammas 1,1.5,2,2.5,3,4,5,6,8 --regimes uncap,cap --ceps 2.0 --workers 10 --threads 1 \
  && echo "GSTAR_D_CE2.0_DONE $(date)"
echo "SBATCH END $(date)"
