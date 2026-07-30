#!/bin/bash
#SBATCH --job-name=d_kallus
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/haghim/gstar_d_kallus_%j.out
# exp_gstar campaign wave job (cluster token license -- no WLS, no chaining).
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/exp_owgap_alpha/run_kallus.py --dgp assets/exp_gstar/dgp.py --out assets/exp_gstar/gstar_kallus.json --n 600 --seeds 5 --gammas 1,1.5,2,2.5,3,4,5,6,8 \
  && echo "GSTAR_D_KALLUS_DONE $(date)"
echo "SBATCH END $(date)"
