#!/bin/bash
#SBATCH --job-name=c_cap30
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/home1/haghim/gstar_c_cap30_%j.out
# exp_gstar campaign wave job (cluster token license -- no WLS, no chaining).
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py --dgp assets/exp_gstar/dgp_cont.py --out assets/exp_gstar/gstar_cont_cap30_ce1.0.json --n 400 --n-test 4000 --seeds 5 --gammas 1,2,3,4,5,6,8 --ceps 1.0 --workers 5 --deploy shapley --cap \
  && echo "GSTAR_C_CAP30_DONE $(date)"
echo "SBATCH END $(date)"
