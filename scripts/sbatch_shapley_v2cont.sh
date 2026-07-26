#!/bin/bash
#SBATCH --job-name=owgap_v2c_shp
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --output=/home1/haghim/owgap_v2cont_shapley_%j.out
# v2 continuous rerun with SHAPLEY deployment (replaces KNN k=50): same LP solves,
# off-support extension now the closed-form Lipschitz min-max operator (exact at support,
# no smoothing floor -> L=inf shows its true oscillation). Also saves raw seed-0 support
# policies (policies_support_seed0). WORKERS=2 THREADS=1 -- only crash-free Gurobi WLS config.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_owgap_v2_cont/dgp.py" \
  --out "assets/exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_shapley.json" \
  --n 400 --n-test 4000 --seeds 8 --ceps 1.0 --workers 2 --deploy shapley \
&& echo "OWGAP_V2CONT_SHAPLEY_DONE $(date)"
echo "SBATCH END $(date)"
