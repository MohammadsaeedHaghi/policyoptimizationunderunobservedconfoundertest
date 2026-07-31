#!/bin/bash
#SBATCH --job-name=grand_smoke
#SBATCH --partition=debug
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=0:25:00
#SBATCH --output=/home1/haghim/grand_smoke_%j.out
echo "SMOKE START $(date)"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/run_grid_seed.py --dgp assets/exp_kz18/dgp.py \
  --out /tmp/grand_smoke.json --seed 0 --n 120 --n-test 500 \
  --gammas 1,8,32 --ceps 1.0 --caps none,0.3 --workers 2 \
  && echo "SMOKE_OK $(date)"
