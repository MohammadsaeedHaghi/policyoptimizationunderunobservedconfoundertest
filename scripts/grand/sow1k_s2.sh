#!/bin/bash
#SBATCH --job-name=sow1k2
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=4:00:00
#SBATCH --output=/home1/haghim/sow1k_s2_%j.out
echo "START $(date) on $(hostname)"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/grand/run_sow_n1000.py --seed 2 --out assets/grand/sow1k_s2.json --n 1000 --n-test 4000 \
  && echo "SOW1K_2_DONE $(date)"
echo "END $(date)"
