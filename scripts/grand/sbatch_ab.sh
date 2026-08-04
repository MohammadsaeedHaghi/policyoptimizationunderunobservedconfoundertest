#!/bin/bash
#SBATCH --job-name=ab_w
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=16G --time=3:00:00
#SBATCH --output=/home1/haghim/grand_ab_%j.out
# A/B: MSM box centered on Hajek-normalised w-hat (current) vs raw 1/e-hat (what Kallus-Zhou
# and our sharp baseline use). Paired on seed. Submitted only AFTER the main wave lands.
echo "START $(date) on $(hostname) job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cd "/home1/haghim/code 1.1"
python3 assets/grand/run_ab_weights.py --seeds 5 --n 200 --n-test 4000 --L 3 --workers 10 \
  && echo "GRAND_AB_DONE $(date)"
echo "END $(date)"
