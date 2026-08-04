#!/bin/bash
#SBATCH --job-name=lin3
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=3 --mem=16G --time=4:00:00
#SBATCH --array=0-29%30
#SBATCH --output=/home1/haghim/lin3_%A_%a.out
# Linear (threshold, since these benchmarks are 1-D) policy class on all three benchmarks,
# against free-pi and sharp's 15-bin class. KMZ is the stress test: its oracle treats a
# non-interval set, which no single threshold can represent.
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 assets/grand/run_linear3.py --task-id $SLURM_ARRAY_TASK_ID --n 200 \
  && echo "LIN3_${SLURM_ARRAY_TASK_ID}_DONE"
