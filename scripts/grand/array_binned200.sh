#!/bin/bash
#SBATCH --job-name=b200
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --time=2:00:00
#SBATCH --array=0-29%30
#SBATCH --output=/home1/haghim/b200_%A_%a.out
# Policy-class isolation: O-W/O-X given sharp's 15 quantile bins vs the free-pi class
# every campaign has used. Uncertainty set untouched; only the policy is coarsened.
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 assets/grand/run_binned200.py --task-id $SLURM_ARRAY_TASK_ID --n 200 --workers 4 \
  && echo "B200_${SLURM_ARRAY_TASK_ID}_DONE"
