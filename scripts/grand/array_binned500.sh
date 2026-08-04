#!/bin/bash
#SBATCH --job-name=b500
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --time=4:00:00
#SBATCH --array=0-9%10
#SBATCH --output=/home1/haghim/b500_%A_%a.out
# Does the binned-class win over sharp on Kallus-Mao-Zhou SURVIVE at n=500?
# At n=500 sharp reaches 1.249 (measured, 10 seeds) vs free-pi O-W's 1.187.
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
export BINNED_ONLY=km
python3 assets/grand/run_binned200.py --task-id $SLURM_ARRAY_TASK_ID --n 500 --workers 4 \
  --outdir assets/binned500 && echo "B500_${SLURM_ARRAY_TASK_ID}_DONE"
