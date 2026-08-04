#!/bin/bash
#SBATCH --job-name=g500
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=24G --time=3:00:00
#SBATCH --array=0-359%100
#SBATCH --output=/home1/haghim/g500_%A_%a.out
# N=500 full grid. One array task = one (campaign, seed, c_eps, cap) slice; every
# (method, Gamma, L) inside it is computed and stored separately. Throttled to 100
# concurrent tasks because MaxJobsPU=100 -- see run_task500.py for the sizing argument.
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 assets/grand/run_task500.py --task-id $SLURM_ARRAY_TASK_ID --n 500 --workers 16 \
  && echo "G500_${SLURM_ARRAY_TASK_ID}_DONE"
