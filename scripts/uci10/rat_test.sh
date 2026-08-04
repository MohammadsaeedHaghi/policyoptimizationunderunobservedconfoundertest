#!/bin/bash
#SBATCH --job-name=rattest
#SBATCH --account=vayanou_651
#SBATCH --partition=debug
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
python3 semisynthetic/test_rationality.py
echo "=== RATTEST DONE rc=$? $(date) ==="
