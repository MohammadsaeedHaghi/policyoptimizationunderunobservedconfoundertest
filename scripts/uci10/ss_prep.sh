#!/bin/bash
#SBATCH --job-name=ss_prep
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1/semisynthetic"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
python3 prepare_semisynth.py --dataset bank_marketing --gammas 0.0,1.0,1.5,2.0
echo "=== PREP DONE rc=$? $(date) ==="
python3 example_generate_and_check.py || true
