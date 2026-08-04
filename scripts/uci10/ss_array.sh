#!/bin/bash
#SBATCH --job-name=ss_run
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-39%40
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1/semisynthetic"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
GAMMAS=(0.0 1.0 1.5 2.0)
i=$SLURM_ARRAY_TASK_ID
G=${GAMMAS[$((i / 10))]}
SEED=$((i % 10))
echo "=== bank_marketing gamma=$G seed=$SEED task=$i host=$(hostname) $(date) ==="
python3 run_semisynth.py --data bank_marketing --gamma "$G" --seed "$SEED" \
        --out "results/bank_marketing_g${G}_s${SEED}.json"
echo "=== DONE rc=$? $(date) ==="
