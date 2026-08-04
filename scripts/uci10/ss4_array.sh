#!/bin/bash
#SBATCH --job-name=ss4
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-159%100
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1/semisynthetic"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
DATA=(wine_quality german_credit credit_default adult)
GAMMAS=(0.0 1.0 1.5 2.0)
i=$SLURM_ARRAY_TASK_ID
D=${DATA[$((i / 40))]}          # 40 cells per dataset
r=$((i % 40))
G=${GAMMAS[$((r / 10))]}
SEED=$((r % 10))
echo "=== $D gamma=$G seed=$SEED task=$i host=$(hostname) $(date) ==="
python3 run_semisynth.py --data "$D" --gamma "$G" --seed "$SEED" \
        --out "results/${D}_g${G}_s${SEED}.json"
echo "=== DONE rc=$? $(date) ==="
