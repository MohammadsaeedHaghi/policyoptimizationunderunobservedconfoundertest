#!/bin/bash
#SBATCH --job-name=rct
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-29%30
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
REPO="/home1/haghim/code 1.1"; cd "$REPO/RCT datasets"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"

DATA=(ihdp twins ist)
i=$SLURM_ARRAY_TASK_ID
D=${DATA[$((i / 10))]}          # 10 cells per dataset
r=$((i % 10))
SEED=$((r / 2))                 # 5 seeds
CE=$((r % 2))                   # 2 c_eps
if [ $CE -eq 0 ]; then CEPS=1.0; else CEPS=2.0; fi

echo "=== $D seed=$SEED ceps=$CEPS task=$i host=$(hostname) $(date) ==="
python3 run_rct.py --data "$D" --seed "$SEED" --ceps "$CEPS" \
        --out "results/${D}_s${SEED}_ce${CEPS}.json"
echo "=== DONE rc=$? $(date) ==="
