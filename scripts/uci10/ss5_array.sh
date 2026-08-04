#!/bin/bash
#SBATCH --job-name=ss5
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-799%100
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1/semisynthetic"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
DATA=(bank_marketing wine_quality german_credit credit_default adult)
GAMMAS=(0.0 1.0 1.5 2.0)
i=$SLURM_ARRAY_TASK_ID
D=${DATA[$((i / 160))]}          # 160 cells per dataset = 4 dgp seeds x 4 gammas x 10 splits
r=$((i % 160))
DGP=$(( r / 40 + 1 ))            # dgp seeds 1..4 (0 already done)
q=$((r % 40))
G=${GAMMAS[$((q / 10))]}
SEED=$((q % 10))
OUT="results/${D}_g${G}_d${DGP}_s${SEED}.json"
if [ -s "$OUT" ]; then echo "already have $OUT"; exit 0; fi
echo "=== $D gamma=$G dgp=$DGP split=$SEED task=$i $(date) ==="
python3 run_semisynth.py --data "$D" --gamma "$G" --seed "$SEED" --dgp-seed "$DGP" --out "$OUT"
echo "=== DONE rc=$? $(date) ==="
