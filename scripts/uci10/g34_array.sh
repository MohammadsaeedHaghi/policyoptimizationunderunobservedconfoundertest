#!/bin/bash
#SBATCH --job-name=g34
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-499%100
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1/semisynthetic"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
DATA=(bank_marketing wine_quality german_credit credit_default adult)
GAMMAS=(3.0 4.0)
i=$SLURM_ARRAY_TASK_ID
D=${DATA[$((i / 100))]}          # 100 cells per dataset = 5 dgp seeds x 2 gammas x 10 splits
r=$((i % 100))
DGP=$(( r / 20 ))                # dgp seeds 0..4
q=$((r % 20))
G=${GAMMAS[$((q / 10))]}
SEED=$((q % 10))
OUT="results/${D}_g${G}_d${DGP}_s${SEED}.json"
if [ -s "$OUT" ]; then echo "already have $OUT"; exit 0; fi
echo "=== $D gamma=$G dgp=$DGP split=$SEED task=$i $(date) ==="
python3 run_semisynth.py --data "$D" --gamma "$G" --seed "$SEED" --dgp-seed "$DGP" --out "$OUT"
echo "=== DONE rc=$? $(date) ==="
