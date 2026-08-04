#!/bin/bash
#SBATCH --job-name=uci10
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --array=0-9%10

set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3

REPO="/home1/haghim/code 1.1"
cd "$REPO"
export TMPDIR=/scratch1/$USER/tmp
mkdir -p "$TMPDIR" "$REPO/assets/uci10/results" "$REPO/assets/uci10/logs"

NAMES=(adult bank_marketing credit_default online_shoppers mushroom \
       wine_quality spambase support2 aids_clinical communities_crime)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

echo "=== $NAME  task=$SLURM_ARRAY_TASK_ID  host=$(hostname)  $(date) ==="
python3 assets/uci10/run_uci.py --data "$NAME" \
        --out "assets/uci10/results/${NAME}.json" --seeds 10 --workers 10
echo "=== DONE $NAME rc=$? $(date) ==="
touch "assets/uci10/results/${NAME}.DONE"
