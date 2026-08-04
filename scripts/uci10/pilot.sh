#!/bin/bash
#SBATCH --job-name=uci_pilot
#SBATCH --account=vayanou_651
#SBATCH --partition=debug
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
echo "=== pilot aids_clinical $(date) ==="
python3 assets/uci10/run_uci.py --data aids_clinical \
   --out assets/uci10/results/_pilot_aids.json --seeds 1 --workers 2
echo "=== PILOT DONE rc=$? $(date) ==="
