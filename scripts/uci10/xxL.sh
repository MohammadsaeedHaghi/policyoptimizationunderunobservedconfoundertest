#!/bin/bash
#SBATCH --job-name=xxL
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
echo "=== KMZ $(date) ==="
python3 assets/grand/run_xx_L.py --dgp assets/exp_msmbench/dgp.py --n 400 --seeds 5 \
        --out assets/grand/xxL_kmz.json
echo "=== GSTAR-cont $(date) ==="
python3 assets/grand/run_xx_L.py --dgp assets/exp_gstar/dgp_cont.py --n 400 --seeds 5 \
        --out assets/grand/xxL_gstar_cont.json
echo "=== XXL DONE $(date) ==="
