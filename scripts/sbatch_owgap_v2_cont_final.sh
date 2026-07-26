#!/bin/bash
#SBATCH --job-name=owgap_v2c_fin
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/home1/haghim/owgap_v2cont_final_%j.out
# exp_owgap_v2_cont FINAL: 8 seeds, larger test set (pilot at 3 seeds confirmed the design;
# refs like the oracle carry ~0.1 test noise at 3 seeds x 2000, so tighten for the paper).
# WORKERS=2 THREADS=1 -- only crash-free Gurobi WLS config. Chain after the discrete 20-seed run.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_owgap_v2_cont/dgp.py" \
  --out "assets/exp_owgap_v2_cont/owgap_v2_lip_gamma_2d_final.json" \
  --n 400 --n-test 4000 --seeds 8 --k 50 --ceps 1.0 --workers 2 \
&& echo "OWGAP_V2CONT_FINAL_DONE $(date)"
echo "SBATCH END $(date)"
