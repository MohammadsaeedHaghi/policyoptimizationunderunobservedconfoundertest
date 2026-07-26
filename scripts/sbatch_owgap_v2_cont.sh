#!/bin/bash
#SBATCH --job-name=owgap_v2c
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/haghim/owgap_v2cont_%j.out
# exp_owgap_v2_cont: continuous-X showcase variant (B1=3, THETA=1), L x Gamma 2-D sweep.
# 5 robust methods x 6 gammas x 8 L values, N=400 train / 2000 test, 3 seeds, KNN deploy.
# Also records naive DR-X-X / oracle / never / all-treat refs + seed-0 policy curves.
# WORKERS=2 THREADS=1 -- the ONLY crash-free Gurobi WLS config. Chain after the v2 pilot.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_owgap_v2_cont/dgp.py" \
  --out "assets/exp_owgap_v2_cont/owgap_v2_lip_gamma_2d.json" \
  --n 400 --n-test 2000 --seeds 3 --k 50 --ceps 1.0 --workers 2 \
&& echo "OWGAP_V2CONT_2D_DONE $(date)"
echo "SBATCH END $(date)"
