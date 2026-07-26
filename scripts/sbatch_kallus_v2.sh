#!/bin/bash
#SBATCH --job-name=kallus_v2
#SBATCH --partition=debug
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/home1/haghim/kallus_v2_%j.out
# Kallus & Zhou baseline for BOTH v2 showcase DGPs. Pure numpy (Gurobi monkeypatched to no-op)
# -> safe to run IN PARALLEL with the solver LP chain; the WLS 2-session limit is untouched.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/exp_owgap_alpha/run_kallus.py \
  --dgp assets/exp_owgap_v2/dgp.py \
  --out assets/exp_owgap_v2/kallus_v2.json \
  --n 600 --seeds 20 --gammas 1,1.5,2,2.5,3,4,5,6,8 \
&& echo "KALLUS_V2_DISCRETE_DONE $(date)"
python3 assets/exp_owgap_alpha/run_kallus.py \
  --dgp assets/exp_owgap_v2_cont/dgp.py \
  --out assets/exp_owgap_v2_cont/kallus_v2_cont.json \
  --n 400 --seeds 20 --gammas 1,1.5,2,2.5,3,4,5,6,8 \
&& echo "KALLUS_V2_CONT_DONE $(date)"
echo "SBATCH END $(date)"
