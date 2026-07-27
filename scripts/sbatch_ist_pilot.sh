#!/bin/bash
#SBATCH --job-name=ist_pilot
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/home1/haghim/ist_pilot_%j.out
# IST recipe-D pilot (experiment R1): real RCT outcomes, injected Lambda=4.95 confounding.
# L x Gamma sweep, 3 seeds, Shapley deployment, HT evaluation on the untouched RCT test pool.
# WORKERS=2 THREADS=1 -- only crash-free Gurobi WLS config.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_ist/dgp.py" \
  --out "assets/exp_ist/ist_lip_gamma_2d_pilot.json" \
  --n 400 --n-test 8000 --seeds 3 --ceps 1.0 --workers 2 --deploy shapley \
&& echo "IST_PILOT_DONE $(date)"
echo "SBATCH END $(date)"
