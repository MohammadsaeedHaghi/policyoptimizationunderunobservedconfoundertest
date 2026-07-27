#!/bin/bash
#SBATCH --job-name=s2_pilot
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/home1/haghim/s2_pilot_%j.out
# SUPPORT2 pilots (experiment R2), chained after the IST pilot (Gurobi 2-session licence):
#  B) dgp_b.py -- fully real (x, S=APS, T=DNR) at matched Lambda=2.20, synthetic outcomes;
#     PREDICTED-BOUNDARY test (coupling 0.46): expect a small, Gamma-stable O-W edge.
#  A) dgp_a.py -- amplified in-regime variant (synthetic S/T at Lambda=4.95) on real covariates.
# WORKERS=2 THREADS=1 -- only crash-free Gurobi WLS config.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_support2/dgp_b.py" \
  --out "assets/exp_support2/s2b_lip_gamma_2d_pilot.json" \
  --n 400 --n-test 6000 --seeds 3 --ceps 1.0 --workers 2 --deploy shapley \
&& echo "S2B_PILOT_DONE $(date)"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_support2/dgp_a.py" \
  --out "assets/exp_support2/s2a_lip_gamma_2d_pilot.json" \
  --n 400 --n-test 6000 --seeds 3 --ceps 1.0 --workers 2 --deploy shapley \
&& echo "S2A_PILOT_DONE $(date)"
echo "S2_PILOTS_ALL_DONE $(date)"
echo "SBATCH END $(date)"
