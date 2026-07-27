#!/bin/bash
#SBATCH --job-name=msmb_pilot
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/home1/haghim/msmb_pilot_%j.out
# exp_msmbench pilot: the community-standard KMZ'19 MSM synthetic (as in Hess/Frauen ICLR'26),
# Gamma*=4.95 -> matched Gamma=5. The corr(X,S)=0 anchor of the diagnostic map: declared
# prediction is O-W ~ O-X here (U independent of X leaves the W constraint nothing to grab)
# while box-only performs well on its exactly-specified home turf.
# QUICK pilot: n=200 + FIXED Gamma = Gamma* = 4.95 (known by construction; no sweep needed).
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 assets/run_owgap_lip_gamma_2d.py \
  --dgp "assets/exp_msmbench/dgp.py" \
  --out "assets/exp_msmbench/msmbench_lip_gamma_2d_pilot.json" \
  --n 200 --n-test 4000 --seeds 3 --ceps 1.0 --workers 2 --deploy shapley --gammas 4.95 \
&& echo "MSMBENCH_PILOT_DONE $(date)"
echo "SBATCH END $(date)"
