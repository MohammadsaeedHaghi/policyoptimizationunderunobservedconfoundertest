#!/bin/bash
#SBATCH --job-name=msmb_coupled
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/haghim/msmb_coupled_%j.out
# msmbench-coupled: the KMZ'19 benchmark with the disclosed X-U coupling knob
# U ~ Bern(sigma(BETA x)), BETA in {2.5, 5, 10} -> corr(x,S) = 0.58 / 0.76 / 0.84.
# Gamma fixed at Gamma* = 4.95 (known by construction), n=200 quick protocol.
# Declared prediction: O-W margin over O-X grows with BETA (the alpha-sweep argument
# on the literature's own DGP family; BETA=0 = the original benchmark, already run).
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
for B in 2.5 5 coupled; do
  DGP="assets/exp_msmbench/dgp_beta$B.py"
  OUT="assets/exp_msmbench/msmbench_beta$B.json"
  if [ "$B" = "coupled" ]; then DGP="assets/exp_msmbench/dgp_coupled.py"; OUT="assets/exp_msmbench/msmbench_beta10.json"; fi
  python3 assets/run_owgap_lip_gamma_2d.py \
    --dgp "$DGP" --out "$OUT" \
    --n 200 --n-test 4000 --seeds 3 --ceps 1.0 --workers 2 --deploy shapley --gammas 4.95 \
  && echo "MSMB_COUPLED_B${B}_DONE $(date)"
done
echo "MSMB_COUPLED_ALL_DONE $(date)"
echo "SBATCH END $(date)"
