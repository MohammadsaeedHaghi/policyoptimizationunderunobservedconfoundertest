#!/bin/bash
#SBATCH --job-name=n2000_polopt
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/haghim/n2000_sbatch_%j.out
# Standalone batch job: N=2000, 5-seed reruns of exp_owgap + exp_nmcap.
# Runs independent of any interactive/Jupyter session, up to 12h (partition max is 2 days).
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
THREADS=3 bash "/home1/haghim/code 1.1/run_n2000_5seed.sh"
echo "SBATCH END $(date)"
