#!/bin/bash
#SBATCH --job-name=owgap_n1000
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home1/haghim/owgap_n1000_%j.out
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
THREADS=1 bash "/home1/haghim/code 1.1/run_owgap_n1000_2w.sh"
echo "SBATCH END $(date)"
