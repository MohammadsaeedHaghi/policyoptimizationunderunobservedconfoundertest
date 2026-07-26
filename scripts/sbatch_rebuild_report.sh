#!/bin/bash
#SBATCH --job-name=rebuild_report
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home1/haghim/rebuild_report_%j.out
# Rebuilds "selected experiment/report.html" after the alpha-sweep + 20-seed jobs finish, so the
# new tabs pick up the landed result JSONs without a manual step. No Gurobi involved.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1/selected experiment"
python3 build_report.py && echo "REPORT_REBUILD_DONE $(date)"
echo "SBATCH END $(date)"
