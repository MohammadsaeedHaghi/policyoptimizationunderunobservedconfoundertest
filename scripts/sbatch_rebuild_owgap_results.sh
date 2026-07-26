#!/bin/bash
#SBATCH --job-name=owgap_rebuild
#SBATCH --partition=debug
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:20:00
#SBATCH --output=/home1/haghim/owgap_rebuild_%j.out
# Rebuild "html result/owgap_results.html" after the Shapley reruns land (chain
# afterany the diab_shp job). Pure python, no Gurobi. The builder auto-prefers
# the *_shapley.json files. Artifact republish happens in the next Claude session.
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
cd "/home1/haghim/code 1.1"
python3 "html result/build_results.py" && echo "OWGAP_REPORT_REBUILD_DONE $(date)"
echo "SBATCH END $(date)"
