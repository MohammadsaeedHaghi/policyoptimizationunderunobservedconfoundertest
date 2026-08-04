#!/bin/bash
#SBATCH --job-name=hidim
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=2:00:00
#SBATCH --output=/home1/haghim/hidim_%j.out
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 assets/exp_hidim/pilot.py && echo HIDIM_DONE
