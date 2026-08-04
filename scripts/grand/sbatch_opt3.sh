#!/bin/bash
#SBATCH --job-name=opt3
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=3:00:00
#SBATCH --output=/home1/haghim/opt3_%j.out
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 assets/exp_hidim/pilot_opt3.py && echo OPT3_DONE
