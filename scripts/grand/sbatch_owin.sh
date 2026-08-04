#!/bin/bash
#SBATCH --job-name=owin
#SBATCH --partition=debug
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=0:55:00
#SBATCH --output=/home1/haghim/owin_%j.out
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 assets/exp_owin/pilot.py && echo OWIN_DONE
