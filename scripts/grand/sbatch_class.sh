#!/bin/bash
#SBATCH --job-name=classt
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=2:00:00
#SBATCH --output=/home1/haghim/classt_%j.out
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 assets/grand/classtest.py && echo CLASSTEST_DONE
