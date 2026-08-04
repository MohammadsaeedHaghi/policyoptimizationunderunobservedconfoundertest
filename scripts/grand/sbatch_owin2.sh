#!/bin/bash
#SBATCH --job-name=owin2
#SBATCH --partition=main
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=3:00:00
#SBATCH --output=/home1/haghim/owin2_%j.out
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
echo "### v2 (D0=8, signal/noise 0.18)"
python3 assets/exp_owin/pilot.py --dgp assets/exp_owin/dgp.py --n 1000 --seeds 3
echo "### low (D0=2.5, signal/noise 0.43)"
python3 assets/exp_owin/pilot.py --dgp assets/exp_owin/dgp_low.py --n 1000 --seeds 3
echo OWIN2_DONE
