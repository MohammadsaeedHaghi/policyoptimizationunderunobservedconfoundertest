#!/bin/bash
cd "/home1/haghim/code 1.1"
GAM="1,1.5,2,3,4,6,8,20,100,1000"
flt(){ grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license"; }
echo "[$(date +%T)] START obj sweeps"
python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap_x3/dgp.py \
  --out assets/exp_owgap_x3/owgap_x3_obj.json --n 200 --seeds 4 --gammas "$GAM" \
  --regimes uncap --ceps 1.0 --workers 2 --threads 1 2>&1 | flt | tail -1
echo "[$(date +%T)] X3 done"
python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap/dgp.py \
  --out assets/exp_owgap/owgap_obj.json --n 600 --seeds 4 --gammas "$GAM" \
  --regimes uncap --ceps 1.0 --workers 2 --threads 1 2>&1 | flt | tail -1
echo "[$(date +%T)] N=600 done"
python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap/dgp.py \
  --out assets/exp_owgap/owgap_n1000_obj.json --n 1000 --seeds 4 --gammas "$GAM" \
  --regimes uncap --ceps 1.0 --workers 2 --threads 1 2>&1 | flt | tail -1
echo "[$(date +%T)] N=1000 done -- ALL OBJ SWEEPS COMPLETE"
