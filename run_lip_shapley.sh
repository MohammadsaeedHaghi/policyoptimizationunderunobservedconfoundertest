#!/bin/bash
cd "/home1/haghim/code 1.1"
echo "[$(date +%T)] X-X lipschitz (KNN+Shapley)..."
python3 assets/run_owgap_lipschitz.py --out assets/exp_owgap_cont/owgap_lipschitz.json --n 400 --n-test 2000 --seeds 5 --k 50 2>&1 | grep -vE "Set parameter|Read param|WLS|License|Academic" | tail -20
echo "[$(date +%T)] robust lipschitz (KNN+Shapley)..."
python3 assets/run_owgap_lipschitz_robust.py --out assets/exp_owgap_cont/owgap_lipschitz_robust.json --n 400 --n-test 2000 --seeds 5 --gamma 2 --k 50 --workers 2 2>&1 | grep -vE "Set parameter|Read param|WLS|License|Academic" | tail -18
echo "ALL_LIP_SHAPLEY_DONE"
