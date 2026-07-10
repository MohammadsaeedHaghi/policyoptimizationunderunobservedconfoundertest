#!/usr/bin/env bash
# exp_owgap_x3: 3-value X (-1,0,1), N_train=200, 8 seeds, 3 epsilons. WORKERS=2 (license baseline).
set -e
cd "/home1/haghim/code 1.1"
N=200; SEEDS=8; WORKERS=2; THREADS=1
LOG="run_owgap_x3.log"
echo "=== $(date +%H:%M:%S) owgap_x3 (X in {-1,0,1}) N=200 8-seed START ===" | tee "$LOG"
for CE in 1.0 1.5 2.0; do
  OUT="assets/exp_owgap_x3/owgap_x3_ce${CE}.json"
  echo "=== $(date +%H:%M:%S) ce=$CE ===" | tee -a "$LOG"
  python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap_x3/dgp.py --out "$OUT" \
    --n $N --seeds $SEEDS --gammas "1,1.5,2,2.5,3,4,5,6,8" --regimes uncap,cap \
    --ceps $CE --workers $WORKERS --threads $THREADS \
    2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license" | tee -a "$LOG"
  python3 -c "import json;d=json.load(open('$OUT'));assert set(d['regimes'])>={'uncap','cap'}"
done
echo "OWGAP_X3_ALL_DONE" | tee -a "$LOG"
