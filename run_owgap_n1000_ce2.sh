#!/usr/bin/env bash
# exp_owgap, N=1000, 4 seeds, c_eps=2.0 ONLY, WORKERS=2 (baseline, zero overage).
set -e
cd "/home1/haghim/code 1.1"
N=1000; SEEDS=4; WORKERS=2; THREADS="${THREADS:-1}"
BK="results_n1000_backup"; mkdir -p "$BK" "selected experiment/exp_owgap"
LOG="run_owgap_n1000_ce2.log"
echo "=== $(date +%F_%H:%M:%S) owgap N=1000 4-seed ce2.0 START on $(hostname) ===" | tee "$LOG"
OUT="assets/exp_owgap/owgap_results_n1000_ce2.0.json"
python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap/dgp.py --out "$OUT" \
  --n "$N" --seeds "$SEEDS" --gammas "1,1.5,2,2.5,3,4,5,6,8" --regimes uncap,cap \
  --ceps 2.0 --workers "$WORKERS" --threads "$THREADS" \
  2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license" | tee -a "$LOG"
python3 -c "import json; d=json.load(open('$OUT')); assert set(d['regimes'])>= {'uncap','cap'}"
cp -f "$OUT" "$BK/"; cp -f "$OUT" "selected experiment/exp_owgap/"
echo "=== $(date +%H:%M:%S) ce2.0 DONE (saved + backed up) ===" | tee -a "$LOG"
echo "OWGAP_N1000_CE2_DONE" | tee -a "$LOG"
