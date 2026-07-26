#!/usr/bin/env bash
# exp_owgap ONLY, N_train=1000, 8 seeds, 3 epsilons. ~8x faster than N=2000 (LP solve ~O(n^3)).
# 8 seeds x 2 regimes = 16 jobs -> 16 workers = one wave per epsilon. LP is single-threaded so THREADS=1.
set -e
cd "/home1/haghim/code 1.1"
N=1000; SEEDS=8; WORKERS=8; THREADS="${THREADS:-1}"
BK="results_n1000_backup"; mkdir -p "$BK" "selected experiment/exp_owgap"
LOG="run_owgap_n1000.log"
echo "=== $(date +%F_%H:%M:%S) owgap N=1000 8-seed START on $(hostname) (workers=$WORKERS threads=$THREADS) ===" | tee "$LOG"
for CE in 1.0 1.5 2.0; do
  OUT="assets/exp_owgap/owgap_results_n1000_ce${CE}.json"
  echo "=== $(date +%H:%M:%S) owgap c_eps=${CE} -> ${OUT} ===" | tee -a "$LOG"
  python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap/dgp.py --out "$OUT" \
    --n "$N" --seeds "$SEEDS" --gammas "1,1.5,2,2.5,3,4,5,6,8" --regimes uncap,cap \
    --ceps "$CE" --workers "$WORKERS" --threads "$THREADS" \
    2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license" | tee -a "$LOG"
  cp -f "$OUT" "$BK/" 2>/dev/null
  cp -f "$OUT" "selected experiment/exp_owgap/" 2>/dev/null
  echo "=== $(date +%H:%M:%S) owgap c_eps=${CE} DONE (saved + backed up) ===" | tee -a "$LOG"
done
echo "OWGAP_N1000_ALL_DONE" | tee -a "$LOG"
