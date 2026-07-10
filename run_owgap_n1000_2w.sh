#!/usr/bin/env bash
# exp_owgap, N=1000, 4 seeds, 3 epsilons, WORKERS=2 (== Gurobi WLS baseline => ZERO overage, crash-free).
# 4 seeds x 2 regimes = 8 jobs, 2 workers -> 4 waves/epsilon (~19min each) => ~76 min/epsilon, ~3.8h total.
set -e
cd "/home1/haghim/code 1.1"
N=1000; SEEDS=4; WORKERS=2; THREADS="${THREADS:-1}"
BK="results_n1000_backup"; mkdir -p "$BK" "selected experiment/exp_owgap"
LOG="run_owgap_n1000.log"
echo "=== $(date +%F_%H:%M:%S) owgap N=1000 4-seed 2-worker START on $(hostname) ===" | tee "$LOG"
for CE in 1.0 1.5 2.0; do
  OUT="assets/exp_owgap/owgap_results_n1000_ce${CE}.json"
  echo "=== $(date +%H:%M:%S) ce=$CE (2 workers, ~76min) -> ${OUT} ===" | tee -a "$LOG"
  python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap/dgp.py --out "$OUT" \
    --n "$N" --seeds "$SEEDS" --gammas "1,1.5,2,2.5,3,4,5,6,8" --regimes uncap,cap \
    --ceps "$CE" --workers "$WORKERS" --threads "$THREADS" \
    2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license" | tee -a "$LOG"
  python3 -c "import json; d=json.load(open('$OUT')); assert set(d['regimes'])>= {'uncap','cap'}" \
    || { echo "!! ce=$CE invocation failed (file invalid) — ABORTING" | tee -a "$LOG"; exit 1; }
  cp -f "$OUT" "$BK/" 2>/dev/null; cp -f "$OUT" "selected experiment/exp_owgap/" 2>/dev/null
  echo "=== $(date +%H:%M:%S) ce=$CE DONE (saved + backed up) ===" | tee -a "$LOG"
done
echo "OWGAP_N1000_ALL_DONE" | tee -a "$LOG"
