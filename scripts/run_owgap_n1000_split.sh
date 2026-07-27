#!/usr/bin/env bash
# exp_owgap, N=1000, 8 seeds, 3 epsilons. SPLIT BY REGIME so each Python invocation stays
# under the Gurobi WLS 32-minute overage limit (baseline=2; >2 sessions for >32 min => killed).
# Each (epsilon, regime) = 8 jobs (8 seeds) on 8 workers = ONE wave ~19 min < 32 min. Sessions
# release between invocations, resetting the overage clock. Then merge the two regimes per epsilon.
set -e
cd "/home1/haghim/code 1.1"
N=1000; SEEDS=8; WORKERS=8; THREADS="${THREADS:-1}"
BK="results_n1000_backup"; mkdir -p "$BK" "selected experiment/exp_owgap"
LOG="run_owgap_n1000.log"
echo "=== $(date +%F_%H:%M:%S) owgap N=1000 8-seed SPLIT START on $(hostname) (workers=$WORKERS/regime) ===" | tee "$LOG"
for CE in 1.0 1.5 2.0; do
  for REG in uncap cap; do
    OUT="assets/exp_owgap/.owgap_n1000_ce${CE}_${REG}.json"
    echo "=== $(date +%H:%M:%S) ce=$CE reg=$REG (8 workers, ~19min) ===" | tee -a "$LOG"
    python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap/dgp.py --out "$OUT" \
      --n "$N" --seeds "$SEEDS" --gammas "1,1.5,2,2.5,3,4,5,6,8" --regimes "$REG" \
      --ceps "$CE" --workers "$WORKERS" --threads "$THREADS" \
      2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license" | tee -a "$LOG"
    # guard: detect a masked crash (overage etc.) — file must be valid JSON containing this regime
    python3 -c "import json; d=json.load(open('$OUT')); assert '$REG' in d['regimes']" \
      || { echo "!! INVOCATION CRASHED at ce=$CE reg=$REG (overage?) — ABORTING" | tee -a "$LOG"; exit 1; }
    echo "=== $(date +%H:%M:%S) ce=$CE reg=$REG OK ===" | tee -a "$LOG"
    sleep 25   # let WLS sessions fully release -> reset the overage clock
  done
  MERGED="assets/exp_owgap/owgap_results_n1000_ce${CE}.json"
  python3 merge_regimes.py "assets/exp_owgap/.owgap_n1000_ce${CE}_uncap.json" "assets/exp_owgap/.owgap_n1000_ce${CE}_cap.json" "$MERGED" | tee -a "$LOG"
  cp -f "$MERGED" "$BK/" 2>/dev/null
  cp -f "$MERGED" "selected experiment/exp_owgap/" 2>/dev/null
  echo "=== $(date +%H:%M:%S) EPSILON ce=$CE DONE (merged + saved) ===" | tee -a "$LOG"
done
echo "OWGAP_N1000_ALL_DONE" | tee -a "$LOG"
