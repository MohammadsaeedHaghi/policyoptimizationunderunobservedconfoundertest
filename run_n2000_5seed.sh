#!/usr/bin/env bash
# N=2000, 5-seed reruns of the two selected experiments (exp_owgap + exp_nmcap).
# Run this INSIDE a SLURM allocation with more memory, e.g.:
#   salloc --mem=128G --cpus-per-task=32 --time=12:00:00
# then:  bash "run_n2000_5seed.sh"
#
# Tuning notes:
#   - Gurobi WLS license caps concurrent sessions at 8 -> WORKERS=8 is the ceiling.
#   - Extra cores help via THREADS per solve (8 workers x 4 threads = 32 cores).
#   - N=2000 Wasserstein LPs are memory-heavy; needs ~>=64-128 GB at 8 workers (32 GB OOM-kills).
set -e
cd "$(dirname "$0")"
N=2000; SEEDS=5; WORKERS=10; THREADS="${THREADS:-3}"
LOG="run_n2000_5seed.log"
echo "=== $(date +%F_%H:%M:%S) N=2000 5-seed run START (workers=$WORKERS threads=$THREADS) ===" | tee "$LOG"

for CE in 1.0 1.5 2.0; do
  OUT="assets/exp_owgap/owgap_results_n2000_ce${CE}.json"
  echo "=== $(date +%H:%M:%S) owgap c_eps=${CE} -> ${OUT} ===" | tee -a "$LOG"
  python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap/dgp.py --out "$OUT" \
    --n "$N" --seeds "$SEEDS" --gammas "1,1.5,2,2.5,3,4,5,6,8" --regimes uncap,cap \
    --ceps "$CE" --workers "$WORKERS" --threads "$THREADS" \
    2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license" | tee -a "$LOG"
  echo "=== $(date +%H:%M:%S) owgap c_eps=${CE} DONE ===" | tee -a "$LOG"
done
echo "OWGAP_N2000_DONE" | tee -a "$LOG"

OUT="assets/exp_nmcap/nmcap_results_n2000_ce1.0.json"
echo "=== $(date +%H:%M:%S) nmcap c_eps=1.0 -> ${OUT} ===" | tee -a "$LOG"
python3 assets/run_experiment_parallel.py --dgp assets/exp_nmcap/dgp.py --out "$OUT" \
  --n "$N" --seeds "$SEEDS" --gammas "1,1.5,2,2.5,3,4" --regimes uncap,cap \
  --ceps 1.0 --workers "$WORKERS" --threads "$THREADS" \
  2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license" | tee -a "$LOG"
echo "=== $(date +%H:%M:%S) nmcap DONE ===" | tee -a "$LOG"
echo "ALL_N2000_DONE" | tee -a "$LOG"
