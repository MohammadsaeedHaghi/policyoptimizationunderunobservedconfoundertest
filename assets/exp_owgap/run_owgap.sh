#!/usr/bin/env bash
# Full exp_owgap study: run the 8-method sweep at N=700, 5 seeds, both regimes, ONCE PER epsilon (c_eps).
# Runs are SEQUENTIAL because the Gurobi WLS licence caps concurrent sessions (8 workers is the safe ceiling).
set -e
cd "$(dirname "$0")/../.."   # -> repo root (code 1.1)
DGP=assets/exp_owgap/dgp.py
GAMMAS="1,1.5,2,2.5,3,4,5,6,8"
N=600; SEEDS=5; WORKERS=8; THREADS=1   # N=600 (~700): the O(n^2) Wasserstein LP makes N=700x5x9x3 impractical here
for CE in 1.0 1.5 2.0; do
  OUT="assets/exp_owgap/owgap_results_ce${CE}.json"
  echo "=== c_eps=${CE} -> ${OUT} ==="
  python3 assets/run_experiment_parallel.py --dgp "$DGP" --out "$OUT" \
    --n "$N" --seeds "$SEEDS" --gammas "$GAMMAS" --regimes uncap,cap \
    --ceps "$CE" --workers "$WORKERS" --threads "$THREADS" \
    2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license"
done
echo "ALL EPSILON RUNS DONE"
