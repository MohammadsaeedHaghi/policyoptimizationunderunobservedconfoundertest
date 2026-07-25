#!/usr/bin/env bash
# exp_nmcap ONLY, N=2000, 5 seeds — run in parallel on the interactive node while owgap runs in the batch job.
# 6 workers (keeps total Gurobi sessions <=16 alongside owgap's 10). Writes to the canonical folder.
set -e
cd "/home1/haghim/code 1.1"
LOG="run_nmcap_n2000.log"
echo "=== $(date +%F_%H:%M:%S) nmcap N=2000 5-seed START on $(hostname) (workers=6 threads=3) ===" | tee "$LOG"
python3 assets/run_experiment_parallel.py --dgp assets/exp_nmcap/dgp.py \
  --out assets/exp_nmcap/nmcap_results_n2000_ce1.0.json \
  --n 2000 --seeds 5 --gammas "1,1.5,2,2.5,3,4" --regimes uncap,cap \
  --ceps 1.0 --workers 6 --threads 3 \
  2>&1 | grep -vE "Set parameter|Read parameters|WLSAccess|WLSSecret|LicenseID|Academic license" | tee -a "$LOG"
cp -f assets/exp_nmcap/nmcap_results_n2000_ce1.0.json results_n2000_backup/ 2>/dev/null
cp -f assets/exp_nmcap/nmcap_results_n2000_ce1.0.json "selected experiment/exp_nmcap/" 2>/dev/null
echo "=== $(date +%H:%M:%S) NMCAP_N2000_DONE ===" | tee -a "$LOG"
