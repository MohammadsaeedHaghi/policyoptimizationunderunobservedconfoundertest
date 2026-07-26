#!/bin/bash
# All Kallus baseline runs (box-only, no Gurobi session -- safe alongside the LP chain).
set -e
cd "/home1/haghim/code 1.1"
G=1,1.5,2,2.5,3,4,5,6,8
python3 assets/exp_owgap_alpha/run_kallus.py --dgp assets/exp_owgap/dgp.py \
  --out assets/exp_owgap/owgap_kallus_20seed.json --n 600 --seeds 20 --gammas $G
cp assets/exp_owgap/owgap_kallus_20seed.json "selected experiment/exp_owgap/"
for a in 1 2 4 6 10; do
  python3 assets/exp_owgap_alpha/run_kallus.py --dgp "assets/exp_owgap_alpha/dgp_a${a}.py" \
    --out "assets/exp_owgap_alpha/kallus_alpha${a}.json" --n 600 --seeds 8 --gammas $G
  cp "assets/exp_owgap_alpha/kallus_alpha${a}.json" "selected experiment/exp_owgap_alpha/"
done
echo "KALLUS_ALL_DONE $(date)"
