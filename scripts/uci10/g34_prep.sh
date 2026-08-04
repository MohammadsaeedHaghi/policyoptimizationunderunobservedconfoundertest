#!/bin/bash
#SBATCH --job-name=g34_prep
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1/semisynthetic"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
for DS in bank_marketing wine_quality german_credit credit_default adult; do
  for D in 0 1 2 3 4; do
    python3 prepare_semisynth.py --dataset "$DS" --gammas 3.0,4.0 --seed "$D"
  done
done
python3 - <<'PY'
import json, pathlib, collections
d = json.loads(pathlib.Path("prepared/_index.json").read_text())
new = [r for r in d if r["gamma"] in (3.0, 4.0)]
c = collections.Counter((r["dataset"], r.get("dgp_seed", 0)) for r in new)
print("index: %d total, %d at gamma in {3,4}, %d (dataset,dgp) pairs" % (len(d), len(new), len(c)))
assert len(new) == 50, "expected 5 datasets x 5 dgp seeds x 2 gammas = 50, got %d" % len(new)
assert len(d) == 150, "expected 100 old + 50 new = 150, got %d" % len(d)
h = [r["headroom"] for r in new]
print("headroom at gamma 3/4: %+.4f .. %+.4f" % (min(h), max(h)))
for g in (3.0, 4.0):
    hh = [r["headroom"] for r in new if r["gamma"] == g]
    print("  gamma=%.0f  Gamma=%.1f  headroom mean %+.4f" % (g, 2.718281828 ** (2 * g), sum(hh) / len(hh)))
print("INDEX OK")
PY
echo "=== G34 PREP DONE rc=$? $(date) ==="
