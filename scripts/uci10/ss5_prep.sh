#!/bin/bash
#SBATCH --job-name=ss5_prep
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
  for D in 1 2 3 4; do
    python3 prepare_semisynth.py --dataset "$DS" --gammas 0.0,1.0,1.5,2.0 --seed "$D"
  done
done
python3 - <<'PY'
import json, pathlib, collections
d = json.loads(pathlib.Path("prepared/_index.json").read_text())
c = collections.Counter((r["dataset"], r.get("dgp_seed", 0)) for r in d)
print("\nindex: %d entries, %d (dataset, dgp_seed) pairs" % (len(d), len(c)))
assert len(d) == 100, "expected 5 datasets x 5 dgp seeds x 4 gammas = 100"
a = sorted({round(r["tau_params_a"], 5) for r in d})
print("distinct a values now: %d  %s" % (len(a), a))
h = [r["headroom"] for r in d]
print("headroom range across all draws: %+.4f .. %+.4f" % (min(h), max(h)))
print("INDEX OK")
PY
echo "=== SS5 PREP DONE rc=$? $(date) ==="
