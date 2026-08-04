#!/bin/bash
#SBATCH --job-name=ss4_prep
#SBATCH --account=vayanou_651
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -u
module purge
module load gcc/13.3.0 python/3.11.9 gurobi/12.0.3
cd "/home1/haghim/code 1.1/semisynthetic"
export TMPDIR=/scratch1/$USER/tmp; mkdir -p "$TMPDIR"
rm -f prepared/_index.json                  # rebuild the shared index from scratch
for DS in bank_marketing wine_quality german_credit credit_default adult; do
  echo "=========== $DS ==========="
  python3 prepare_semisynth.py --dataset "$DS" --gammas 0.0,1.0,1.5,2.0
done
python3 - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("prepared/_index.json").read_text())
ds = sorted({r["dataset"] for r in d})
print("\n_index.json holds %d entries over %d datasets: %s" % (len(d), len(ds), ds))
assert len(d) == 20, "expected 5 datasets x 4 gammas"
print("INDEX OK")
PY
echo "=== SS4 PREP DONE rc=$? $(date) ==="
