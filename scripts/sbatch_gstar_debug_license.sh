#!/bin/bash
#SBATCH --job-name=gstar_lic_dbg
#SBATCH --partition=debug
#SBATCH --account=vayanou_651
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --output=/home1/haghim/gstar_lic_dbg_%j.out
# Phase 0 of the exp_gstar campaign: validate that the CARC cluster Gurobi token license
# (module gurobi/12.0.3 -> TOKENSERVER hpc-licenses.hpcc.usc.edu) works through our whole
# solver stack: pip gurobipy discovery, a real O-W solve, tight_epsilon, and the spawn-pool
# discrete runner. Success = three OK markers and NO WLS banners (WLSAccessID).
echo "SBATCH START $(date) on $(hostname)  job=$SLURM_JOB_ID"
module load gurobi/12.0.3
echo "GRB_LICENSE_FILE=$GRB_LICENSE_FILE"
cat "$GRB_LICENSE_FILE"
echo "GUROBI_HOME=$GUROBI_HOME"
ls "$GUROBI_HOME/lib" 2>/dev/null | head
cd "/home1/haghim/code 1.1"
python3 - <<'EOF'
import gurobipy as gp, os
print("gurobipy", gp.gurobi.version(), "lic:", os.environ.get("GRB_LICENSE_FILE"))
from common.gurobi_env import configure_gurobi_license
configure_gurobi_license()
print("LICENSE-PROBE-OK, GRB_LICENSE_FILE now:", os.environ["GRB_LICENSE_FILE"])
EOF
python3 - <<'EOF'
import importlib.util, sys, numpy as np
sys.path.insert(0, ".")
import common
sp = importlib.util.spec_from_file_location("d", "assets/exp_owgap_v2/dgp.py")
d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d)
obs, _ = d.generate(120, 0)
w, _ = common.ipw_weights_from_data(obs["X"], obs["T"], 2)
Dm = common.pairwise_distance_matrix(obs["X"])
eps = tuple(common.tight_epsilon(Dm, obs["T"], w, 2, is_distance=True, c_eps=1.0))
print("TIGHT-EPS-TOKEN-OK:", eps)
sp2 = importlib.util.spec_from_file_location("s", "methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py")
m = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(m)
r = m.solve_ipw_o_w_uncapped(obs["X"], obs["T"], obs["Y"], w, n_arms=2, Gamma=2.0,
                             discretize=False, zscore=False, epsilon=eps)
print("SOLVER-TOKEN-OK obj=%.4f" % r.objective_value)
EOF
python3 assets/run_experiment_parallel.py --dgp assets/exp_owgap_v2/dgp.py \
  --out /tmp/gstar_dbg_$SLURM_JOB_ID.json --n 80 --seeds 2 --gammas 5 \
  --regimes uncap,cap --ceps 1.0 --workers 4 --threads 1 \
  && echo "SPAWN-POOL-TOKEN-OK"
echo "SBATCH END $(date)"
