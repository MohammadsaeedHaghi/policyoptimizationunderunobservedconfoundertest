#!/bin/bash
#SBATCH --job-name=time1k
#SBATCH --partition=debug
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=0:50:00
#SBATCH --output=/home1/haghim/time1k_%j.out
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 - <<'PY'
import sys, importlib.util, time, numpy as np, resource
ROOT="/home1/haghim/code 1.1"; sys.path.insert(0,ROOT)
import common
def L(rel,fn):
    s=importlib.util.spec_from_file_location(fn,ROOT+"/"+rel); m=importlib.util.module_from_spec(s); sys.modules[fn]=m; s.loader.exec_module(m); return getattr(m,fn)
solve=L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py","solve_ipw_o_w_uncapped")
sp=importlib.util.spec_from_file_location("d",ROOT+"/assets/exp_gstar/dgp_cont.py")
d=importlib.util.module_from_spec(sp); sys.modules["d"]=d; sp.loader.exec_module(d)
for n in (500, 1000):
    obs,_=d.generate(n,0); X,T,Y=obs["X"],obs["T"],obs["Y"]
    w,_=common.ipw_weights_from_data(X,T,2)
    t0=time.time(); Dm=common.pairwise_distance_matrix(X)
    eps=tuple(common.tight_epsilon(Dm,T,w,2,is_distance=True,c_eps=1.0)); t_eps=time.time()-t0
    cells=np.clip(np.digitize(X.ravel(),np.quantile(X.ravel(),np.linspace(0,1,11)))-1,0,9)
    for tag,cl in (("O-W",None),("Sharp-O-W(10)",cells)):
        t0=time.time()
        r=solve(X,T,Y,w,n_arms=2,Gamma=5.0,discretize=False,zscore=False,epsilon=eps,lipschitz=3.0,sharp_cells=cl)
        print("n=%-5d %-14s solve %6.1f s | eps-precompute %5.1f s | peakRSS %.1f GB"
              % (n,tag,time.time()-t0,t_eps,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6), flush=True)
PY
echo "TIME1K_DONE"
