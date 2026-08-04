#!/bin/bash
#SBATCH --job-name=t500
#SBATCH --partition=debug
#SBATCH --account=vayanou_651
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=0:30:00
#SBATCH --output=/home1/haghim/t500_%j.out
module load gurobi/12.0.3
cd "/home1/haghim/code 1.1"
python3 - <<'PY'
import sys, importlib.util, time, numpy as np
ROOT="/home1/haghim/code 1.1"; sys.path.insert(0,ROOT)
import common
def L(rel,fn):
    s=importlib.util.spec_from_file_location(fn,ROOT+"/"+rel); m=importlib.util.module_from_spec(s); sys.modules[fn]=m; s.loader.exec_module(m); return getattr(m,fn)
S={"IPW-O-X":L("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py","solve_ipw_o_x_uncapped"),
   "DoublyRobust-O-X":L("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py","solve_doublyrobust_o_x_uncapped"),
   "Hajek-O-X":L("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py","solve_hajek_o_x_uncapped"),
   "IPW-O-W":L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py","solve_ipw_o_w_uncapped"),
   "DoublyRobust-O-W":L("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py","solve_doublyrobust_o_w_uncapped")}
sp=importlib.util.spec_from_file_location("d",ROOT+"/assets/exp_gstar/dgp_cont.py")
d=importlib.util.module_from_spec(sp); sys.modules["d"]=d; sp.loader.exec_module(d)
n=500
obs,_=d.generate(n,0); X,T,Y=obs["X"],obs["T"],obs["Y"]
w,_=common.ipw_weights_from_data(X,T,2); wr,_=common.ipw_weights_from_data(X,T,2,normalize=False)
mu=common.outcome_means(X,T,Y,n_arms=2,cross_fit=True)
t0=time.time(); Dm=common.pairwise_distance_matrix(X)
eps=tuple(common.tight_epsilon(Dm,T,w,2,is_distance=True,c_eps=1.0)); print("setup(eps LP) %.1fs"%(time.time()-t0),flush=True)
tot=0
for m,f in S.items():
    t0=time.time()
    if m=="IPW-O-X": f(X,T,Y,w,n_arms=2,Gamma=5.0,discretize=False,lipschitz=3.0)
    elif m=="DoublyRobust-O-X": f(X,T,Y,w,mu,n_arms=2,Gamma=5.0,discretize=False,lipschitz=3.0)
    elif m=="Hajek-O-X": f(X,T,Y,wr,n_arms=2,Gamma=5.0,maximize=True,discretize=False,lipschitz=3.0)
    elif m=="IPW-O-W": f(X,T,Y,w,n_arms=2,Gamma=5.0,discretize=False,zscore=False,epsilon=eps,lipschitz=3.0)
    else: f(X,T,Y,w,mu,n_arms=2,Gamma=5.0,discretize=False,zscore=False,epsilon=eps,lipschitz=3.0)
    dt=time.time()-t0; tot+=dt; print("  %-18s %6.2f s"%(m,dt),flush=True)
print("SUM per (Gamma,L,eps,cap) cell over 5 methods: %.1f s"%tot)
PY
echo T500_DONE
