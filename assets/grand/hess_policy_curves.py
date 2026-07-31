"""Hess et al. policy curves pi(x) on the report grid, for the pi(x) viewers (1-D, binary A)."""
import sys, json, importlib.util
import numpy as np
ROOT="/home1/haghim/code 1.1"; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/methods/SharpHess")
from sharp_hess import fit_scores, learn_policy_parametric, apply_policy
def LD(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
PGRID=np.linspace(-1.0,1.0,41)
JOBS=[("km","assets/exp_msmbench/dgp_g15.py",400,["1","2","3","4.4817","6","8"]),
      ("kz","assets/exp_kz18/dgp.py",200,["1","2","3","4.4817","6","8"]),
      ("gs_cont","assets/exp_gstar/dgp_cont.py",400,["1","1.5","2","2.5","3","4","5","6","8"])]
out={"policy_grid":[round(float(v),4) for v in PGRID]}
for tag,path,N,GG in JOBS:
    d=LD(ROOT+"/"+path,"pc_"+tag)
    obs,_=d.generate(N,0); X,T,Y=obs["X"],obs["T"],obs["Y"]      # seed 0, matching the viewers
    cur={}
    for g in GG:
        sc=fit_scores(X,T,Y,float(g),k=max(40,N//10),n_folds=2,maximize=True,seed=0,standardize=True)
        pol=learn_policy_parametric(X,sc,n_iter=600,lr=0.3,restarts=5,seed=0,maximize=True,hidden=8)
        cur[g]=[round(float(v),4) for v in apply_policy(pol,PGRID.reshape(-1,1))]
        print("  %s G=%-8s treats %.2f on the grid" % (tag,g,float(np.mean(cur[g]))),flush=True)
    out[tag]=cur
json.dump(out,open(ROOT+"/assets/grand/hess_policy_curves.json","w"))
print("saved assets/grand/hess_policy_curves.json")
