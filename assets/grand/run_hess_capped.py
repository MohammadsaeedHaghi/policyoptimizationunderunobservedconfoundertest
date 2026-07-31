"""Hess et al. under a 30% capacity constraint -- the capped panels need a CAPPED policy.

Eq. 15 is linear in pi, so under a mass budget the exact optimum is greedy on the score gap
(sharp_hess_policy(cap=...) does this). The capped charts were previously showing the UNCAPPED
Hess curve against capped references, which is why it appeared to beat the capped oracle: it was
not paying the budget.
"""
import sys, json, importlib.util
import numpy as np
ROOT="/home1/haghim/code 1.1"; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/methods/SharpHess")
from sharp_hess import fit_scores
def LD(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
PGRID=np.linspace(-1.0,1.0,41)
CAP=0.3
JOBS=[("km","assets/exp_msmbench/dgp_g15.py",400,5,["1","2","3","4.4817","6","8"]),
      ("kz","assets/exp_kz18/dgp.py",200,5,["1","2","3","4.4817","6","8"])]
out={"cap":CAP,"policy_grid":[round(float(v),4) for v in PGRID]}
for tag,path,N,seeds,GG in JOBS:
    d=LD(ROOT+"/"+path,"hc_"+tag)
    te,ft=d.generate(200000,999); xte=te["X"].ravel(); Y1,Y0=ft["Y1"],ft["Y0"]
    means,sds,curves={},{},{}
    for g in GG:
        vals=[]
        for sd in range(seeds):
            obs,_=d.generate(N,sd); X,T,Y=obs["X"],obs["T"],obs["Y"]; x=X.ravel()
            sc=fit_scores(X,T,Y,float(g),k=max(40,N//10),n_folds=2,maximize=True,seed=sd,standardize=True)
            gap=sc[:,1]-sc[:,0]
            # greedy top-cap on the score gap = exact optimum of a linear objective under a budget
            thr=np.quantile(gap,1.0-CAP)
            pol_tr=((gap>thr)&(gap>0)).astype(float)
            nn=np.argmin(np.abs(xte[:,None]-x[None,:]),axis=1)
            pe=pol_tr[nn]
            # enforce the budget on the deployed policy too
            if pe.mean()>CAP:
                keep=np.argsort(-gap[nn])[:int(CAP*len(pe))]
                pe=np.zeros(len(pe)); pe[keep]=1.0
            vals.append(float(np.mean(pe*Y1+(1-pe)*Y0)))
            if sd==0:
                cg=np.argmin(np.abs(PGRID[:,None]-x[None,:]),axis=1)
                curves[g]=[round(float(v),4) for v in pol_tr[cg]]
        means[g]=round(float(np.mean(vals)),4); sds[g]=round(float(np.std(vals)),4)
        print("  %s cap30 G=%-8s %.3f +- %.3f (treats %.2f)" % (tag,g,means[g],sds[g],float(np.mean(curves[g]))),flush=True)
    out[tag]={"n":N,"seeds":seeds,"gammas":GG,"mean":[means[g] for g in GG],
              "sd":[sds[g] for g in GG],"curves":curves}
json.dump(out,open(ROOT+"/assets/grand/hess_capped.json","w"))
print("saved assets/grand/hess_capped.json")
