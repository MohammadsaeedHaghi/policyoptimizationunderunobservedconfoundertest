"""The X-X (unconfoundedness-assuming) methods on the KMZ benchmark, at the report's protocol.

The continuous runner only ever solved the five ROBUST methods plus a hand-computed naive_dr
reference, so the KMZ report has no IPW-X-X / DoublyRobust-X-X / Direct-X-X rows -- the very
baselines the paper's claim is about. These are Gamma-free (they assume unconfoundedness), so
each is a single number per regime, not a curve.
"""
import sys, json, importlib.util
import numpy as np
ROOT="/home1/haghim/code 1.1"; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/extensions/Shapley")
import common
from shapley import extract_support
def LD(rel,fn):
    s=importlib.util.spec_from_file_location(fn,ROOT+"/"+rel); m=importlib.util.module_from_spec(s); sys.modules[fn]=m; s.loader.exec_module(m); return getattr(m,fn)
S={"IPW-X-X":(LD("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py","solve_ipw_x_x_uncapped"),"ipw"),
   "DoublyRobust-X-X":(LD("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py","solve_doublyrobust_x_x_uncapped"),"dr"),
   "Direct-X-X":(LD("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py","solve_direct_x_x_uncapped"),"direct")}
def shap(Xn,sX,spv,block=200):
    Xn=np.asarray(Xn,float).reshape(-1,1); out=np.empty(len(Xn)); dg=np.arange(len(sX))
    for s0 in range(0,len(Xn),block):
        xb=Xn[s0:s0+block]; dd=np.sqrt(((xb[:,None,:]-sX[None,:,:])**2).sum(-1))
        Sm=dd[:,:,None]+dd[:,None,:]; Sm=np.where(Sm>0,Sm,1.0)
        A=(dd[:,None,:]*spv[None,:,None]+dd[:,:,None]*spv[None,None,:])/Sm
        A[:,dg,dg]=spv[None,:]; v=A.max(axis=1).min(axis=1)
        ex=dd.min(axis=1)<=0
        if ex.any(): v[ex]=spv[dd[ex].argmin(axis=1)]
        out[s0:s0+block]=v
    return out
sp=importlib.util.spec_from_file_location("d",ROOT+"/assets/exp_msmbench/dgp_g15.py")
d=importlib.util.module_from_spec(sp); sys.modules["d"]=d; sp.loader.exec_module(d)
N,NTE,SEEDS=400,4000,5
PG=np.linspace(-1,1,41)
out={"n":N,"seeds":SEEDS,"methods":list(S)}
vals={m:[] for m in S}; curves={}
for sd in range(SEEDS):
    obs,_=d.generate(N,sd); X,T,Y=obs["X"],obs["T"],obs["Y"]
    te,ft=d.generate(NTE,sd+1000); Xte=te["X"]; Y1,Y0=ft["Y1"],ft["Y0"]
    w,_=common.ipw_weights_from_data(X,T,2)
    mu=common.outcome_means(X,T,Y,n_arms=2,cross_fit=True)
    for m,(fn,kind) in S.items():
        try:
            if kind=="ipw":   r=fn(X,T,Y,w,n_arms=2,discretize=False)
            elif kind=="dr":  r=fn(X,T,Y,w,mu,n_arms=2,discretize=False)
            else:             r=fn(X,T,Y,n_arms=2,discretize=False)
            sX,spv=extract_support(X,r.pi[1])
            pe=shap(Xte.ravel(),np.asarray(sX,float),np.asarray(spv,float).ravel())
            vals[m].append(float(np.mean(pe*Y1+(1-pe)*Y0)))
            if sd==0: curves[m]=[round(float(v),4) for v in shap(PG,np.asarray(sX,float),np.asarray(spv,float).ravel())]
        except Exception as ex: print("FAIL",m,sd,ex,flush=True)
for m in S: print("  %-18s %.3f +- %.3f" % (m,np.mean(vals[m]),np.std(vals[m])),flush=True)
out["mean"]={m:round(float(np.mean(vals[m])),4) for m in S}
out["sd"]={m:round(float(np.std(vals[m])),4) for m in S}
out["curves"]=curves; out["policy_grid"]=[round(float(v),4) for v in PG]
json.dump(out,open(ROOT+"/assets/grand/xx_kmz.json","w"))
print("saved assets/grand/xx_kmz.json")
