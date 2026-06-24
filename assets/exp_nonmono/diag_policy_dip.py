"""Diagnose the 'treat-prob dips inside the beneficial band' question: refit R-OW (nonmono seed 0, n=500) at the matched Γ
under different capacity caps, and at a lower Γ. Shows (1) the cap binds (treated frac≈cap < the 52% CATE>0 band), and
(2) at the large matched Γ the per-cell worst-case values are nearly tied → LP degeneracy → an arbitrary bang-bang subset."""
import sys, importlib.util
import numpy as np
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
HERE=NEW/"assets"/"exp_nonmono"
sp=importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
row=getattr(importlib.util.module_from_spec(importlib.util.spec_from_file_location("row",str(NEW/"methods/IPW-O-W/Capped/ipw_o_w_capped.py"))),"__name__",None)
def load(n,rel):
    s=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
solve=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
K=dgp.K; mG=dgp.GAMMAS[dgp.mi]
rng=np.random.default_rng(0); tr=dgp.generate(dgp.n_tr,rng)
w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0))
uniq={}
for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
uk=np.array(sorted(uniq)); uidx=np.array([uniq[k] for k in uk])
cate=(0.5*(dgp.pa(uk,0,1)+dgp.pa(uk,1,1)))-(0.5*(dgp.pa(uk,0,0)+dgp.pa(uk,1,0)))
band=cate>1e-6
print("grid X:",np.round(uk,2))
print("true CATE>0 band:",band.astype(int),"  (%d of %d = %.3f of units)"%(band.sum(),len(uk),band.mean()))
def polrow(G,cap):
    pi=np.asarray(solve(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=cap,discretize=False,zscore=False,epsilon=eps).pi,float)
    p=pi[:,uidx][1]; frac=float((pi[1]).mean()); return p,frac
for G,cap,lab in [(mG,(1.0,0.5),"matchedΓ cap=0.50 (the chart)"),(mG,(1.0,0.6),"matchedΓ cap=0.60 (looser)"),
                  (mG,(1.0,1.0),"matchedΓ cap=1.00 (no cap)"),(3.0,(1.0,0.5),"Γ=3 cap=0.50 (smaller box)")]:
    p,frac=polrow(G,cap)
    inb=p[band].mean(); outb=p[~band].mean()
    print("\n%s  | treated frac=%.3f"%(lab,frac))
    print("  P(treat) per cell:",np.round(p,2))
    print("  mean P(treat) IN-band=%.2f  OUT-band=%.2f"%(inb,outb))
