"""The REAL Hess et al. (2502.13022) estimator vs our plug-in sharp, on their own DGP and ours."""
import sys, importlib.util, numpy as np
ROOT="/home1/haghim/code 1.1"; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/assets/grand"); sys.path.insert(0,ROOT+"/methods/SharpHess")
from sharp_hess import sharp_hess_policy
from run_sharp_all import sharp_scores, policy as plugin_policy
def LD(p,n):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
CAMPS=[("KMZ (their own synthetic)","assets/exp_msmbench/dgp_g15.py",4.4817,
        [1,1.6487,2.7183,4.4817,7.3891,12,16]),
       ("gstar (ours)","assets/exp_gstar/dgp_cont.py",5.0,[1,2,3,5,8,12,16])]
N,NTE,SEEDS,BINS=200,200000,10,15
for lab,path,gstar,GG in CAMPS:
    d=LD(ROOT+"/"+path,"d_"+lab[:3])
    te,ft=d.generate(NTE,999); xte=te["X"].ravel(); Y1,Y0=ft["Y1"],ft["Y0"]
    print("\n=== %s  (n=%d, %d seeds, matched Gamma*=%.4f)" % (lab,N,SEEDS,gstar))
    print("   %-8s %18s %18s" % ("Gamma","Hess et al. (Eq15)","plug-in (ours)"))
    for g in GG:
        hv,pv=[],[]
        for sd in range(SEEDS):
            obs,_=d.generate(N,sd); X,T,Y=obs["X"],obs["T"],obs["Y"]
            x=X.ravel()
            # Hess: per-unit scores -> pointwise policy, deployed by nearest support point
            pi,_=sharp_hess_policy(X,T,Y,g,k=40,n_folds=2,maximize=True,seed=sd)
            nn=np.argmin(np.abs(xte[:,None]-x[None,:]),axis=1)
            pe=pi[nn]; hv.append(float(np.mean(pe*Y1+(1-pe)*Y0)))
            # plug-in: quantile bins from train
            ed=np.quantile(x,np.linspace(0,1,BINS+1)); ed[0]-=1e-9; ed[-1]+=1e-9
            ci=np.clip(np.digitize(x,ed)-1,0,BINS-1); cte=np.clip(np.digitize(xte,ed)-1,0,BINS-1)
            mass=np.array([(cte==j).mean() for j in range(BINS)])
            m1,m0=sharp_scores(Y,T,ci,BINS,float(g)); sc=np.nan_to_num(m1,nan=-1e9)-np.nan_to_num(m0,nan=0.0)
            p=plugin_policy(sc,mass)[cte]; pv.append(float(np.mean(p*Y1+(1-p)*Y0)))
        mark=" <-- Hess better" if np.mean(hv)>np.mean(pv) else ""
        print("   %-8s %8.3f +- %.3f %8.3f +- %.3f%s" % (g,np.mean(hv),np.std(hv),np.mean(pv),np.std(pv),mark))
    orc=d.oracle_policy(xte); print("   oracle %.3f | never %.3f" % (float((orc*Y1+(1-orc)*Y0).mean()), float(Y0.mean())))
