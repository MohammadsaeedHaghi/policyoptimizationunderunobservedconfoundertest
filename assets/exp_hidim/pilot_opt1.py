"""Option 1 pilot: index reduction, applied EQUALLY to our methods and to sharp.

Representations compared (n=200, d=8, matched Gamma=10):
  raw      -- the 8-D covariate (what failed: IPW-O-W 0.056 vs sharp 0.414)
  idx1     -- 1-D supervised CATE index
  idx2     -- 2-D (CATE index, orthogonalised propensity index)
Within each representation: our 5 methods (free-pi and k-means-cell classes) and sharp
(k-means cells, its own count swept) all see the SAME covariate.
"""
import sys, importlib.util, numpy as np
from sklearn.cluster import KMeans
ROOT="/home1/haghim/code 1.1"
sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/extensions/Shapley"); sys.path.insert(0,ROOT+"/assets/grand"); sys.path.insert(0,ROOT+"/assets/exp_hidim")
import common
from shapley import extract_support
from run_sharp_all import sharp_scores
from reduce import index_directions, reduce_X
def LD(rel,fn):
    s=importlib.util.spec_from_file_location(fn,ROOT+"/"+rel); m=importlib.util.module_from_spec(s); sys.modules[fn]=m; s.loader.exec_module(m); return getattr(m,fn)
SOL={"IPW-O-X":LD("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py","solve_ipw_o_x_uncapped"),
     "DoublyRobust-O-X":LD("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py","solve_doublyrobust_o_x_uncapped"),
     "IPW-O-W":LD("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py","solve_ipw_o_w_uncapped"),
     "DoublyRobust-O-W":LD("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py","solve_doublyrobust_o_w_uncapped")}
sp=importlib.util.spec_from_file_location("d",ROOT+"/assets/exp_hidim/dgp.py"); d=importlib.util.module_from_spec(sp); sys.modules["d"]=d; sp.loader.exec_module(d)
def shap(Xn,sX,spv,block=200):
    Xn=np.asarray(Xn,float); out=np.empty(len(Xn)); dg=np.arange(len(sX))
    for s0 in range(0,len(Xn),block):
        xb=Xn[s0:s0+block]; dd=np.sqrt(((xb[:,None,:]-sX[None,:,:])**2).sum(-1))
        Sm=dd[:,:,None]+dd[:,None,:]; Sm=np.where(Sm>0,Sm,1.0)
        A=(dd[:,None,:]*spv[None,:,None]+dd[:,:,None]*spv[None,None,:])/Sm
        A[:,dg,dg]=spv[None,:]; v=A.max(axis=1).min(axis=1)
        ex=dd.min(axis=1)<=0
        if ex.any(): v[ex]=spv[dd[ex].argmin(axis=1)]
        out[s0:s0+block]=v
    return out
N,NTE,G=200,4000,10.0
acc={}
for sd in range(5):
    obs,_=d.generate(N,sd); X,T,Y=obs["X"],obs["T"],obs["Y"]
    te,ft=d.generate(NTE,sd+1000); Xte=te["X"]; Y1,Y0=ft["Y1"],ft["Y0"]
    mu=common.outcome_means(X,T,Y,n_arms=2,cross_fit=True)
    w,_=common.ipw_weights_from_data(X,T,2)
    b1,b2=index_directions(X,T,Y,mu=mu)
    orc=d.oracle_policy(Xte)
    acc.setdefault("_oracle",[]).append(float((orc*Y1+(1-orc)*Y0).mean()))
    acc.setdefault("_never",[]).append(float(Y0.mean()))
    for rep,(Z,Zte) in (("raw",(X,Xte)), ("idx1",reduce_X(X,Xte,[b1])), ("idx2",reduce_X(X,Xte,[b1,b2]))):
        Dm=common.pairwise_distance_matrix(Z)
        try: eps=tuple(common.tight_epsilon(Dm,T,w,2,is_distance=True,c_eps=1.0))
        except Exception as ex: print("eps FAIL",rep,ex,flush=True); continue
        # sharp on the SAME representation
        for nc in (15,30):
            km=KMeans(n_clusters=nc,n_init=4,random_state=0).fit(Z); ci=km.labels_; cte=km.predict(Zte)
            m1,m0=sharp_scores(Y,T,ci,nc,G); sc=np.nan_to_num(m1,nan=-1e9)-np.nan_to_num(m0,nan=0.0)
            p=(sc>0).astype(float)[cte]
            acc.setdefault("%s | Sharp-O-X(%d)"%(rep,nc),[]).append(float((p*Y1+(1-p)*Y0).mean()))
        # our methods: free-pi (Lipschitz) and k-means-cell classes, same representation
        for m in SOL:
            best=-9e9
            for Lv in (None,5.0,3.0,1.0):
                try:
                    if m=="IPW-O-X": r=SOL[m](Z,T,Y,w,n_arms=2,Gamma=G,discretize=False,lipschitz=Lv)
                    elif m=="DoublyRobust-O-X": r=SOL[m](Z,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False,lipschitz=Lv)
                    elif m=="IPW-O-W": r=SOL[m](Z,T,Y,w,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=eps,lipschitz=Lv)
                    else: r=SOL[m](Z,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=eps,lipschitz=Lv)
                    sX,spv=extract_support(Z,r.pi[1]); pe=shap(Zte,np.asarray(sX,float),np.asarray(spv,float).ravel())
                    best=max(best,float((pe*Y1+(1-pe)*Y0).mean()))
                except Exception as ex: print("FAIL",rep,m,Lv,ex,flush=True)
            if best>-9e8: acc.setdefault("%s | %s"%(rep,m),[]).append(best)
            for nc in (15,30):
                km=KMeans(n_clusters=nc,n_init=4,random_state=0).fit(Z); ci=km.labels_; cte=km.predict(Zte)
                Zb=km.cluster_centers_[ci]
                try:
                    Db=common.pairwise_distance_matrix(Zb); epsb=tuple(common.tight_epsilon(Db,T,w,2,is_distance=True,c_eps=1.0))
                    if m=="IPW-O-X": r=SOL[m](Zb,T,Y,w,n_arms=2,Gamma=G,discretize=False)
                    elif m=="DoublyRobust-O-X": r=SOL[m](Zb,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False)
                    elif m=="IPW-O-W": r=SOL[m](Zb,T,Y,w,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=epsb)
                    else: r=SOL[m](Zb,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=epsb)
                    pi=np.asarray(r.pi[1],float)
                    cp=np.array([pi[ci==j].mean() if (ci==j).any() else 0.0 for j in range(nc)])
                    pe=cp[cte]
                    acc.setdefault("%s | %s [km%d]"%(rep,m,nc),[]).append(float((pe*Y1+(1-pe)*Y0).mean()))
                except Exception as ex: print("FAIL km",rep,m,nc,ex,flush=True)
    print("seed",sd,"done",flush=True)
print("\n=== OPTION 1: index reduction, n=200, d=8, matched Gamma=10, 5 seeds ===")
print("    (sharp gets the SAME representation -- reduction is preprocessing, not our method)")
for k,v in sorted(acc.items(),key=lambda kv:-np.mean(kv[1])):
    if len(v)>=3: print("  %-38s %7.3f +- %.3f" % (k,np.mean(v),np.std(v)))
