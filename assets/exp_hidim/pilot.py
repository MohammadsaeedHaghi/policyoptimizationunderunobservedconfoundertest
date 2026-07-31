"""exp_hidim pilot: n=200, d=8. Sharp is capped at 0.664 by its cell-oracle ceiling; can we clear it?"""
import sys, importlib.util, numpy as np
from sklearn.cluster import KMeans
ROOT="/home1/haghim/code 1.1"; sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/extensions/Shapley"); sys.path.insert(0,ROOT+"/assets/grand")
import common
from shapley import extract_support
from run_sharp_all import sharp_scores
def LD(rel,fn):
    s=importlib.util.spec_from_file_location(fn,ROOT+"/"+rel); m=importlib.util.module_from_spec(s); sys.modules[fn]=m; s.loader.exec_module(m); return getattr(m,fn)
S={"IPW-O-X":LD("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py","solve_ipw_o_x_uncapped"),
   "DoublyRobust-O-X":LD("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py","solve_doublyrobust_o_x_uncapped"),
   "Hajek-O-X":LD("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py","solve_hajek_o_x_uncapped"),
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
LS=[None,5.0,3.0,1.0]
acc={}
for sd in range(5):
    obs,_=d.generate(N,sd); X,T,Y=obs["X"],obs["T"],obs["Y"]
    te,ft=d.generate(NTE,sd+1000); Xte=te["X"]; Y1,Y0=ft["Y1"],ft["Y0"]
    w,_=common.ipw_weights_from_data(X,T,2); wr,_=common.ipw_weights_from_data(X,T,2,normalize=False)
    mu=common.outcome_means(X,T,Y,n_arms=2,cross_fit=True)
    Dm=common.pairwise_distance_matrix(X); eps=tuple(common.tight_epsilon(Dm,T,w,2,is_distance=True,c_eps=1.0))
    orc=d.oracle_policy(Xte); acc.setdefault("_oracle",[]).append(float((orc*Y1+(1-orc)*Y0).mean()))
    acc.setdefault("_never",[]).append(float(Y0.mean()))
    nv=(mu[:,1]-mu[:,0]>0).astype(float); sX,spv=extract_support(X,nv)
    pe=shap(Xte,np.asarray(sX,float),np.asarray(spv,float).ravel())
    acc.setdefault("naive-linear",[]).append(float((pe*Y1+(1-pe)*Y0).mean()))
    for nc in (15,30,60):
        km=KMeans(n_clusters=nc,n_init=4,random_state=0).fit(X); ci=km.labels_; cte=km.predict(Xte)
        m1,m0=sharp_scores(Y,T,ci,nc,G); sc=np.nan_to_num(m1,nan=-1e9)-np.nan_to_num(m0,nan=0.0)
        p=(sc>0).astype(float)[cte]
        acc.setdefault("Sharp-O-X(%d)"%nc,[]).append(float((p*Y1+(1-p)*Y0).mean()))
        m1n,m0n=np.zeros(nc),np.zeros(nc)
        for j in range(nc):
            mm=ci==j; yt,yc=Y[mm&(T==1)],Y[mm&(T==0)]
            if len(yt)>1 and len(yc)>1: m1n[j],m0n[j]=yt.mean(),yc.mean()
        pn=((m1n-m0n)>0).astype(float)[cte]
        acc.setdefault("naive-binned(%d)"%nc,[]).append(float((pn*Y1+(1-pn)*Y0).mean()))
    # k-means-CELL arms: our methods given sharp's exact partition (same 30 cells),
    # so both sides have ~30 policy parameters and the SAME cell-oracle ceiling. The only
    # remaining difference is the uncertainty set: box+Wasserstein vs box+per-cell calibration.
    for nc in (15,30):
        km=KMeans(n_clusters=nc,n_init=4,random_state=0).fit(X); ci=km.labels_; cte=km.predict(Xte)
        Xb=km.cluster_centers_[ci]                     # snap to centroid -> tie_groups pools pi
        Db=common.pairwise_distance_matrix(Xb)
        try: epsb=tuple(common.tight_epsilon(Db,T,w,2,is_distance=True,c_eps=1.0))
        except Exception as ex:
            print("eps FAIL kmeans%d"%nc,ex,flush=True); continue
        for m in ("IPW-O-W","DoublyRobust-O-W","IPW-O-X","DoublyRobust-O-X"):
            try:
                if m=="IPW-O-X": r=S[m](Xb,T,Y,w,n_arms=2,Gamma=G,discretize=False)
                elif m=="DoublyRobust-O-X": r=S[m](Xb,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False)
                elif m=="IPW-O-W": r=S[m](Xb,T,Y,w,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=epsb)
                else: r=S[m](Xb,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=epsb)
                pi=np.asarray(r.pi[1],float)
                cellpi=np.array([pi[ci==j].mean() if (ci==j).any() else 0.0 for j in range(nc)])
                pe=cellpi[cte]
                acc.setdefault("%s [kmeans%d]"%(m,nc),[]).append(float((pe*Y1+(1-pe)*Y0).mean()))
            except Exception as ex: print("FAIL kmeans%d"%nc,m,ex,flush=True)
    for m in S:
        best=-9e9
        for Lv in LS:
            try:
                if m=="IPW-O-X": r=S[m](X,T,Y,w,n_arms=2,Gamma=G,discretize=False,lipschitz=Lv)
                elif m=="DoublyRobust-O-X": r=S[m](X,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False,lipschitz=Lv)
                elif m=="Hajek-O-X": r=S[m](X,T,Y,wr,n_arms=2,Gamma=G,maximize=True,discretize=False,lipschitz=Lv)
                elif m=="IPW-O-W": r=S[m](X,T,Y,w,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=eps,lipschitz=Lv)
                else: r=S[m](X,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=eps,lipschitz=Lv)
                sX,spv=extract_support(X,r.pi[1]); pe=shap(Xte,np.asarray(sX,float),np.asarray(spv,float).ravel())
                best=max(best,float((pe*Y1+(1-pe)*Y0).mean()))
            except Exception as ex: print("FAIL",m,Lv,ex,flush=True)
        acc.setdefault(m,[]).append(best)
    print("seed",sd,"done",flush=True)
print("\n=== exp_hidim pilot: n=%d, d=%d, matched Gamma=%g, 5 seeds (best L) ===" % (N,d.D,G))
print("    cell-oracle CEILING for sharp: 0.664 (30 cells, infinite data)")
for k,v in sorted(acc.items(),key=lambda kv:-np.mean(kv[1])):
    print("  %-22s %7.3f +- %.3f" % (k,np.mean(v),np.std(v)))
