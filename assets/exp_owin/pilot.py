"""Pilot: does O-W beat sharp (and everything else) on the multivariate exp_owin DGP?

Sharp is given its FAIREST multivariate form: k-means cells over the full covariate vector,
swept over cell counts, so it is not handicapped by a naive per-axis grid.
"""
import sys, json, importlib.util, time
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent.parent; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"extensions"/"Shapley"))
import common
from shapley import extract_support
from sklearn.cluster import KMeans
sys.path.insert(0,str(ROOT/"assets"/"grand"))
from run_sharp_all import sharp_scores  # (now import-safe: __main__ guard added)
def LD(rel,fn):
    s=importlib.util.spec_from_file_location(fn,str(ROOT/rel)); m=importlib.util.module_from_spec(s); sys.modules[fn]=m; s.loader.exec_module(m); return getattr(m,fn)
S={"IPW-O-X":LD("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py","solve_ipw_o_x_uncapped"),
   "DoublyRobust-O-X":LD("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py","solve_doublyrobust_o_x_uncapped"),
   "Hajek-O-X":LD("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py","solve_hajek_o_x_uncapped"),
   "IPW-O-W":LD("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py","solve_ipw_o_w_uncapped"),
   "DoublyRobust-O-W":LD("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py","solve_doublyrobust_o_w_uncapped")}
import argparse
_ap=argparse.ArgumentParser(); _ap.add_argument("--dgp",default="assets/exp_owin/dgp.py")
_ap.add_argument("--n",type=int,default=1000); _ap.add_argument("--seeds",type=int,default=3)
_ap.add_argument("--workers",type=int,default=3); _a=_ap.parse_args()
sp=importlib.util.spec_from_file_location("d",str(ROOT/_a.dgp)); d=importlib.util.module_from_spec(sp); sys.modules["d"]=d; sp.loader.exec_module(d)
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
N,NTE,SEEDS,G=_a.n,4000,range(_a.seeds),5.0
LS=[None,3.0,1.0]
acc={}
for sd in SEEDS:
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
        km=KMeans(n_clusters=nc,n_init=4,random_state=0).fit(X); ci=km.labels_
        cte=km.predict(Xte)
        m1,m0=sharp_scores(Y,T,ci,nc,G); sc=np.nan_to_num(m1,nan=-1e9)-np.nan_to_num(m0,nan=0.0)
        pol=(sc>0).astype(float); pv=pol[cte]
        acc.setdefault("Sharp-O-X(%d)"%nc,[]).append(float((pv*Y1+(1-pv)*Y0).mean()))
        m1n,m0n=np.zeros(nc),np.zeros(nc)
        for j in range(nc):
            mm=ci==j; yt,yc=Y[mm&(T==1)],Y[mm&(T==0)]
            if len(yt)>1 and len(yc)>1: m1n[j],m0n[j]=yt.mean(),yc.mean()
        pn=((m1n-m0n)>0).astype(float)[cte]
        acc.setdefault("naive-binned(%d)"%nc,[]).append(float((pn*Y1+(1-pn)*Y0).mean()))
    for m in S:
        best=-1e9
        for Lv in LS:
            try:
                if m=="IPW-O-X": r=S[m](X,T,Y,w,n_arms=2,Gamma=G,discretize=False,lipschitz=Lv)
                elif m=="DoublyRobust-O-X": r=S[m](X,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False,lipschitz=Lv)
                elif m=="Hajek-O-X": r=S[m](X,T,Y,wr,n_arms=2,Gamma=G,maximize=True,discretize=False,lipschitz=Lv)
                elif m=="IPW-O-W": r=S[m](X,T,Y,w,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=eps,lipschitz=Lv)
                else: r=S[m](X,T,Y,w,mu,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=eps,lipschitz=Lv)
                sX,spv=extract_support(X,r.pi[1]); pe=shap(Xte,np.asarray(sX,float),np.asarray(spv,float).ravel())
                v=float((pe*Y1+(1-pe)*Y0).mean()); best=max(best,v)
            except Exception as ex: print("FAIL",m,Lv,ex,flush=True)
        acc.setdefault(m,[]).append(best)
    print("seed %d done"%sd,flush=True)
print("\n=== exp_owin pilot [%s]: n=%d, d=%d, matched Gamma=%g, %d seeds (best L per method) ==="%(_a.dgp.split("/")[-1],N,d.D,G,len(list(SEEDS))))
for k,v in sorted(acc.items(),key=lambda kv:-np.mean(kv[1])):
    print("  %-20s %7.3f +- %.3f"%(k,np.mean(v),np.std(v)))
