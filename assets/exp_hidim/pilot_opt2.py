"""Option 2: CROSS-VALIDATED selection of representation and policy class -- for BOTH sides.

Option 1 showed the best configuration is not fixed: our best was idx1/free-pi (0.772) while
sharp's was idx2/15-cells (0.898), and in 1-D at n=500 the ordering of binned vs free-pi reversed
entirely. So the coarseness of the policy class is a knob that must be CHOSEN, not set.

Selection rule (identical for every candidate, ours and sharp's): split train 50/50, learn the
policy on fold A, and score it on fold B by the SHARP LOWER BOUND of its value (the two-point
Dorn-Guo closed form on a fixed 10-cell held-out partition). That is a valid confounding-aware
criterion computable without counterfactuals, it needs no LP, and it is the same yardstick for
everyone. The winner is refit on the full sample.
"""
import sys, importlib.util, numpy as np
from sklearn.cluster import KMeans
ROOT="/home1/haghim/code 1.1"
sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/extensions/Shapley"); sys.path.insert(0,ROOT+"/assets/grand"); sys.path.insert(0,ROOT+"/assets/exp_hidim")
import common
from shapley import extract_support
from run_sharp_all import sharp_scores, sharp_min_arm
from reduce import index_directions, reduce_X
def LD(rel,fn):
    s=importlib.util.spec_from_file_location(fn,ROOT+"/"+rel); m=importlib.util.module_from_spec(s); sys.modules[fn]=m; s.loader.exec_module(m); return getattr(m,fn)
SOL={"IPW-O-W":LD("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py","solve_ipw_o_w_uncapped"),
     "DoublyRobust-O-W":LD("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py","solve_doublyrobust_o_w_uncapped"),
     "IPW-O-X":LD("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py","solve_ipw_o_x_uncapped")}
sp=importlib.util.spec_from_file_location("d",ROOT+"/assets/exp_hidim/dgp.py"); d=importlib.util.module_from_spec(sp); sys.modules["d"]=d; sp.loader.exec_module(d)
G,N,NTE=10.0,200,4000

def holdout_score(Zb, Tb, Yb, pib, ncell=10):
    """Sharp lower bound of the value of policy pib on held-out fold (common criterion)."""
    nb=len(Yb); k=min(ncell,max(2,nb//10))
    km=KMeans(n_clusters=k,n_init=3,random_state=0).fit(np.atleast_2d(Zb)); ci=km.labels_
    tot=0.0
    for j in range(k):
        m=ci==j
        if m.sum()==0: continue
        e=float(np.clip(Tb[m].mean(),0.05,0.95))
        for arm,wh in ((1,1.0/e),(0,1.0/(1-e))):
            sel=m&(Tb==arm)
            if sel.sum()==0: continue
            pw=pib[sel] if arm==1 else (1.0-pib[sel])
            tot+=sharp_min_arm(Yb[sel]*pw, wh, G)*(sel.sum()/nb)*(m.sum()/max(sel.sum(),1))*0+ \
                 sharp_min_arm(Yb[sel]*pw, wh, G)*(sel.sum()/nb)
    return float(tot)

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

acc={}
for sd in range(5):
    obs,_=d.generate(N,sd); X,T,Y=obs["X"],obs["T"],obs["Y"]
    te,ft=d.generate(NTE,sd+1000); Xte=te["X"]; Y1,Y0=ft["Y1"],ft["Y0"]
    rng=np.random.default_rng(1000+sd); perm=rng.permutation(N); A,B=perm[:N//2],perm[N//2:]
    mu=common.outcome_means(X,T,Y,n_arms=2,cross_fit=True); w,_=common.ipw_weights_from_data(X,T,2)
    b1,b2=index_directions(X,T,Y,mu=mu)
    orc=d.oracle_policy(Xte)
    acc.setdefault("_oracle",[]).append(float((orc*Y1+(1-orc)*Y0).mean()))
    reps={"idx1":reduce_X(X,Xte,[b1]),"idx2":reduce_X(X,Xte,[b1,b2])}
    # ---- candidate set (identical structure for ours and sharp) ----
    cand_ours,cand_sharp=[],[]
    for rep,(Z,Zte) in reps.items():
        for cls in ("free","km15","km30"):
            cand_ours.append((rep,Z,Zte,cls))
        for nc in (10,15,30):
            cand_sharp.append((rep,Z,Zte,nc))
    def fit_ours(Z,Zte,cls,idx,m):
        Zt=Z[idx]; Tt=T[idx]; Yt=Y[idx]; mut=mu[idx]
        wt,_=common.ipw_weights_from_data(Zt,Tt,2)
        if cls.startswith("km"):
            nc=min(int(cls[2:]),max(2,len(idx)//4))
            km=KMeans(n_clusters=nc,n_init=3,random_state=0).fit(Zt)
            Zt2=km.cluster_centers_[km.labels_]
        else: Zt2=Zt
        Dm=common.pairwise_distance_matrix(Zt2); eps=tuple(common.tight_epsilon(Dm,Tt,wt,2,is_distance=True,c_eps=1.0))
        if m=="IPW-O-W": r=SOL[m](Zt2,Tt,Yt,wt,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=eps)
        elif m=="DoublyRobust-O-W": r=SOL[m](Zt2,Tt,Yt,wt,mut,n_arms=2,Gamma=G,discretize=False,zscore=False,epsilon=eps)
        else: r=SOL[m](Zt2,Tt,Yt,wt,n_arms=2,Gamma=G,discretize=False)
        sX,spv=extract_support(Zt2,r.pi[1])
        return lambda Q: shap(Q,np.asarray(sX,float),np.asarray(spv,float).ravel())
    for m in ("IPW-O-W","DoublyRobust-O-W","IPW-O-X"):
        scored=[]
        for rep,Z,Zte,cls in cand_ours:
            try:
                f=fit_ours(Z,Zte,cls,A,m); piB=f(Z[B])
                scored.append((holdout_score(Z[B],T[B],Y[B],piB),rep,Z,Zte,cls))
            except Exception as ex: print("cvFAIL",m,rep,cls,ex,flush=True)
        if not scored: continue
        _,rep,Z,Zte,cls=max(scored)
        try:
            f=fit_ours(Z,Zte,cls,np.arange(N),m); pe=f(Zte)
            acc.setdefault("CV | %s"%m,[]).append(float((pe*Y1+(1-pe)*Y0).mean()))
            acc.setdefault("_pick_%s"%m,[]).append(hash((rep,cls))%1)
            print("  seed%d %s -> picked %s/%s"%(sd,m,rep,cls),flush=True)
        except Exception as ex: print("refitFAIL",m,ex,flush=True)
    # sharp with the same CV
    scored=[]
    for rep,Z,Zte,nc in cand_sharp:
        try:
            km=KMeans(n_clusters=nc,n_init=3,random_state=0).fit(Z[A]); ci=km.labels_
            m1,m0=sharp_scores(Y[A],T[A],ci,nc,G); sc=np.nan_to_num(m1,nan=-1e9)-np.nan_to_num(m0,nan=0.0)
            pol=(sc>0).astype(float); piB=pol[km.predict(Z[B])]
            scored.append((holdout_score(Z[B],T[B],Y[B],piB),rep,Z,Zte,nc))
        except Exception as ex: print("cvFAIL sharp",rep,nc,ex,flush=True)
    if scored:
        _,rep,Z,Zte,nc=max(scored)
        km=KMeans(n_clusters=nc,n_init=3,random_state=0).fit(Z); ci=km.labels_
        m1,m0=sharp_scores(Y,T,ci,nc,G); sc=np.nan_to_num(m1,nan=-1e9)-np.nan_to_num(m0,nan=0.0)
        p=(sc>0).astype(float)[km.predict(Zte)]
        acc.setdefault("CV | Sharp-O-X",[]).append(float((p*Y1+(1-p)*Y0).mean()))
        print("  seed%d sharp -> picked %s/%d"%(sd,rep,nc),flush=True)
    print("seed",sd,"done",flush=True)
print("\n=== OPTION 2: CV-selected representation+class, n=200 d=8, 5 seeds ===")
for k,v in sorted(acc.items(),key=lambda kv:-np.mean(kv[1])):
    if not k.startswith("_pick"): print("  %-26s %7.3f +- %.3f" % (k,np.mean(v),np.std(v)))
