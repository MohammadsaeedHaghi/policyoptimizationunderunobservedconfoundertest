"""Option 3 pilot: linear (halfspace) policy class via MILP, on raw 8-D and on the indices.

Validation first: the returned policy must actually BE a halfspace in the covariate handed in
(i.e. perfectly separable by some beta), otherwise the MILP linkage is wrong.
"""
import sys, importlib.util, time, numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
ROOT="/home1/haghim/code 1.1"
sys.path.insert(0,ROOT); sys.path.insert(0,ROOT+"/extensions/Shapley"); sys.path.insert(0,ROOT+"/assets/grand"); sys.path.insert(0,ROOT+"/assets/exp_hidim")
import common
from run_sharp_all import sharp_scores
from reduce import index_directions, reduce_X
def LD(rel,fn):
    s=importlib.util.spec_from_file_location(fn,ROOT+"/"+rel); m=importlib.util.module_from_spec(s); sys.modules[fn]=m; s.loader.exec_module(m); return getattr(m,fn)
OW=LD("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py","solve_ipw_o_w_uncapped")
OX=LD("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py","solve_ipw_o_x_uncapped")
sp=importlib.util.spec_from_file_location("d",ROOT+"/assets/exp_hidim/dgp.py"); d=importlib.util.module_from_spec(sp); sys.modules["d"]=d; sp.loader.exec_module(d)
G,N,NTE=10.0,200,4000
acc={}; sep=[]
for sd in range(5):
    obs,_=d.generate(N,sd); X,T,Y=obs["X"],obs["T"],obs["Y"]
    te,ft=d.generate(NTE,sd+1000); Xte=te["X"]; Y1,Y0=ft["Y1"],ft["Y0"]
    mu=common.outcome_means(X,T,Y,n_arms=2,cross_fit=True); w,_=common.ipw_weights_from_data(X,T,2)
    b1,b2=index_directions(X,T,Y,mu=mu)
    orc=d.oracle_policy(Xte); acc.setdefault("_oracle",[]).append(float((orc*Y1+(1-orc)*Y0).mean()))
    acc.setdefault("_never",[]).append(float(Y0.mean()))
    for rep,(Z,Zte) in (("raw",(X,Xte)),("idx1",reduce_X(X,Xte,[b1])),("idx2",reduce_X(X,Xte,[b1,b2]))):
        Dm=common.pairwise_distance_matrix(Z); eps=tuple(common.tight_epsilon(Dm,T,w,2,is_distance=True,c_eps=1.0))
        for nm,fn,kw in (("IPW-O-W",OW,dict(zscore=False,epsilon=eps)),("IPW-O-X",OX,{})):
            try:
                t0=time.time()
                r=fn(Z,T,Y,w,n_arms=2,Gamma=G,discretize=False,linear_policy=True,**kw)
                pi=np.asarray(r.pi[1],float); lab=(pi>0.5).astype(int)
                # validation: is the learned labelling actually a halfspace in Z?
                if len(set(lab.tolist()))>1:
                    acc_sep=LogisticRegression(C=1e6,max_iter=5000).fit(Z,lab).score(Z,lab)
                else: acc_sep=1.0
                sep.append(acc_sep)
                # deploy: refit the separating halfspace, apply to test
                if len(set(lab.tolist()))>1:
                    clf=LogisticRegression(C=1e6,max_iter=5000).fit(Z,lab)
                    pe=clf.predict(Zte).astype(float)
                else: pe=np.full(len(Zte),float(lab[0]))
                acc.setdefault("%s | %s [linear]"%(rep,nm),[]).append(float((pe*Y1+(1-pe)*Y0).mean()))
                if sd==0: print("  %s %s linear: %.0fs, separability %.3f, treats %.2f"%(rep,nm,time.time()-t0,acc_sep,pe.mean()),flush=True)
            except Exception as ex: print("FAIL",rep,nm,ex,flush=True)
        for nc in (15,):
            km=KMeans(n_clusters=nc,n_init=4,random_state=0).fit(Z)
            m1,m0=sharp_scores(Y,T,km.labels_,nc,G); sc=np.nan_to_num(m1,nan=-1e9)-np.nan_to_num(m0,nan=0.0)
            p=(sc>0).astype(float)[km.predict(Zte)]
            acc.setdefault("%s | Sharp-O-X(%d)"%(rep,nc),[]).append(float((p*Y1+(1-p)*Y0).mean()))
    print("seed",sd,"done",flush=True)
print("\n=== OPTION 3: linear policy class (MILP), n=200 d=8, 5 seeds ===")
print("    halfspace validation: mean separability %.3f (1.000 = the MILP really returns a halfspace)"%np.mean(sep))
for k,v in sorted(acc.items(),key=lambda kv:-np.mean(kv[1])):
    if len(v)>=3: print("  %-32s %7.3f +- %.3f" % (k,np.mean(v),np.std(v)))
