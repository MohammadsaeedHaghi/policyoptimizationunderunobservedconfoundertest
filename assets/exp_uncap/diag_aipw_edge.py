"""Why does AIPW give ~0.5 at the edges (uncapped) while R-OW gives 1.0? Inspect, per seed, the edge policies of AIPW vs
R-OW + the fitted outcome model mu_hat that AIPW relies on, at a few X (edges vs middle)."""
import sys, importlib.util
import numpy as np
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
sp=importlib.util.spec_from_file_location("dgp",str(NEW/"assets/exp_nonmono/dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
def load(n,rel):
    s=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
rodr=load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
K=dgp.K; NOCAP=(1.0,1.0); mG=dgp.GAMMAS[dgp.mi]; X=np.sort(dgp.GRID)
show=[0,2,5,10,15,18,20]   # X = -1,-0.8,-0.5,0,0.5,0.8,1
true_t=0.5*(dgp.pa(X,0,1)+dgp.pa(X,1,1)); true_c=0.5*(dgp.pa(X,0,0)+dgp.pa(X,1,0))
print("X            :"+"".join("%7.1f"%X[i] for i in show))
print("true E[Y1|X] :"+"".join("%7.2f"%true_t[i] for i in show))
print("true E[Y0|X] :"+"".join("%7.2f"%true_c[i] for i in show)+"   (true control = 0.50 everywhere)")
MU1=[];MU0=[];PA=[];PR=[]
for s in range(6):
    rng=np.random.default_rng(s); tr=dgp.generate(dgp.n_tr,rng)
    w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
    eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
    byv={}
    for j,xv in enumerate(np.round(tr.X.ravel(),6)): byv.setdefault(round(float(xv),6),j)
    def gridpol(pi):
        pi=np.asarray(pi,float); g=np.full(len(X),np.nan)
        for b,cv in enumerate(X):
            k=round(float(cv),6)
            if k in byv: g[b]=pi[1,byv[k]]
        return g
    # muhat per grid X (take a representative unit per X)
    m1=np.array([muhat[byv[round(float(cv),6)],1] if round(float(cv),6) in byv else np.nan for cv in X])
    m0=np.array([muhat[byv[round(float(cv),6)],0] if round(float(cv),6) in byv else np.nan for cv in X])
    pa=gridpol(rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=NOCAP,discretize=False).pi)
    pr=gridpol(row(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=NOCAP,discretize=False,zscore=False,epsilon=eps).pi)
    MU1.append(m1);MU0.append(m0);PA.append(pa);PR.append(pr)
MU1=np.array(MU1);MU0=np.array(MU0);PA=np.array(PA);PR=np.array(PR)
print("\nmu_hat_1(X)  :"+"".join("%7.2f"%np.nanmean(MU1,0)[i] for i in show)+"   <- AIPW's outcome model for TREAT")
print("mu_hat_0(X)  :"+"".join("%7.2f"%np.nanmean(MU0,0)[i] for i in show)+"   <- AIPW's outcome model for CONTROL")
print("mu1-mu0 (CATE_hat):"+"".join("%7.2f"%(np.nanmean(MU1,0)-np.nanmean(MU0,0))[i] for i in show))
print("\nAIPW policy per seed at the EDGES (X=-1 / X=+1):")
for s in range(6): print("   seed%d: X=-1 -> %.2f   X=+1 -> %.2f"%(s,PA[s,0],PA[s,20]))
print("AIPW mean policy :"+"".join("%7.2f"%np.nanmean(PA,0)[i] for i in show))
print("R-OW mean policy :"+"".join("%7.2f"%np.nanmean(PR,0)[i] for i in show))
