"""Explore v3: S-dependent control Y(0) + harder confounding, to widen R-OW/R-OW-DR margin over AIPW/IPW.
Y(0)=σ(B0+D0·X+CS0·(S-½)); Y(1)=σ(BA·(BT-X²)+CS1·(S-½)) (non-monotone band); T~σ(BX·X+G(S-½)(1+EK·X²)).
Making BOTH arms S-confounded inverts the observed effect at the edges (where high-S treated look good and low-S
controls look bad) ⇒ IPW/AIPW are fooled harder; R-OW box+Wasserstein hedges. 2 seeds/config, key methods @ matched Γ."""
import sys, importlib.util
import numpy as np
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
kal=load("kal","methods/Kallus/kallus.py")
GRID=np.round(np.linspace(-1,1,21),6); n_tr,n_te=800,2000; K=2; CAP=(1.0,0.5)
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def gen(p,seed):
    def pa(X,S,k):
        X=np.asarray(X,float); S=np.asarray(S,float)
        if k==0: return np.clip(sig(p["B0"]+p["D0"]*X+p["CS0"]*(S-0.5)),1e-3,1-1e-3)
        return np.clip(sig(p["BA"]*(p["BT"]-X**2)+p["CS1"]*(S-0.5)),1e-3,1-1e-3)
    rng=np.random.default_rng(seed)
    def g(nn):
        X=rng.choice(GRID,size=nn); S=(rng.uniform(size=nn)<0.5).astype(int)
        Yp=np.column_stack([(rng.uniform(size=nn)<pa(X,S,0)).astype(float),(rng.uniform(size=nn)<pa(X,S,1)).astype(float)])
        U1=p["BX"]*X+p["G"]*(S-0.5)*(1.0+p["EK"]*X**2); p1=np.exp(U1)/(1+np.exp(U1)); T=(rng.uniform(size=nn)<p1).astype(int)
        mu=np.column_stack([0.5*pa(X,0,0)+0.5*pa(X,1,0),0.5*pa(X,0,1)+0.5*pa(X,1,1)])
        from types import SimpleNamespace
        return SimpleNamespace(X=X.reshape(-1,1),T=T,Y=Yp[np.arange(nn),T],Ypot=Yp,mu=mu)
    return g(n_tr),g(n_te)
def es(p,seed):
    tr,te=gen(p,seed); mG=round(float(np.exp(p["G"]/2)),4)
    w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
    eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
    uniq={}
    for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
    uk=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in uk]); nn_te=np.array([int(np.argmin(np.abs(uk-x))) for x in np.round(te.X.ravel(),6)])
    def rt(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_te]; return float((c*te.Ypot.T).sum()/c.shape[1])
    o={}
    o["R-OW"]=rt(row(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi)
    o["R-OW-DR"]=rt(rowdr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi)
    o["R-O"]=rt(ro(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False).pi)
    o["IPW"]=rt(ipw(tr.X,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False).pi)
    o["AIPW"]=rt(rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi)
    th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
    o["Kallus"]=float((kal.predict_kallus(th.theta,te.X,("affine",))*te.Ypot).sum(1).mean())
    o["_bm"]=float(te.Ypot[np.arange(n_te),np.argmax(te.mu,1)].mean()); o["_ctrl"]=float(te.Ypot[:,0].mean()); o["_orc"]=float(te.Ypot.max(1).mean())
    return o
CONFIGS=[
 ("v1:ctrlS-CS5",  dict(B0=0.0,D0=0.0,CS0=5.0,CS1=5.0,BA=7.0,BT=0.36,BX=0.0,G=5.0,EK=5.0)),
 ("v2:CS6-G6-EK6", dict(B0=0.0,D0=0.0,CS0=6.0,CS1=6.0,BA=8.0,BT=0.34,BX=0.0,G=6.0,EK=6.0)),
 ("v3:CS7-EK8",    dict(B0=0.0,D0=0.0,CS0=7.0,CS1=6.0,BA=8.0,BT=0.34,BX=0.0,G=6.0,EK=8.0)),
 ("v4:CS6-G7",     dict(B0=0.0,D0=0.0,CS0=6.0,CS1=6.0,BA=8.0,BT=0.36,BX=0.0,G=7.0,EK=6.0)),
]
for lab,p in CONFIGS:
    A={k:[] for k in ["R-OW","R-OW-DR","R-O","IPW","AIPW","Kallus","_bm","_ctrl","_orc"]}
    for s in range(1):
        try:
            o=es(p,s)
            for k in A: A[k].append(o[k])
        except Exception as e: print(f"{lab} s{s} FAIL {str(e)[:60]}",flush=True)
    M={k:np.mean(v) for k,v in A.items() if v}
    mine=min(M["R-OW"],M["R-OW-DR"]); base=max(M["AIPW"],M["IPW"],M["Kallus"])
    print(f"\n{lab} (2 seeds, mG={round(float(np.exp(p['G']/2)),2)}): R-OW={M['R-OW']:.3f} R-OW-DR={M['R-OW-DR']:.3f} R-O={M['R-O']:.3f} | IPW={M['IPW']:.3f} AIPW={M['AIPW']:.3f} Kallus={M['Kallus']:.3f} | ctrl={M['_ctrl']:.3f} bm={M['_bm']:.3f} orc={M['_orc']:.3f}")
    print(f"   R-OW over AIPW={M['R-OW']-M['AIPW']:+.3f} IPW={M['R-OW']-M['IPW']:+.3f} Kallus={M['R-OW']-M['Kallus']:+.3f} | Wedge={M['R-OW']-M['R-O']:+.3f} -> {'WIN' if mine>base else 'no'}",flush=True)
