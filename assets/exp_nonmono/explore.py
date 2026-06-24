"""DGP exploration: find a 1-D 2-arm DGP where the free-π robust methods (R-OW, R-OW-DR) beat Kallus, AIPW, IPW.
Hypothesis: a NON-MONOTONE optimal policy (treatment helps a middle band of X) breaks the parametric Kallus
(logistic policy) and AIPW's logistic μ̂, while free-π R-OW captures it; confounding sinks plain IPW.
Evaluates each candidate config at the matched Γ (seed 0) and prints realised test E[Y] per method.
Usage: python3 explore.py   (edit CONFIGS below)"""
import sys, importlib.util, time
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
def make_gen(p):
    # p: dict with ctrl0, BA, BT(threshold^2), Cs(S-effect on treat), BX(X effect on assign), G(S conf), mono(bool)
    def pa(X,S,k):
        X=np.asarray(X,float); S=np.asarray(S,float)
        if k==0: return np.clip(sig(p["ctrl0"]+p.get("ctrlX",0.0)*X),1e-3,1-1e-3)
        core = (p["BA"]*(p["BX_treat"]*X) if p.get("mono") else p["BA"]*(p["BT"]-X**2))
        return np.clip(sig(p.get("t0",0.0)+core+p["Cs"]*(S-0.5)),1e-3,1-1e-3)
    def generate(nn,rng):
        X=rng.choice(GRID,size=nn); S=(rng.uniform(size=nn)<0.5).astype(int)
        Yp=np.column_stack([(rng.uniform(size=nn)<pa(X,S,0)).astype(float),(rng.uniform(size=nn)<pa(X,S,1)).astype(float)])
        U1=-p["BX"]*X+p["G"]*(S-0.5)*(1.0+p.get("EK",0.0)*X**2); p1=np.exp(U1)/(1+np.exp(U1)); T=(rng.uniform(size=nn)<p1).astype(int)
        mu=np.column_stack([0.5*pa(X,0,0)+0.5*pa(X,1,0),0.5*pa(X,0,1)+0.5*pa(X,1,1)])
        from types import SimpleNamespace
        return SimpleNamespace(X=X.reshape(-1,1),S=S,T=T,Y=Yp[np.arange(nn),T],Ypot=Yp,mu=mu),pa
    return generate
def evaluate(p,label):
    gt=p["G"]; mG=round(float(np.exp(gt/2)),4); gen=make_gen(p)
    rng=np.random.default_rng(0); tr,pa=gen(n_tr,rng); te,_=gen(n_te,rng)
    w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
    eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0))
    muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
    uniq={}
    for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
    ukeys=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in ukeys])
    nn_te=np.array([int(np.argmin(np.abs(ukeys-x))) for x in np.round(te.X.ravel(),6)])
    def rt(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_te]; return float((c*te.Ypot.T).sum()/c.shape[1])
    fi=float(te.Ypot.max(1).mean()); bm=float(te.Ypot[np.arange(n_te),np.argmax(te.mu,1)].mean()); base=float(te.Ypot[:,0].mean())
    out={}
    out["R-OW"]=rt(row(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi)
    out["R-O"]=rt(ro(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False).pi)
    out["IPW"]=rt(ipw(tr.X,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False).pi)
    out["AIPW"]=rt(rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi)
    out["R-OW-DR"]=rt(rowdr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi)
    out["R-O-DR"]=rt(rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=mG,cap=CAP,discretize=False).pi)
    th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
    out["Kallus"]=float((kal.predict_kallus(th.theta,te.X,("affine",))*te.Ypot).sum(1).mean())
    # true optimal-treat region (where μ1>μ0 on the marginal grid)
    xs=np.sort(ukeys); mug=np.column_stack([0.5*pa(xs,0,0)+0.5*pa(xs,1,0),0.5*pa(xs,0,1)+0.5*pa(xs,1,1)])
    treatX=xs[mug[:,1]>mug[:,0]]
    mine=max(out["R-OW"],out["R-OW-DR"]); base3=max(out["AIPW"],out["IPW"],out["Kallus"])
    win="WIN" if mine>base3 else "no"
    print(f"\n=== {label} (matched Γ={mG}) === ceilings: oracle={fi:.3f} best-means={bm:.3f} control={base:.3f}")
    print("  true treat-region X∈["+(f"{treatX.min():.2f},{treatX.max():.2f}]" if len(treatX) else "EMPTY]")+f"  ({len(treatX)}/21 cells)")
    print("  "+"  ".join(f"{k}={v:.3f}" for k,v in out.items()))
    print(f"  my-best={mine:.3f}  baselines-best(AIPW/IPW/Kallus)={base3:.3f}  -> {win}  (R-OW over AIPW {out['R-OW']-out['AIPW']:+.3f}, over Kallus {out['R-OW']-out['Kallus']:+.3f}, over IPW {out['R-OW']-out['IPW']:+.3f})")
    return out
CONFIGS=[
 ("edgeconf-EK3",   dict(ctrl0=0.0,BA=7.0,BT=0.36,Cs=5.0,BX=0.0,G=5.0,EK=3.0)),
 ("edgeconf-EK5",   dict(ctrl0=0.0,BA=7.0,BT=0.36,Cs=5.0,BX=0.0,G=5.0,EK=5.0)),
 ("edgeconf-wide",  dict(ctrl0=0.0,BA=6.0,BT=0.42,Cs=5.0,BX=0.0,G=5.0,EK=4.0)),
 ("edgeconf-bigCs", dict(ctrl0=0.0,BA=7.0,BT=0.36,Cs=7.0,BX=0.0,G=5.0,EK=4.0)),
]
if __name__=="__main__":
    t0=time.time()
    for lab,p in CONFIGS:
        try: evaluate(p,lab)
        except Exception as e: print(f"\n=== {lab} FAILED: {str(e)[:100]}")
        print(f"  [{time.time()-t0:.0f}s elapsed]",flush=True)
