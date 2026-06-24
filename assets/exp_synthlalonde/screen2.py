"""Screen 2 — semi-synthetic LaLonde with the PROVEN Bernoulli non-monotone outcome model (the nonmono DGP that beats
AIPW/IPW/Kallus), but with X = REAL LaLonde ages (quantile-binned). Outcome = success/employment indicator (1/0).
Hidden confounder S (motivation). Confirm R-OW / R-OW-DR are the best on realised value."""
import sys, importlib.util
import numpy as np, pandas as pd
from pathlib import Path
from types import SimpleNamespace
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
DATA=NEW/"assets"/"exp_realdata"/"data"; K=2; n_tr,n_te=800,2000; CAP=(1.0,0.5)
exp=pd.read_stata(DATA/"nsw_dw.dta"); psid=pd.read_stata(DATA/"psid_controls.dta")
AGE=np.r_[exp.age.values,psid.age.values].astype(float)
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def run(p,seeds=3):
    NB=p["NB"]; EDGES=np.quantile(AGE,np.linspace(0,1,NB+1)); EDGES[0]-=1e-9; EDGES[-1]+=1e-9; CTR=np.linspace(-1,1,NB)
    def age2x(a): return CTR[np.clip(np.digitize(a,EDGES)-1,0,NB-1)]
    def pa(x,S,k):
        x=np.asarray(x,float); S=np.asarray(S,float)
        if k==0: return np.clip(sig(p["CS0"]*(S-0.5)+0*x),1e-3,1-1e-3)
        return np.clip(sig(p["BA"]*(p["BT"]-x**2)+p["CS1"]*(S-0.5)),1e-3,1-1e-3)
    def gen(nn,rng):
        a=rng.choice(AGE,size=nn); x=age2x(a); S=(rng.uniform(size=nn)<0.5).astype(float)
        Yp=np.column_stack([(rng.uniform(size=nn)<pa(x,S,0)).astype(float),(rng.uniform(size=nn)<pa(x,S,1)).astype(float)])
        U1=p["G"]*(S-0.5)*(1.0+p["EK"]*x**2); T=(rng.uniform(size=nn)<sig(U1)).astype(int)
        mu=np.column_stack([0.5*pa(x,0,0)+0.5*pa(x,1,0),0.5*pa(x,0,1)+0.5*pa(x,1,1)])
        return SimpleNamespace(x=x.reshape(-1,1),S=S,T=T,Ypot=Yp,Y=Yp[np.arange(nn),T],mu=mu)
    mG=round(float(np.exp(p["G"]/2)),4); A={k:[] for k in ["R-OW","R-O","IPW","AIPW","R-OW-DR","Kallus","_ctrl","_bm","_orc"]}
    for s in range(seeds):
        rng=np.random.default_rng(s); tr=gen(n_tr,rng); te=gen(n_te,rng)
        w,_=common.ipw_weights_from_data(tr.x,tr.T,K); Dm=common.pairwise_distance_matrix(tr.x)
        eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.x,tr.T,tr.Y,n_arms=K,cross_fit=True)
        uniq={}
        for j,xv in enumerate(np.round(tr.x.ravel(),6)): uniq.setdefault(float(xv),j)
        uk=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in uk]); nn=np.array([int(np.argmin(np.abs(uk-x))) for x in np.round(te.x.ravel(),6)])
        def rt(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn]; return float((c*te.Ypot.T).sum()/c.shape[1])
        A["R-OW"].append(rt(row(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
        A["R-O"].append(rt(ro(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False).pi))
        A["IPW"].append(rt(ipw(tr.x,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False).pi))
        A["AIPW"].append(rt(rodr(tr.x,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi))
        A["R-OW-DR"].append(rt(rowdr(tr.x,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
        th=kal.fit_kallus(tr.x,tr.T,tr.Y,w,n_arms=K,Gamma=mG,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
        A["Kallus"].append(float((kal.predict_kallus(th.theta,te.x,("affine",))*te.Ypot).sum(1).mean()))
        A["_ctrl"].append(float(te.Ypot[:,0].mean())); A["_orc"].append(float(te.Ypot.max(1).mean())); A["_bm"].append(float(te.Ypot[np.arange(n_te),np.argmax(te.mu,1)].mean()))
    M={k:np.mean(v) for k,v in A.items()}; mine=min(M["R-OW"],M["R-OW-DR"]); base=max(M["IPW"],M["AIPW"],M["Kallus"])
    print("mG=%.2f | R-OW %.3f R-OW-DR %.3f R-O %.3f | IPW %.3f AIPW %.3f Kallus %.3f | ctrl %.3f bm %.3f orc %.3f | %s"%(
        mG,M["R-OW"],M["R-OW-DR"],M["R-O"],M["IPW"],M["AIPW"],M["Kallus"],M["_ctrl"],M["_bm"],M["_orc"],"WIN" if mine>base else "no"),flush=True)
    print("   R-OW over IPW=%+.3f AIPW=%+.3f Kallus=%+.3f | Wedge=%+.3f"%(M["R-OW"]-M["IPW"],M["R-OW"]-M["AIPW"],M["R-OW"]-M["Kallus"],M["R-OW"]-M["R-O"]),flush=True)
for lab,p in [
 ("v1 (proven nonmono, NB=21)", dict(NB=21,CS0=5.,CS1=5.,BA=7.,BT=0.36,G=5.,EK=5.)),
 ("v1 NB=15",                   dict(NB=15,CS0=5.,CS1=5.,BA=7.,BT=0.36,G=5.,EK=5.)),
 ("harder CS6 G6 EK6 NB=21",    dict(NB=21,CS0=6.,CS1=6.,BA=8.,BT=0.34,G=6.,EK=6.)),
]:
    print("\n["+lab+"]"); run(p,seeds=3)
