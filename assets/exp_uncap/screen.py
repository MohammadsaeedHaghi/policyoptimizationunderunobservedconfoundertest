"""Screen for an UNCAPPED 1-D 2-arm DGP where R-OW beats Kallus, AIPW, IPW (no capacity → fair to Kallus).
Idea: CLEAN control (stable worst-case baseline) + non-monotone treatment whose HIGH-S units look good even at the
harmful edges (fools IPW/AIPW), + edge-concentrated S-selection. Then R-OW's worst-case treated value should fall
BELOW the clean control at the edges → it avoids them without a cap. Tunable; reports realised value UNCAPPED."""
import sys, importlib.util
import numpy as np
from types import SimpleNamespace
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
def load(n,rel):
    s=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
kal=load("kal","methods/Kallus/kallus.py")
GRID=np.round(np.linspace(-1,1,21),6); n_tr,n_te=800,2000; K=2; NOCAP=(1.0,1.0)
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def make(p):
    def Y0(X,S): return np.full(np.asarray(X,float).shape, sig(p["B0"]))          # CLEAN control (no X/S)
    def Y1(X,S): X=np.asarray(X,float); S=np.asarray(S,float); return np.clip(sig(p["BA"]*(p["BT"]-X**2)+p["CS1"]*(S-0.5)),1e-3,1-1e-3)
    def gen(nn,rng):
        X=rng.choice(GRID,size=nn); S=(rng.uniform(size=nn)<0.5).astype(float)
        p0=Y0(X,S); p1=Y1(X,S)
        Yp=np.column_stack([(rng.uniform(size=nn)<p0).astype(float),(rng.uniform(size=nn)<p1).astype(float)])
        U1=p["G"]*(S-0.5)*(1.0+p["EK"]*X**2); T=(rng.uniform(size=nn)<sig(U1)).astype(int)
        mu=np.column_stack([Y0(X,0), 0.5*Y1(X,0)+0.5*Y1(X,1)])
        return SimpleNamespace(X=X.reshape(-1,1),S=S,T=T,Y=Yp[np.arange(nn),T],Ypot=Yp,mu=mu)
    return gen
def screen(lab,p,G,seeds=2):
    gen=make(p); M={m:[] for m in ["R-OW","R-OW-DR","R-O","IPW","AIPW","Kallus"]}; FR={m:[] for m in M}; C={"ctrl":[],"bm":[],"orc":[]}
    for s in range(seeds):
        rng=np.random.default_rng(s); tr=gen(n_tr,rng); te=gen(n_te,rng)
        w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
        eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
        uniq={}
        for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
        uk=np.array(sorted(uniq)); uidx=np.array([uniq[k] for k in uk]); nn=np.array([int(np.argmin(np.abs(uk-x))) for x in np.round(te.X.ravel(),6)])
        def rt(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn]; return float((c*te.Ypot.T).sum()/c.shape[1]), float(pi[1].mean())
        for m,pi in [("R-OW",row(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=NOCAP,discretize=False,zscore=False,epsilon=eps).pi),
                     ("R-OW-DR",rowdr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=NOCAP,discretize=False,zscore=False,epsilon=eps).pi),
                     ("R-O",ro(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=NOCAP,discretize=False).pi),
                     ("IPW",ipw(tr.X,tr.T,tr.Y,w,n_arms=K,cap=NOCAP,discretize=False).pi),
                     ("AIPW",rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=NOCAP,discretize=False).pi)]:
            v,f=rt(pi); M[m].append(v); FR[m].append(f)
        th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
        kp=kal.predict_kallus(th.theta,te.X,("affine",)); M["Kallus"].append(float((kp*te.Ypot).sum(1).mean())); FR["Kallus"].append(float(kp[:,1].mean()))
        C["ctrl"].append(float(te.Ypot[:,0].mean())); C["orc"].append(float(te.Ypot.max(1).mean())); C["bm"].append(float(te.Ypot[np.arange(n_te),np.argmax(te.mu,1)].mean()))
    mn={m:np.mean(M[m]) for m in M}; mine=max(mn["R-OW"],mn["R-OW-DR"]); base=max(mn["IPW"],mn["AIPW"],mn["Kallus"])
    print("[%s] Γ=%.2f UNCAPPED: "%(lab,G)+" ".join("%s=%.3f(%.2f)"%(m,mn[m],np.mean(FR[m])) for m in ["R-OW","R-OW-DR","R-O","IPW","AIPW","Kallus"]))
    print("    ctrl %.3f bm %.3f orc %.3f | R-OW over IPW=%+.3f AIPW=%+.3f Kallus=%+.3f -> %s"%(np.mean(C["ctrl"]),np.mean(C["bm"]),np.mean(C["orc"]),mn["R-OW"]-mn["IPW"],mn["R-OW"]-mn["AIPW"],mn["R-OW"]-mn["Kallus"],"WIN" if mine>base else "no"),flush=True)
for lab,p in [
 ("A B0=0 CS1=8 BA=6 G=5 EK=5", dict(B0=0.0,CS1=8.,BA=6.,BT=0.36,G=5.,EK=5.)),
 ("B B0=0 CS1=10 BA=7 G=6 EK=6",dict(B0=0.0,CS1=10.,BA=7.,BT=0.34,G=6.,EK=6.)),
 ("C B0=0.2 CS1=9 BA=6 G=5 EK=8",dict(B0=0.2,CS1=9.,BA=6.,BT=0.36,G=5.,EK=8.)),
]:
    for G in [round(float(np.exp(p["G"]/2)),3), round(float(np.exp(p["G"]/2*1.6)),3)]:
        screen(lab,p,G,seeds=2)
    print()
