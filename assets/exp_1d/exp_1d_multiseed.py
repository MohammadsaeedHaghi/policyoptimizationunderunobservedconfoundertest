"""Run the 1-D 2-arm method suite for ONE seed, in BOTH regimes (UNCAPPED cap=(1,1) and CAPPED cap=(1,0.5)).
9 methods (RW, RO, IPW, AIPW, RW_DR, RO_DR, RegretO, RegretOW, Kallus) × 6 Γ. Records realised train/test E[Y],
worst-case objective, treat-fraction; capacity-constrained ceilings (Full-info, Best-means); and — for EVERY seed —
the per-Γ treat-probability grid π(treat|X) over the 21-X grid per method/regime, so the interactive chart can pick a
seed (or average policies across seeds). Mirrors the exp_first/run_multiseed.py schema. Reproduces the exact DGP of the
original exp_1d.py. Writes assets/exp_1d{,_uniformS}/_multiseed/{mode}_seed{seed}.json.
Usage: python3 exp_1d_multiseed.py {bern|uni} {seed}"""
import sys, importlib.util, json
import numpy as np
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
MODE=sys.argv[1]; SEED=int(sys.argv[2])
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
ASSETS=(NEW/"assets"/"exp_1d") if MODE=="bern" else (NEW/"assets"/"exp_1d_uniformS")
OUT=ASSETS/"_multiseed"; OUT.mkdir(parents=True,exist_ok=True)
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
A,C,B=4.0,5.0,1.0; theta0,theta1=0.5,-1.2; K=2
REGIMES=[("uncap",(1.0,1.0)),("cap",(1.0,0.5))]
GRID=np.round(np.linspace(-1,1,21),6); S_GRID=np.round(np.arange(0,1.0001,0.1),6); n_tr,n_te=500,2000; gt=5.0
GAMMAS=[1.0,3.0,6.0,9.0,round(float(np.exp(gt/2)),4),16.0]; mi=4; MG=GAMMAS[mi]
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def pa(X,S,k): return np.clip(sig(theta0+0*np.asarray(X,float)) if k==0 else sig(theta1+A*np.asarray(X,float)+C*np.asarray(S,float)),1e-3,1-1e-3)
def generate(nn,gamma,rng):
    X=rng.choice(GRID,size=nn)
    S=(rng.uniform(size=nn)<0.5).astype(int) if MODE=="bern" else rng.choice(S_GRID,size=nn)
    Yp=np.column_stack([(rng.uniform(size=nn)<pa(X,S,0)).astype(float),(rng.uniform(size=nn)<pa(X,S,1)).astype(float)])
    U1=-B*X+gamma*(S-0.5); p1=np.exp(U1)/(1+np.exp(U1)); T=(rng.uniform(size=nn)<p1).astype(int)
    from types import SimpleNamespace
    mu=np.column_stack([0.5*pa(X,0,0)+0.5*pa(X,1,0),0.5*pa(X,0,1)+0.5*pa(X,1,1)]) if MODE=="bern" else \
       np.column_stack([np.mean([pa(X,s,0) for s in S_GRID],0),np.mean([pa(X,s,1) for s in S_GRID],0)])
    return SimpleNamespace(X=X.reshape(-1,1),T=T,Y=Yp[np.arange(nn),T],Ypot=Yp,mu=mu)

def constrained_oracle(X, V, cap):
    """max Σ_i Σ_k π_k(x_i)V[i,k] s.t. per-unit simplex, same-X tying, (1/n)Σ_i π_k ≤ cap_k. Returns (K,n)."""
    Xr=np.round(X.ravel(),6); uniq=np.unique(Xr); n=len(Xr); idx={u:np.where(Xr==u)[0] for u in uniq}
    Vs=np.array([[V[idx[u],k].sum() for k in range(K)] for u in uniq]); cnt=np.array([len(idx[u]) for u in uniq]); ns=len(uniq)
    m=gp.Model(); m.Params.OutputFlag=0; pi=m.addVars(ns,K,lb=0,ub=1)
    m.setObjective(gp.quicksum(pi[s,k]*Vs[s,k] for s in range(ns) for k in range(K)),GRB.MAXIMIZE)
    for s in range(ns): m.addConstr(gp.quicksum(pi[s,k] for k in range(K))==1)
    for k in range(K): m.addConstr(gp.quicksum(cnt[s]*pi[s,k] for s in range(ns))<=cap[k]*n)
    m.optimize(); out=np.zeros((K,n))
    for si,u in enumerate(uniq):
        for k in range(K): out[k,idx[u]]=pi[si,k].X
    return out

rng=np.random.default_rng(SEED); tr=generate(n_tr,gt,rng); te=generate(n_te,gt,rng)
w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0))
muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
# deploy maps: each train col -> a unique-X representative; test/grid -> nearest unique-X
uniq={}
for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
ukeys=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in ukeys])
nn_te=np.array([int(np.argmin(np.abs(ukeys-x))) for x in np.round(te.X.ravel(),6)])
nn_tr=np.array([int(np.argmin(np.abs(ukeys-x))) for x in np.round(tr.X.ravel(),6)])
grid_nn=np.array([int(np.argmin(np.abs(ukeys-x))) for x in GRID])  # 21-pt grid -> unique-X col
def rt_test(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_te]; return float((c*te.Ypot.T).sum()/c.shape[1])
def rt_train(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_tr]; return float((c*tr.Ypot.T).sum()/c.shape[1])
def treat_frac(pi): pi=np.asarray(pi,float); return float(pi[1].mean())
def treat_grid(pi): pi=np.asarray(pi,float); return [round(float(v),4) for v in pi[:,uidx][1,grid_nn]]
LPf=lambda rel,fn:getattr(load(fn,rel),fn)
kal=load("kal","methods/Kallus/kallus.py")
def solve(method,G,CAP):
    if method=="RW": return LPf("methods/IPW-O-W/Capped/ipw_o_w_capped.py","solve_ipw_o_w_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,discretize=False,zscore=False,epsilon=eps)
    if method=="RO": return LPf("methods/IPW-O-X/Capped/ipw_o_x_capped.py","solve_ipw_o_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,discretize=False)
    if method=="IPW": return LPf("methods/IPW-X-X/Capped/ipw_x_x_capped.py","solve_ipw_x_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False)
    if method=="AIPW": return LPf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py","solve_doublyrobust_o_x_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False)
    if method=="RegretO": return LPf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py","solve_hajek_o_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,maximize=True,discretize=False)
    if method=="RegretOW": return LPf("methods/Hajek-O-W/Capped/hajek_o_w_capped.py","solve_hajek_o_w_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,maximize=True,discretize=False,zscore=False,epsilon=eps)
    if method=="RW_DR": return LPf("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py","solve_doublyrobust_o_w_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=CAP,discretize=False,zscore=False,epsilon=eps)
    if method=="RO_DR": return LPf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py","solve_doublyrobust_o_x_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=CAP,discretize=False)
METH=["RW","RO","IPW","AIPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]; GVAR=[m for m in METH if m not in ("IPW","AIPW","Kallus")]
out={"mode":MODE,"seed":SEED,"gammas":GAMMAS,"matched_gamma":MG,"grid":GRID.tolist(),"regimes":{}}
for rname,CAP in REGIMES:
    R={"cap":list(CAP),"methods":{m:{} for m in METH}}
    fi=constrained_oracle(te.X,te.Ypot,CAP); bm=constrained_oracle(te.X,te.mu,CAP)
    fitr=constrained_oracle(tr.X,tr.Ypot,CAP); bmtr=constrained_oracle(tr.X,tr.mu,CAP)
    R["ceilings"]={"full_info_test":float((fi*te.Ypot.T).sum()/te.Ypot.shape[0]),
                   "best_means_test":float((bm*te.mu.T).sum()/te.mu.shape[0]),
                   "full_info_train":float((fitr*tr.Ypot.T).sum()/tr.Ypot.shape[0]),
                   "best_means_train":float((bmtr*tr.mu.T).sum()/tr.mu.shape[0])}
    # Γ-independent baselines solved once
    pi_ipw=solve("IPW",1.0,CAP).pi; m_ipw=(rt_train(pi_ipw),rt_test(pi_ipw),treat_frac(pi_ipw))
    pi_ai=solve("AIPW",1.0,CAP).pi; m_ai=(rt_train(pi_ai),rt_test(pi_ai),treat_frac(pi_ai))
    g_ipw=treat_grid(pi_ipw); g_ai=treat_grid(pi_ai)
    gFI=[round(float(v),4) for v in fitr[:,uidx][1,grid_nn]]; gBM=[round(float(v),4) for v in bmtr[:,uidx][1,grid_nn]]
    grids={}
    for gi,G in enumerate(GAMMAS):
        R["methods"]["IPW"][gi]={"rt_train":m_ipw[0],"rt_test":m_ipw[1],"treat":m_ipw[2],"obj":float("nan")}
        R["methods"]["AIPW"][gi]={"rt_train":m_ai[0],"rt_test":m_ai[1],"treat":m_ai[2],"obj":float("nan")}
        g={"IPW":g_ipw,"AIPW":g_ai,"Full-info":gFI,"Best-means":gBM}
        for m in GVAR:
            try:
                sv=solve(m,float(G),CAP)
                R["methods"][m][gi]={"rt_train":rt_train(sv.pi),"rt_test":rt_test(sv.pi),"treat":treat_frac(sv.pi),"obj":float(getattr(sv,"objective_value",float("nan")))}
                g[m]=treat_grid(sv.pi)
            except Exception as e:
                R["methods"][m][gi]={"rt_train":None,"rt_test":None,"treat":None,"obj":None}; g[m]=None
        # Kallus (free logistic policy; cannot enforce cap — reported with caveat)
        try:
            th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=float(G),maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
            pi_tr=kal.predict_kallus(th.theta,tr.X,("affine",)); pi_te=kal.predict_kallus(th.theta,te.X,("affine",))
            R["methods"]["Kallus"][gi]={"rt_train":float((pi_tr*tr.Ypot).sum(1).mean()),"rt_test":float((pi_te*te.Ypot).sum(1).mean()),"treat":float(pi_tr[:,1].mean()),"obj":float(th.objective_value)}
            g["Kallus"]=[round(float(v),4) for v in kal.predict_kallus(th.theta,GRID.reshape(-1,1),("affine",))[:,1]]
        except Exception as e:
            R["methods"]["Kallus"][gi]={"rt_train":None,"rt_test":None,"treat":None,"obj":None}; g["Kallus"]=None
        grids[gi]=g
        print("[%s/seed%d] %s Γ=%.2f | R-OW=%.3f IPW=%.3f AIPW=%.3f RW_DR=%.3f"%(MODE,SEED,rname,G,
              R["methods"]["RW"][gi]["rt_test"] or float("nan"),m_ipw[1],m_ai[1],R["methods"]["RW_DR"][gi]["rt_test"] or float("nan")),flush=True)
    R["grids"]=grids
    out["regimes"][rname]=R
(OUT/f"{MODE}_seed{SEED}.json").write_text(json.dumps(out))
print("[%s/seed%d] wrote %s_seed%d.json (matchedΓ=%.4f)"%(MODE,SEED,MODE,SEED,MG),flush=True)
