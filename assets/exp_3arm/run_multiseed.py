"""Run the 3-arm exp_a (γ=5) experiment for ONE seed: 8 methods (all except Kallus) × Γ sweep, per-arm capacity from
train shares. Records realised train/test, expected E[Y] (on μ), worst-case objective, per-arm capacity usage; plus the
two capacity-constrained ceilings (Full-info on Ypot, Best-means on μ). On seed 0 also saves per-grid policies at matched Γ.
Saves _multiseed/seed{seed}.json.  Usage: python3 run_multiseed.py {seed}"""
import sys, json, importlib.util
import numpy as np
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
SEED=int(sys.argv[1])
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
HERE=NEW/"assets"/"exp_3arm"
sp=importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
def load(n,rel):
    s=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
rego =load("rego","methods/Hajek-O-X/Capped/hajek_o_x_capped.py").solve_hajek_o_x_capped
regow=load("regow","methods/Hajek-O-W/Capped/hajek_o_w_capped.py").solve_hajek_o_w_capped
K=dgp.K; GAMMAS=dgp.GAMMAS; mi=dgp.mi
GVAR=["R-OW","R-O","R-OW-DR","R-O-DR","Regret-O","Hajek-OW"]; GFIX=["IPW","AIPW"]; METHODS=GVAR+GFIX
def constrained_oracle(X,V,cap):
    """max Σ_i Σ_k π_k(x_i) V[i,k]  s.t. per-unit simplex, same-X tying, (1/n)Σ_i π_k ≤ cap_k. Returns (K,n)."""
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
rng=np.random.default_rng(SEED); tr=dgp.generate(dgp.n_tr,rng); te=dgp.generate(dgp.n_te,rng)
cap=dgp.cap_from_train(tr.T)
w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
# deploy: map each test/grid unit to a representative train column at the same X (discrete X => exact)
byv={}
for j,xv in enumerate(np.round(tr.X.ravel(),6)): byv.setdefault(round(float(xv),6),j)
uk=np.array(sorted(byv)); nn_te=np.array([byv[round(float(x),6)] for x in np.round(te.X.ravel(),6)])
GRIDcol=np.array([byv[round(float(x),6)] for x in dgp.GRID])     # train col per grid X
def metrics(pi):
    pi=np.asarray(pi,float)
    rt_tr=float((pi*tr.Ypot.T).sum()/pi.shape[1]); ex_tr=float((pi*tr.mu.T).sum()/pi.shape[1])
    pte=pi[:,nn_te]; rt_te=float((pte*te.Ypot.T).sum()/pte.shape[1]); ex_te=float((pte*te.mu.T).sum()/pte.shape[1])
    return rt_tr,rt_te,ex_tr,ex_te,[float(u) for u in pi.mean(1)]
out={"seed":SEED,"gammas":GAMMAS,"cap":[float(c) for c in cap],"methods":{m:{} for m in METHODS}}
# constrained ceilings (capacity-constrained oracles)
fi_tr=constrained_oracle(tr.X,tr.Ypot,cap); fi_te=constrained_oracle(te.X,te.Ypot,cap)
bm_tr=constrained_oracle(tr.X,tr.mu,cap);   bm_te=constrained_oracle(te.X,te.mu,cap)
out["ceilings"]={"full_info_train":float((fi_tr*tr.Ypot.T).sum()/tr.Ypot.shape[0]),
                 "full_info_test":float((fi_te*te.Ypot.T).sum()/te.Ypot.shape[0]),
                 "best_means_train":float((bm_tr*tr.mu.T).sum()/tr.mu.shape[0]),
                 "best_means_test":float((bm_te*te.mu.T).sum()/te.mu.shape[0])}
# Γ-independent methods (once)
pi_ipw=ipw(tr.X,tr.T,tr.Y,w,n_arms=K,cap=cap,discretize=False); m_ipw=metrics(pi_ipw.pi); o_ipw=float(getattr(pi_ipw,'objective_value',np.nan))
pi_ai =rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=cap,discretize=False); m_ai=metrics(pi_ai.pi); o_ai=float(getattr(pi_ai,'objective_value',np.nan))
polgrid={}; polgrid_byG={}                                       # seed-0 per-grid policy: matched Γ + per-Γ (for the interactive chart)
def rec(m,res):
    mm=metrics(res.pi); ob=float(getattr(res,'objective_value',np.nan))
    out["methods"][m][gi]={"rt_train":mm[0],"rt_test":mm[1],"exp_train":mm[2],"exp_test":mm[3],"obj":ob,"use":mm[4]}
    return res
for gi,G in enumerate(GAMMAS):
    out["methods"]["IPW"][gi]={"rt_train":m_ipw[0],"rt_test":m_ipw[1],"exp_train":m_ipw[2],"exp_test":m_ipw[3],"obj":o_ipw,"use":m_ipw[4]}
    out["methods"]["AIPW"][gi]={"rt_train":m_ai[0],"rt_test":m_ai[1],"exp_train":m_ai[2],"exp_test":m_ai[3],"obj":o_ai,"use":m_ai[4]}
    R={}
    R["R-OW"]=rec("R-OW",row(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=cap,discretize=False,zscore=False,epsilon=eps))
    R["R-O"]=rec("R-O",ro(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=cap,discretize=False))
    R["R-OW-DR"]=rec("R-OW-DR",rowdr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=cap,discretize=False,zscore=False,epsilon=eps))
    R["R-O-DR"]=rec("R-O-DR",rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=cap,discretize=False))
    R["Regret-O"]=rec("Regret-O",rego(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,maximize=True,cap=cap,discretize=False))
    R["Hajek-OW"]=rec("Hajek-OW",regow(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,maximize=True,cap=cap,discretize=False,zscore=False,epsilon=eps))
    if gi==mi and SEED==0:
        for m in GVAR: polgrid[m]=np.asarray(R[m].pi,float)[:,GRIDcol].tolist()
        polgrid["IPW"]=np.asarray(pi_ipw.pi,float)[:,GRIDcol].tolist(); polgrid["AIPW"]=np.asarray(pi_ai.pi,float)[:,GRIDcol].tolist()
        polgrid["Full-info"]=fi_tr[:,GRIDcol].tolist(); polgrid["Best-means"]=bm_tr[:,GRIDcol].tolist()
    if SEED==0: polgrid_byG[gi]={m:np.asarray(R[m].pi,float)[:,GRIDcol].tolist() for m in GVAR}   # per-Γ GVAR policies
    print("seed%d Γ=%.2f done (R-OW rt_te=%.3f, IPW=%.3f, AIPW=%.3f)"%(SEED,G,out["methods"]["R-OW"][gi]["rt_test"],m_ipw[1],m_ai[1]),flush=True)
if SEED==0:
    out["policy_grid"]=polgrid; out["grid"]=dgp.GRID.tolist(); out["policy_grid_byG"]=polgrid_byG
    out["policy_grid_fixed"]={"IPW":np.asarray(pi_ipw.pi,float)[:,GRIDcol].tolist(),"AIPW":np.asarray(pi_ai.pi,float)[:,GRIDcol].tolist(),
                              "Full-info":fi_tr[:,GRIDcol].tolist(),"Best-means":bm_tr[:,GRIDcol].tolist()}
OUT=HERE/"_multiseed"; OUT.mkdir(exist_ok=True); (OUT/("seed%d.json"%SEED)).write_text(json.dumps(out))
print("wrote seed%d.json | cap=%s"%(SEED,tuple(round(c,3) for c in cap)),flush=True)
