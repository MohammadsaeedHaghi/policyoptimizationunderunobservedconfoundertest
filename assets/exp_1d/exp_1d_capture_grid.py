"""Capture the per-Γ deployed policy grid π(treat | x) for ONE method on the 1-D 2-arm experiment, so the
interactive policy chart can show the policy learned at EVERY Γ (not just the matched Γ). Reproduces the exact
seed-0 data + capped LP setup of exp_1d.py; gates the matched-Γ grid against the stored value (Δ=0). Writes
assets/exp_1d{,_uniformS}/_bygamma/{mode}_{method}.json = {mode,method,gammas,grids:[6×21]}.
Usage: python3 exp_1d_capture_grid.py {bern|uni} {RW|RO|IPW|AIPW|RegretO|RegretOW|RW_DR|RO_DR|Kallus}"""
import sys, importlib.util, json
import numpy as np
from pathlib import Path
MODE=sys.argv[1]; METHOD=sys.argv[2]
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
ASSETS=(NEW/"assets"/"exp_1d") if MODE=="bern" else (NEW/"assets"/"exp_1d_uniformS")
NPZ="exp_1d_data.npz" if MODE=="bern" else "exp_1d_uniformS_data.npz"
OUT=ASSETS/"_bygamma"; OUT.mkdir(parents=True,exist_ok=True)
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
A,C,B=4.0,5.0,1.0; theta0,theta1=0.5,-1.2; K=2; CAP=(1.0,0.5)
GRID=np.round(np.linspace(-1,1,21),6); S_GRID=np.round(np.arange(0,1.0001,0.1),6); n_tr,n_te=800,2000; seed=0; gt=5.0
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def pa(X,S,k): return np.clip(sig(theta0+0*np.asarray(X,float)) if k==0 else sig(theta1+A*np.asarray(X,float)+C*np.asarray(S,float)),1e-3,1-1e-3)
def generate(nn,gamma,rng):
    X=rng.choice(GRID,size=nn)
    S=(rng.uniform(size=nn)<0.5).astype(int) if MODE=="bern" else rng.choice(S_GRID,size=nn)
    Yp=np.column_stack([(rng.uniform(size=nn)<pa(X,S,0)).astype(float),(rng.uniform(size=nn)<pa(X,S,1)).astype(float)])
    U1=-B*X+gamma*(S-0.5); p1=np.exp(U1)/(1+np.exp(U1)); T=(rng.uniform(size=nn)<p1).astype(int)
    from types import SimpleNamespace
    return SimpleNamespace(X=X.reshape(-1,1),T=T,Y=Yp[np.arange(nn),T],Ypot=Yp)
rng=np.random.default_rng(seed); tr=generate(n_tr,gt,rng); te=generate(n_te,gt,rng)
w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0))
muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
uniq={}
for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
ukeys=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in ukeys]); xs=np.sort(ukeys)
nn_xs=np.array([int(np.argmin(np.abs(ukeys-x))) for x in xs])
def polx_treat(pi): pi=np.asarray(pi,float); return pi[:,uidx][:,nn_xs][1]    # π(treat) on the 21-X grid
d=dict(np.load(ASSETS/NPZ)); GAMMAS=[float(g) for g in d["GAMMAS"]]; mi=4
# solver dispatch
LP=lambda rel,fn:getattr(load(fn,rel),fn)
SOLVE={
 "RW":   lambda G: LP("methods/IPW-O-W/Capped/ipw_o_w_capped.py","solve_ipw_o_w_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,discretize=False,zscore=False,epsilon=eps),
 "RO":   lambda G: LP("methods/IPW-O-X/Capped/ipw_o_x_capped.py","solve_ipw_o_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,discretize=False),
 "RegretO": lambda G: LP("methods/Hajek-O-X/Capped/hajek_o_x_capped.py","solve_hajek_o_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,maximize=True,discretize=False),
 "RegretOW":lambda G: LP("methods/Hajek-O-W/Capped/hajek_o_w_capped.py","solve_hajek_o_w_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,maximize=True,discretize=False,zscore=False,epsilon=eps),
 "RW_DR":lambda G: LP("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py","solve_doublyrobust_o_w_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=CAP,discretize=False,zscore=False,epsilon=eps),
 "RO_DR":lambda G: LP("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py","solve_doublyrobust_o_x_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=CAP,discretize=False),
}
GAMMA_INDEP={"IPW","AIPW"}
def grid_at(method,G):
    if method=="IPW":
        return polx_treat(LP("methods/IPW-X-X/Capped/ipw_x_x_capped.py","solve_ipw_x_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False).pi)
    if method=="AIPW":   # = capped DoublyRobust(=R-O-DR) at Γ=1 (box collapses, no robustness)
        return polx_treat(LP("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py","solve_doublyrobust_o_x_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi)
    if method=="Kallus":
        kal=load("kal","methods/Kallus/kallus.py")
        th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0).theta
        return kal.predict_kallus(th,xs.reshape(-1,1),("affine",))[:,1]
    return polx_treat(SOLVE[method](G).pi)
grids=[]
for gi,G in enumerate(GAMMAS):
    if METHOD in GAMMA_INDEP and gi>0:
        grids.append(grids[0]); continue       # Γ-independent: reuse Γ=1 grid
    try:
        g=grid_at(METHOD,G); grids.append([round(float(v),4) for v in g])
    except Exception as e:
        print(f"  {METHOD} Γ={G} FAIL: {str(e)[:80]}",flush=True); grids.append(None)
# gate: matched-Γ grid must match the stored {METHOD}_treat_grid
stored=d.get(METHOD+"_treat_grid")
gate="n/a"
if stored is not None and grids[mi] is not None:
    delta=float(np.max(np.abs(np.array(grids[mi])-np.array(stored,float))))
    gate=f"{delta:.4f}"; assert delta<1e-3, f"GATE FAIL {METHOD} {MODE}: matched-Γ grid Δ={delta}"
(OUT/f"{MODE}_{METHOD}.json").write_text(json.dumps({"mode":MODE,"method":METHOD,"gammas":GAMMAS,"grids":grids}))
print(f"[{MODE}/{METHOD}] captured {sum(g is not None for g in grids)}/{len(GAMMAS)} Γ-grids; matched-Γ gate Δ={gate} -> {OUT}/{MODE}_{METHOD}.json")
