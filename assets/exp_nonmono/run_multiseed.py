"""Run the full 9-method suite × 6 Γ for ONE seed of the non-monotone DGP. Saves _multiseed/seed{seed}.json
with rt/rt_train/obj per method per Γ + per-seed ceilings. On seed 0 also saves per-Γ policy grids (treat_grid
per method per Γ) + xs, for the interactive policy chart. Usage: python3 run_multiseed.py {seed}"""
import sys, importlib.util, json
import numpy as np
from pathlib import Path
SEED=int(sys.argv[1])
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
HERE=NEW/"assets"/"exp_nonmono"; sys.path.insert(0,str(HERE)); import dgp
OUT=HERE/"_multiseed"; OUT.mkdir(parents=True,exist_ok=True)
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
LPf=lambda rel,fn:getattr(load(fn,rel),fn)
kal=load("kal","methods/Kallus/kallus.py")
K=dgp.K; CAP=dgp.CAP; GAMMAS=dgp.GAMMAS; mi=dgp.mi
rng=np.random.default_rng(SEED); tr=dgp.generate(dgp.n_tr,rng); te=dgp.generate(dgp.n_te,rng)
w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
uniq={}
for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
ukeys=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in ukeys]); xs=np.sort(ukeys)
nn_xs=np.array([int(np.argmin(np.abs(ukeys-x))) for x in xs])
nn_te=np.array([int(np.argmin(np.abs(ukeys-x))) for x in np.round(te.X.ravel(),6)])
nn_tr=np.array([int(np.argmin(np.abs(ukeys-x))) for x in np.round(tr.X.ravel(),6)])
def rt_test(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_te]; return float((c*te.Ypot.T).sum()/c.shape[1])
def rt_train(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_tr]; return float((c*tr.Ypot.T).sum()/c.shape[1])
def grid(pi): pi=np.asarray(pi,float); return [round(float(v),4) for v in pi[:,uidx][:,nn_xs][1]]
def solve(m,G):
    if m=="RW": return LPf("methods/IPW-O-W/Capped/ipw_o_w_capped.py","solve_ipw_o_w_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,discretize=False,zscore=False,epsilon=eps)
    if m=="RO": return LPf("methods/IPW-O-X/Capped/ipw_o_x_capped.py","solve_ipw_o_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,discretize=False)
    if m=="IPW": return LPf("methods/IPW-X-X/Capped/ipw_x_x_capped.py","solve_ipw_x_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False)
    if m=="AIPW": return LPf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py","solve_doublyrobust_o_x_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False)
    if m=="RegretO": return LPf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py","solve_hajek_o_x_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,maximize=True,discretize=False)
    if m=="RegretOW": return LPf("methods/Hajek-O-W/Capped/hajek_o_w_capped.py","solve_hajek_o_w_capped")(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,maximize=True,discretize=False,zscore=False,epsilon=eps)
    if m=="RW_DR": return LPf("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py","solve_doublyrobust_o_w_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=CAP,discretize=False,zscore=False,epsilon=eps)
    if m=="RO_DR": return LPf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py","solve_doublyrobust_o_x_capped")(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=G,cap=CAP,discretize=False)
METH=["RW","RO","IPW","AIPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]; INDEP={"IPW","AIPW"}
res={}
for m in METH:
    rtt=[];rttr=[];obj=[];grids=[];sv0=None
    for gi,G in enumerate(GAMMAS):
        try:
            if m=="Kallus":
                th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=float(G),maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
                pi_tr=kal.predict_kallus(th.theta,tr.X,("affine",)); pi_te=kal.predict_kallus(th.theta,te.X,("affine",))
                rtt.append(float((pi_te*te.Ypot).sum(1).mean())); rttr.append(float((pi_tr*tr.Ypot).sum(1).mean())); obj.append(float(th.objective_value))
                if SEED==0: grids.append([round(float(v),4) for v in kal.predict_kallus(th.theta,xs.reshape(-1,1),("affine",))[:,1]])
            else:
                sv=sv0 if (m in INDEP and sv0 is not None) else solve(m,float(G)); sv0=sv if m in INDEP else None
                rtt.append(rt_test(sv.pi)); rttr.append(rt_train(sv.pi)); obj.append(float(sv.objective_value))
                if SEED==0: grids.append(grid(sv.pi))
        except Exception as e:
            rtt.append(None);rttr.append(None);obj.append(None)
            if SEED==0: grids.append(None)
    res[m]={"rt":rtt,"rt_train":rttr,"obj":obj}
    if SEED==0: res[m]["grids"]=grids
ceil={"full_info":float(te.Ypot.max(1).mean()),"full_info_train":float(tr.Ypot.max(1).mean()),
      "best_means":float(te.Ypot[np.arange(dgp.n_te),np.argmax(te.mu,1)].mean()),
      "best_means_train":float(tr.Ypot[np.arange(dgp.n_tr),np.argmax(tr.mu,1)].mean())}
payload={"seed":SEED,"gammas":GAMMAS,"methods":res,"ceilings":ceil}
if SEED==0: payload["xs"]=[round(float(v),4) for v in xs]
(OUT/f"seed{SEED}.json").write_text(json.dumps(payload))
print(f"[seed{SEED}] done. rt_test@matchedΓ: "+" ".join(f"{m}={res[m]['rt'][mi]:.3f}" for m in METH if res[m]['rt'][mi] is not None))
