"""Quick 5-seed sanity: run the key methods at the matched Γ for seeds 0-4, report mean±SD + win verdict."""
import sys, importlib.util
import numpy as np
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
sys.path.insert(0,str(NEW/"assets"/"exp_nonmono")); import dgp
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
kal=load("kal","methods/Kallus/kallus.py")
K=dgp.K; CAP=dgp.CAP; mG=dgp.GAMMAS[dgp.mi]
METH=["R-OW","R-O","IPW","AIPW","R-OW-DR","R-O-DR","Kallus"]
acc={m:[] for m in METH}; ceil={"oracle":[],"best-means":[],"control":[]}
for seed in range(5):
    rng=np.random.default_rng(seed); tr=dgp.generate(dgp.n_tr,rng); te=dgp.generate(dgp.n_te,rng)
    w,_=common.ipw_weights_from_data(tr.X,tr.T,K); Dm=common.pairwise_distance_matrix(tr.X)
    eps=tuple(common.tight_epsilon(Dm,tr.T,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)
    uniq={}
    for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
    ukeys=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in ukeys]); nn_te=np.array([int(np.argmin(np.abs(ukeys-x))) for x in np.round(te.X.ravel(),6)])
    def rt(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_te]; return float((c*te.Ypot.T).sum()/c.shape[1])
    acc["R-OW"].append(rt(row(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
    acc["R-O"].append(rt(ro(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,cap=CAP,discretize=False).pi))
    acc["IPW"].append(rt(ipw(tr.X,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False).pi))
    acc["AIPW"].append(rt(rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi))
    acc["R-OW-DR"].append(rt(rowdr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=mG,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
    acc["R-O-DR"].append(rt(rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=mG,cap=CAP,discretize=False).pi))
    th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=mG,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
    acc["Kallus"].append(float((kal.predict_kallus(th.theta,te.X,("affine",))*te.Ypot).sum(1).mean()))
    ceil["oracle"].append(float(te.Ypot.max(1).mean())); ceil["best-means"].append(float(te.Ypot[np.arange(dgp.n_te),np.argmax(te.mu,1)].mean())); ceil["control"].append(float(te.Ypot[:,0].mean()))
    print(f"  seed{seed} done: R-OW={acc['R-OW'][-1]:.3f} R-OW-DR={acc['R-OW-DR'][-1]:.3f} AIPW={acc['AIPW'][-1]:.3f} IPW={acc['IPW'][-1]:.3f} Kallus={acc['Kallus'][-1]:.3f}",flush=True)
print("\n=== 5-seed mean ± SD @ matched Γ (TEST) ===")
rows=sorted(((m,np.mean(acc[m]),np.std(acc[m])) for m in METH),key=lambda r:-r[1])
for m,mu,sd in rows: print(f"  {m:9s} {mu:.3f} ± {sd:.3f}")
print(f"  ceilings: control {np.mean(ceil['control']):.3f} · best-means {np.mean(ceil['best-means']):.3f} · oracle {np.mean(ceil['oracle']):.3f}")
mine=min(np.mean(acc["R-OW"]),np.mean(acc["R-OW-DR"])); base=max(np.mean(acc["AIPW"]),np.mean(acc["IPW"]),np.mean(acc["Kallus"]))
print(f"  -> R-OW&R-OW-DR worst {mine:.3f} vs best baseline {base:.3f} : {'BOTH WIN' if mine>base else 'NOT both win'}")
