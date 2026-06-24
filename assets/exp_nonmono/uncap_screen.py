"""UNCAPPED screen of the non-monotone experiment (cap=(1.0,1.0)) — all methods on equal footing (no capacity, fair to
Kallus). 3 seeds, matched Γ (+ a low Γ for the trend). Report realised test outcome + treated fraction + ceilings."""
import sys, importlib.util
import numpy as np
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
HERE=NEW/"assets"/"exp_nonmono"
sp=importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
def load(n,rel):
    s=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
kal=load("kal","methods/Kallus/kallus.py")
K=dgp.K; NOCAP=(1.0,1.0); mG=dgp.GAMMAS[dgp.mi]
METHODS=["R-OW","R-OW-DR","R-O","IPW","AIPW","Kallus"]
def runG(G,seeds=3):
    A={m:[] for m in METHODS}; FR={m:[] for m in METHODS}; C={"ctrl":[],"bm":[],"orc":[]}
    for s in range(seeds):
        rng=np.random.default_rng(s); tr=dgp.generate(dgp.n_tr,rng); te=dgp.generate(dgp.n_te,rng)
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
            v,f=rt(pi); A[m].append(v); FR[m].append(f)
        th=kal.fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
        kp=kal.predict_kallus(th.theta,te.X,("affine",)); A["Kallus"].append(float((kp*te.Ypot).sum(1).mean())); FR["Kallus"].append(float(kp[:,1].mean()))
        C["ctrl"].append(float(te.Ypot[:,0].mean())); C["orc"].append(float(te.Ypot.max(1).mean())); C["bm"].append(float(te.Ypot[np.arange(dgp.n_te),np.argmax(te.mu,1)].mean()))
    print("Γ=%.2f (UNCAPPED, %d seeds):"%(G,seeds))
    for m in sorted(METHODS,key=lambda m:-np.mean(A[m])):
        print("  %-9s realised %.3f  (treated frac %.2f)"%(m,np.mean(A[m]),np.mean(FR[m])))
    print("  ceilings: never-treat %.3f  best-means %.3f  oracle %.3f"%(np.mean(C["ctrl"]),np.mean(C["bm"]),np.mean(C["orc"])))
for G in [mG,3.0,1.0]: runG(G,3); print()
