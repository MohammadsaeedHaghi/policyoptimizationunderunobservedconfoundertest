"""IHDP real-data experiment — real covariates (25 infant-health features) + KNOWN counterfactuals mu0/mu1
(the standard ACIC-style semi-synthetic benchmark; we use the public CEVAE/NPCI reps since the raw ACIC-2016
download is unavailable). HONEST framing: real covariate distribution, synthetic-but-known potential outcomes,
so we can measure the TRUE realised value of each learned policy.

  • OBSERVE one covariate x6 (the strongest CATE driver), binned to a 1-D grid; HIDE x1..x5,x7..x25 (unobserved confounders).
  • No capacity cap (fair to the parametric Kallus, which cannot enforce one).
  • Deploy each policy by bin to a held-out test split; realised value = mean over test of π·[mu0,mu1].
  • 10 replications as seeds (mean±SD).

FINDING (reported honestly): IHDP's treatment assignment is only weakly confounded by covariates (corr(T,x)≈0.1) and
its effect is positive for ~99% of units, so the optimal policy is "treat almost everyone" and ALL methods land near
the oracle and near each other. This validates that the methods run correctly on real-world covariates, but this
benchmark's weak confounding does not separate them — the contrast lives in the strongly-confounded LaLonde study and
the designed synthetic DGPs. Saves ihdp_results.json. Usage: python3 ihdp_run.py"""
import sys, json, importlib.util
import numpy as np, pandas as pd
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
DATA=NEW/"assets"/"exp_realdata"/"data"; K=2; NB=10; OBS=6; CAP=(1.0,1.0); GAMMA=2.0
cols=['treat','yf','ycf','mu0','mu1']+['x%d'%i for i in range(1,26)]
METHODS=["R-OW","R-O","IPW","AIPW","R-OW-DR","Kallus"]
acc={m:[] for m in METHODS}; refs={"ctrl":[],"treatall":[],"oracle":[],"bestbin":[]}
conf=[]  # confounding diagnostic: max |corr(T, x_j)| over hidden covariates
for rep in range(1,11):
    d=pd.read_csv(DATA/f"ihdp_{rep}.csv",header=None,names=cols)
    xo=d['x%d'%OBS].values.astype(float); T=d['treat'].values.astype(int); Y=d['yf'].values.astype(float)
    mu0=d['mu0'].values.astype(float); mu1=d['mu1'].values.astype(float)
    Xall=d[['x%d'%i for i in range(1,26)]].values.astype(float)
    cmax=max(abs(np.corrcoef(T,Xall[:,j])[0,1]) for j in range(25)); conf.append(float(cmax))
    edges=np.quantile(xo,np.linspace(0,1,NB+1)); edges[0]-=1e-9; b=np.clip(np.digitize(xo,edges)-1,0,NB-1)
    ctr=np.linspace(-1,1,NB); Xg=ctr[b].reshape(-1,1)
    rng=np.random.default_rng(rep); idx=rng.permutation(len(d)); ntr=int(0.6*len(d)); tri,tei=idx[:ntr],idx[ntr:]
    Xtr,Ttr,Ytr=Xg[tri],T[tri],Y[tri]; Xte=Xg[tei]; m0,m1=mu0[tei],mu1[tei]
    w,_=common.ipw_weights_from_data(Xtr,Ttr,K); Dm=common.pairwise_distance_matrix(Xtr)
    eps=tuple(common.tight_epsilon(Dm,Ttr,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(Xtr,Ttr,Ytr,n_arms=K,cross_fit=True)
    uniq={}
    for j,xv in enumerate(np.round(Xtr.ravel(),6)): uniq.setdefault(float(xv),j)
    uk=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in uk]); nn=np.array([int(np.argmin(np.abs(uk-x))) for x in np.round(Xte.ravel(),6)])
    def rt(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn]; return float((c[0]*m0+c[1]*m1).mean())
    acc["R-OW"].append(rt(row(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
    acc["R-O"].append(rt(ro(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False).pi))
    acc["IPW"].append(rt(ipw(Xtr,Ttr,Ytr,w,n_arms=K,cap=CAP,discretize=False).pi))
    acc["AIPW"].append(rt(rodr(Xtr,Ttr,Ytr,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi))
    acc["R-OW-DR"].append(rt(rowdr(Xtr,Ttr,Ytr,w,muhat,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
    th=kal.fit_kallus(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
    acc["Kallus"].append(float((kal.predict_kallus(th.theta,Xte,("affine",))*np.column_stack([m0,m1])).sum(1).mean()))
    refs["ctrl"].append(float(m0.mean())); refs["treatall"].append(float(m1.mean()))
    refs["oracle"].append(float(np.maximum(m0,m1).mean()))
    binte=np.clip(np.digitize(d['x%d'%OBS].values[tei],edges)-1,0,NB-1); cate=m1-m0
    bm=np.array([cate[binte==bb].mean() if (binte==bb).any() else -9 for bb in range(NB)])
    refs["bestbin"].append(float(np.array([(m1[i] if bm[binte[i]]>0 else m0[i]) for i in range(len(tei))]).mean()))
    print(f"rep{rep}: "+" ".join(f"{m}={acc[m][-1]:.3f}" for m in METHODS)+f" | conf(maxcorr)={cmax:.3f} oracle={refs['oracle'][-1]:.3f}",flush=True)
out={"methods":{m:{"mean":float(np.mean(acc[m])),"sd":float(np.std(acc[m])),"per_rep":acc[m]} for m in METHODS},
     "refs":{k:float(np.mean(v)) for k,v in refs.items()},"conf_maxcorr":float(np.mean(conf)),
     "obs_cov":OBS,"n_bins":NB,"gamma":GAMMA,"n_reps":10}
(NEW/"assets"/"exp_realdata"/"ihdp_results.json").write_text(json.dumps(out,indent=1))
print("\n=== IHDP 10-rep mean (obs=x%d, no cap, Γ=%.1f) ==="%(OBS,GAMMA))
for m in METHODS: print(f"  {m:9s} {out['methods'][m]['mean']:.3f} ± {out['methods'][m]['sd']:.3f}")
print("  refs:",{k:round(v,3) for k,v in out["refs"].items()},"| mean confounding maxcorr=%.3f"%out["conf_maxcorr"])
