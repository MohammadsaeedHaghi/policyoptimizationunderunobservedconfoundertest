"""Prototype: IHDP (real covariates + KNOWN counterfactuals mu0/mu1). Observe ONE covariate (binned to a 1-D grid),
HIDE the rest (unobserved confounders). Capacity cap forces budget allocation. Run the key methods, deploy by bin
to a held-out test set, evaluate realised outcome via the known mu0/mu1. Sanity-check before building the full thing.
Usage: python3 proto_ihdp.py [obs_cov_index] [cap]"""
import sys, importlib.util
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
OBS=int(sys.argv[1]) if len(sys.argv)>1 else 6          # observed covariate index (x6 = main CATE driver)
CAPt=float(sys.argv[2]) if len(sys.argv)>2 else 0.30    # treat at most this fraction
CAP=(1.0,CAPt); K=2; NB=10; GAMMA=2.0                   # 10 bins; Γ=2 sensitivity
DATA=NEW/"assets"/"exp_realdata"/"data"
cols=['treat','yf','ycf','mu0','mu1']+['x%d'%i for i in range(1,26)]
acc={}
for rep in range(1,6):
    d=pd.read_csv(DATA/f"ihdp_{rep}.csv",header=None,names=cols)
    xo=d['x%d'%OBS].values.astype(float); T=d['treat'].values.astype(int); Y=d['yf'].values.astype(float)
    mu0=d['mu0'].values.astype(float); mu1=d['mu1'].values.astype(float)
    # bin observed covariate to NB quantile bins -> grid in [-1,1]
    edges=np.quantile(xo,np.linspace(0,1,NB+1)); edges[0]-=1e-9; b=np.clip(np.digitize(xo,edges)-1,0,NB-1)
    ctr=np.linspace(-1,1,NB); Xg=ctr[b].reshape(-1,1)
    rng=np.random.default_rng(rep); idx=rng.permutation(len(d)); ntr=int(0.6*len(d)); tri,tei=idx[:ntr],idx[ntr:]
    Xtr,Ttr,Ytr=Xg[tri],T[tri],Y[tri]; Xte=Xg[tei]
    w,_=common.ipw_weights_from_data(Xtr,Ttr,K); Dm=common.pairwise_distance_matrix(Xtr)
    eps=tuple(common.tight_epsilon(Dm,Ttr,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(Xtr,Ttr,Ytr,n_arms=K,cross_fit=True)
    # deploy: map each test unit to its train-representative bin policy
    uniq={}
    for j,xv in enumerate(np.round(Xtr.ravel(),6)): uniq.setdefault(float(xv),j)
    uk=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in uk]); nn_te=np.array([int(np.argmin(np.abs(uk-x))) for x in np.round(Xte.ravel(),6)])
    m0te,m1te=mu0[tei],mu1[tei]
    def rt(pi):
        pi=np.asarray(pi,float); pcols=pi[:,uidx][:,nn_te]      # (K, n_te) deployed
        return float((pcols[0]*m0te+pcols[1]*m1te).mean())
    res={}
    res["R-OW"]=rt(row(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi)
    res["R-O"]=rt(ro(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False).pi)
    res["IPW"]=rt(ipw(Xtr,Ttr,Ytr,w,n_arms=K,cap=CAP,discretize=False).pi)
    res["AIPW"]=rt(rodr(Xtr,Ttr,Ytr,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi)
    res["R-OW-DR"]=rt(rowdr(Xtr,Ttr,Ytr,w,muhat,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi)
    th=kal.fit_kallus(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
    res["Kallus"]=float((kal.predict_kallus(th.theta,Xte,("affine",))*np.column_stack([m0te,m1te])).sum(1).mean())
    # ceilings (deployable best by binned-x CATE, with the cap; oracle = per-unit best; control = treat none)
    res["_ctrl"]=float(m0te.mean()); res["_orcUncap"]=float(np.maximum(m0te,m1te).mean())
    # best-on-bin with cap: per bin mean CATE, treat the highest-CATE bins up to cap fraction
    cate_te=m1te-m0te; binte=np.clip(np.digitize(d['x%d'%OBS].values[tei],edges)-1,0,NB-1)
    binmean=np.array([cate_te[binte==bb].mean() if (binte==bb).any() else -9 for bb in range(NB)]); order=np.argsort(-binmean)
    treatbins=set(); frac=0.0
    for bb in order:
        f=(binte==bb).mean()
        if frac+f<=CAPt+1e-9 and binmean[bb]>0: treatbins.add(bb); frac+=f
    res["_bestbin"]=float(np.array([(m1te[i] if binte[i] in treatbins else m0te[i]) for i in range(len(tei))]).mean())
    for k,v in res.items(): acc.setdefault(k,[]).append(v)
    print(f"rep{rep}: "+" ".join(f"{k}={v:.3f}" for k,v in res.items() if not k.startswith('_'))+f" | ctrl={res['_ctrl']:.3f} bestbin={res['_bestbin']:.3f} orc={res['_orcUncap']:.3f}",flush=True)
print("\n=== IHDP 5-rep mean (obs=x%d, cap=%.2f, Γ=%.1f) ==="%(OBS,CAPt,GAMMA))
for k in ["R-OW","R-OW-DR","R-O","IPW","AIPW","Kallus","_ctrl","_bestbin","_orcUncap"]:
    print(f"  {k:10s} {np.mean(acc[k]):.3f} ± {np.std(acc[k]):.3f}")
mine=min(np.mean(acc['R-OW']),np.mean(acc['R-OW-DR'])); base=max(np.mean(acc['IPW']),np.mean(acc['AIPW']),np.mean(acc['Kallus']))
print(f"  -> my worst {mine:.3f} vs best baseline {base:.3f} : {'WIN' if mine>base else 'tie/lose'}")
