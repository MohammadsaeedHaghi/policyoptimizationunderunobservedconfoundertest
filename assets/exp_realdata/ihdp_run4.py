"""IHDP variant — observe only 4 BINARY features, hide the rest (per user request: "turn all continuous and 15 of the
discrete features into unobserved confounders, rerun with the 4 remaining"). So the methods see a 4-bit covariate vector
(up to 16 cells); the 21 hidden covariates = all 6 continuous (x1..x6, incl. the dominant CATE driver x6) + 15 of the 19
binary features. The 4 observed binary features are the ones most predictive of the treatment effect — consistent with how
the original single-covariate version chose x6 ("strongest CATE driver"). Everything else (no cap, 10 reps, Γ=2, 60/40 split,
realised value via the known mu0/mu1, scale-free fraction-of-oracle-gain metric) matches ihdp_run.py.
Saves ihdp4_results.json. Usage: python3 ihdp_run4.py"""
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
DATA=NEW/"assets"/"exp_realdata"/"data"; K=2; CAP=(1.0,1.0); GAMMA=2.0
cols=['treat','yf','ycf','mu0','mu1']+['x%d'%i for i in range(1,26)]
# --- classify features + pick the 4 observed binary features (top mean|corr(CATE,x)| over reps) ---
d1=pd.read_csv(DATA/"ihdp_1.csv",header=None,names=cols)
CONT=[j for j in range(1,26) if d1['x%d'%j].nunique()>2]; BINF=[j for j in range(1,26) if d1['x%d'%j].nunique()<=2]
cc={j:[] for j in BINF}
for rep in range(1,11):
    d=pd.read_csv(DATA/f"ihdp_{rep}.csv",header=None,names=cols); cate=(d.mu1-d.mu0).values
    for j in BINF: cc[j].append(abs(np.corrcoef(cate,d['x%d'%j].values.astype(float))[0,1]))
OBS=sorted(sorted(BINF,key=lambda j:-np.mean(cc[j]))[:4])              # 4 observed binary features
HID=[j for j in range(1,26) if j not in OBS]                          # 21 hidden
print(f"observed 4 binary features: {['x%d'%j for j in OBS]}  | hidden 21 (all {len(CONT)} continuous + {len(HID)-len(CONT)} binary)",flush=True)
METHODS=["R-OW","R-O","IPW","AIPW","R-OW-DR","Kallus"]
acc={m:[] for m in METHODS}; refs={"ctrl":[],"treatall":[],"oracle":[],"bestcell":[]}
frac={m:[] for m in METHODS}; confH=[]; confAll=[]
for rep in range(1,11):
    d=pd.read_csv(DATA/f"ihdp_{rep}.csv",header=None,names=cols)
    X=d[['x%d'%j for j in OBS]].values.astype(float)                  # (n,4) binary observed
    T=d['treat'].values.astype(int); Y=d['yf'].values.astype(float)
    mu0=d['mu0'].values.astype(float); mu1=d['mu1'].values.astype(float)
    confH.append(max(abs(np.corrcoef(T,d['x%d'%j].values.astype(float))[0,1]) for j in HID))
    confAll.append(max(abs(np.corrcoef(T,d['x%d'%j].values.astype(float))[0,1]) for j in range(1,26)))
    rng=np.random.default_rng(rep); idx=rng.permutation(len(d)); ntr=int(0.6*len(d)); tri,tei=idx[:ntr],idx[ntr:]
    Xtr,Ttr,Ytr=X[tri],T[tri],Y[tri]; Xte=X[tei]; m0,m1=mu0[tei],mu1[tei]
    w,_=common.ipw_weights_from_data(Xtr,Ttr,K); Dm=common.pairwise_distance_matrix(Xtr)
    eps=tuple(common.tight_epsilon(Dm,Ttr,w,K,is_distance=True,c_eps=1.0)); muhat=common.outcome_means(Xtr,Ttr,Ytr,n_arms=K,cross_fit=True)
    # deploy: map each test unit to a representative train unit sharing its 4-bit cell (else nearest by Hamming)
    seen={}
    for j,r in enumerate(np.round(Xtr,6)): seen.setdefault(tuple(r),j)
    keys=np.array(list(seen.keys())); kcol=np.array([seen[tuple(k)] for k in keys])
    nn=np.array([int(kcol[np.argmin(((keys-r)**2).sum(1))]) for r in np.round(Xte,6)])
    def rt(pi): pi=np.asarray(pi,float); c=pi[:,nn]; return float((c[0]*m0+c[1]*m1).mean())
    acc["R-OW"].append(rt(row(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
    acc["R-O"].append(rt(ro(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False).pi))
    acc["IPW"].append(rt(ipw(Xtr,Ttr,Ytr,w,n_arms=K,cap=CAP,discretize=False).pi))
    acc["AIPW"].append(rt(rodr(Xtr,Ttr,Ytr,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False).pi))
    acc["R-OW-DR"].append(rt(rowdr(Xtr,Ttr,Ytr,w,muhat,n_arms=K,Gamma=GAMMA,cap=CAP,discretize=False,zscore=False,epsilon=eps).pi))
    th=kal.fit_kallus(Xtr,Ttr,Ytr,w,n_arms=K,Gamma=GAMMA,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
    acc["Kallus"].append(float((kal.predict_kallus(th.theta,Xte,("affine",))*np.column_stack([m0,m1])).sum(1).mean()))
    c=float(m0.mean()); o=float(np.maximum(m0,m1).mean()); ta=float(m1.mean())
    refs["ctrl"].append(c); refs["treatall"].append(ta); refs["oracle"].append(o)
    # best deployable per observed-cell: treat a cell iff its mean CATE>0
    cate=m1-m0; cellid={tuple(r):i for i,r in enumerate(np.unique(np.round(Xte,6),axis=0))}
    cid=np.array([cellid[tuple(r)] for r in np.round(Xte,6)])
    cm=np.array([cate[cid==i].mean() for i in range(len(cellid))])
    refs["bestcell"].append(float(np.array([(m1[k] if cm[cid[k]]>0 else m0[k]) for k in range(len(tei))]).mean()))
    for m in METHODS: frac[m].append((acc[m][-1]-c)/(o-c) if o>c else np.nan)
    print(f"rep{rep}: "+" ".join(f"{m}={acc[m][-1]:.3f}" for m in METHODS)+f" | ctrl={c:.3f} oracle={o:.3f} confH={confH[-1]:.3f}",flush=True)
out={"methods":{m:{"mean":float(np.mean(acc[m])),"sd":float(np.std(acc[m])),"per_rep":acc[m]} for m in METHODS},
     "refs":{k:float(np.mean(v)) for k,v in refs.items()},
     "conf_maxcorr_hidden":float(np.mean(confH)),"conf_maxcorr_all":float(np.mean(confAll)),
     "obs_features":['x%d'%j for j in OBS],"n_obs":4,"n_hidden":len(HID),"n_continuous_hidden":len(CONT),
     "gamma":GAMMA,"n_reps":10,
     "normalized":{m:{"frac_mean":float(np.nanmean(frac[m])),"frac_sd":float(np.nanstd(frac[m])),"frac_per_rep":[float(x) for x in frac[m]]} for m in METHODS}}
out["normalized"]["_treatall_frac"]=float(np.nanmean([(refs["treatall"][i]-refs["ctrl"][i])/(refs["oracle"][i]-refs["ctrl"][i]) for i in range(10)]))
(NEW/"assets"/"exp_realdata"/"ihdp4_results.json").write_text(json.dumps(out,indent=1))
print("\n=== IHDP 4-feature (obs=%s, 21 hidden, no cap, Γ=%.1f, 10 reps) ==="%(out["obs_features"],GAMMA))
print("fraction of oracle gain captured (scale-free):")
for m in sorted(METHODS,key=lambda m:-out["normalized"][m]["frac_mean"]):
    print(f"  {m:9s} {out['normalized'][m]['frac_mean']*100:5.1f}% ± {out['normalized'][m]['frac_sd']*100:.1f}%")
print(f"  treat-everyone baseline {out['normalized']['_treatall_frac']*100:.1f}%")
print(f"  confounding: max|corr(T, hidden x)|={out['conf_maxcorr_hidden']:.3f}  (all 25: {out['conf_maxcorr_all']:.3f})")
