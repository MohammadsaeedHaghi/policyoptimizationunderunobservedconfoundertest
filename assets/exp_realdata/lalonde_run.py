"""LaLonde (NSW job-training) real-data experiment — the canonical STRONG-confounding benchmark.

SETUP (honest, no fabrication):
  • TRAINING = OBSERVATIONAL LaLonde: NSW treated (185 real trainees) + PSID controls (general-population, subsampled).
    This pairing is strongly confounded: trainees are disadvantaged (low prior earnings), PSID controls are not.
  • We HIDE the prior-earnings columns re74, re75 (the strongest confounders — they drive both selection and re78).
    Methods see only ONE observed covariate: years of education (binned), which has a clean, interpretable benefit
    gradient (experimental ATE rises with education). Everything else (incl. re74/re75) is the UNOBSERVED confounder.
  • EVALUATION ground truth = the EXPERIMENTAL NSW (randomized 185 vs 260): deploy each learned policy π(treat|edu)
    and estimate its TRUE value by inverse-propensity weighting on the randomised data (e=185/445 known exactly → unbiased).
  • EXPERIMENTAL ATE (+$1794) is the truth; the naive OBSERVATIONAL ATE is hugely biased (the famous LaLonde result).

WHAT THIS SHOWS (honest): under this strong, benign-direction confounding the observational signal says "treatment looks
harmful", so ALL methods that trust the data hedge toward not-treating; the robust methods quantify, via Γ, how much
hidden confounding it would take to change the conclusion (sensitivity analysis — exactly what the MSM box is for).
We report the realised value of every method's policy + the worst-case objective vs Γ + the confounding diagnostic.
Saves lalonde_results.json (raw numbers) for the HTML + plots. Usage: python3 lalonde_run.py"""
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
DATA=NEW/"assets"/"exp_realdata"/"data"; K=2; NCTRL=420; CAP=(1.0,1.0)   # no binding capacity (1-D discrete education)
GAMMAS=[1.0,1.5,2.0,3.0,5.0,8.0]; SEEDS=list(range(5)); SC=1000.0          # re78 in $1000s
EDGES=np.array([-0.5,8.5,9.5,10.5,11.5,12.5,20.0]); NB=len(EDGES)-1         # education bins: <=8,9,10,11,12,>=13
CTR=np.linspace(-1,1,NB)
exp=pd.read_stata(DATA/"nsw_dw.dta"); psid=pd.read_stata(DATA/"psid_controls.dta")
COVS=['age','education','black','hispanic','married','nodegree','re74','re75']
def ebin(ed): return np.clip(np.digitize(np.asarray(ed,float),EDGES)-1,0,NB-1)
# ---- confounding diagnostic (treated NSW vs PSID controls on the HIDDEN earnings) ----
nsw_t=exp[exp.treat==1]
diag={c:[float(nsw_t[c].mean()),float(psid[c].mean())] for c in COVS}
exp_ate=float((exp[exp.treat==1].re78.mean()-exp[exp.treat==0].re78.mean())/SC)
# experimental ATE by education bin (the deployable benefit gradient + the realised ceiling reference)
eb=ebin(exp.education.values); ate_by_bin=[]
for b in range(NB):
    m=eb==b; t=exp.re78.values[m & (exp.treat.values==1)]; c=exp.re78.values[m & (exp.treat.values==0)]
    ate_by_bin.append(float((t.mean()-c.mean())/SC) if len(t) and len(c) else float('nan'))
# experimental IPW evaluator of a per-bin treat-probability policy (randomised e known -> unbiased)
eX=exp.education.values; eT=exp.treat.values.astype(int); eY=exp.re78.values/SC; e1=float((eT==1).mean()); ebn=ebin(eX)
def realized(treatprob_by_bin):
    pt=np.asarray(treatprob_by_bin,float)[ebn]                       # P(treat|edu) for each experimental unit
    pol=np.where(eT==1,pt,1-pt); prop=np.where(eT==1,e1,1-e1)
    return float(np.mean(pol*eY/prop))                               # unbiased policy value, $1000s
ctrl_val=realized(np.zeros(NB)); treatall_val=realized(np.ones(NB))  # never-treat / always-treat references
# best deployable on the experimental gradient: treat bins with ATE>0
best_val=realized((np.nan_to_num(np.array(ate_by_bin),nan=-9)>0).astype(float))
print(f"exp ATE={exp_ate:.3f} ($1000s); ctrl={ctrl_val:.3f} treat-all={treatall_val:.3f} best-by-bin={best_val:.3f}",flush=True)
print("ATE by edu bin:",[round(a,3) for a in ate_by_bin],flush=True)
def grid_policy(rescol):                                             # collapse a per-unit policy to per-bin treat-prob on TRAIN support
    return rescol
out={"diag":diag,"exp_ate":exp_ate,"ctrl":ctrl_val,"treatall":treatall_val,"best":best_val,
     "ate_by_bin":ate_by_bin,"gammas":GAMMAS,"bins":NB,"edges":EDGES.tolist(),
     "methods":{},"naive_obs_ate":None}
METHODS=["R-OW","R-O","IPW","AIPW","R-OW-DR","Kallus"]
acc={m:{g:[] for g in GAMMAS} for m in METHODS}     # realized value per method,Γ,seed
pol_acc={m:{g:[] for g in GAMMAS} for m in METHODS}  # per-bin treat-prob
obj_acc={m:{g:[] for g in GAMMAS} for m in METHODS}
naive_ates=[]
for seed in SEEDS:
    rng=np.random.default_rng(seed)
    ctrl_idx=rng.choice(len(psid),size=NCTRL,replace=False)
    tr=pd.concat([exp[exp.treat==1][COVS+['re78']], psid.iloc[ctrl_idx][COVS+['re78']]],ignore_index=True)
    Tt=np.r_[np.ones(len(exp[exp.treat==1]),int),np.zeros(NCTRL,int)]
    Yt=tr.re78.values/SC; b=ebin(tr.education.values); Xg=CTR[b].reshape(-1,1)
    # naive observational ATE (what the data 'looks like') — strongly biased
    naive_ates.append(float((Yt[Tt==1].mean()-Yt[Tt==0].mean())))
    w,_=common.ipw_weights_from_data(Xg,Tt,K); Dm=common.pairwise_distance_matrix(Xg)
    muhat=common.outcome_means(Xg,Tt,Yt,n_arms=K,cross_fit=True)
    # map train support -> bins, to read off a per-bin treat-probability from any policy matrix
    ub=np.array(sorted(set(np.round(Xg.ravel(),6))));
    binof=np.array([int(np.argmin(np.abs(CTR-x))) for x in ub])
    colidx={}                                  # first train column at each unique support point
    for j,xv in enumerate(np.round(Xg.ravel(),6)): colidx.setdefault(float(xv),j)
    cidx=np.array([colidx[float(x)] for x in ub])
    def perbin(pi):
        pi=np.asarray(pi,float); tp=np.full(NB,np.nan)
        for u,bb in zip(cidx,binof): tp[bb]=pi[1,u]
        # bins absent from train support -> fall back to nearest present bin
        present=np.where(~np.isnan(tp))[0]
        for bb in range(NB):
            if np.isnan(tp[bb]): tp[bb]=tp[present[np.argmin(np.abs(present-bb))]]
        return tp
    for g in GAMMAS:
        eps=tuple(common.tight_epsilon(Dm,Tt,w,K,is_distance=True,c_eps=1.0))
        rr={}
        rr["R-OW"]=row(Xg,Tt,Yt,w,n_arms=K,Gamma=g,cap=CAP,discretize=False,zscore=False,epsilon=eps)
        rr["R-O"]=ro(Xg,Tt,Yt,w,n_arms=K,Gamma=g,cap=CAP,discretize=False)
        rr["IPW"]=ipw(Xg,Tt,Yt,w,n_arms=K,cap=CAP,discretize=False)            # Γ-independent
        rr["AIPW"]=rodr(Xg,Tt,Yt,w,muhat,n_arms=K,Gamma=1.0,cap=CAP,discretize=False)  # non-robust DR
        rr["R-OW-DR"]=rowdr(Xg,Tt,Yt,w,muhat,n_arms=K,Gamma=g,cap=CAP,discretize=False,zscore=False,epsilon=eps)
        for m in ["R-OW","R-O","IPW","AIPW","R-OW-DR"]:
            tp=perbin(rr[m].pi); pol_acc[m][g].append(tp.tolist()); acc[m][g].append(realized(tp))
            obj_acc[m][g].append(float(getattr(rr[m],'objective_value',np.nan)))
        th=kal.fit_kallus(Xg,Tt,Yt,w,n_arms=K,Gamma=g,maximize=True,wasserstein=False,basis=("affine",),n_iters=20,n_restarts=4,seed=0)
        kp=kal.predict_kallus(th.theta,CTR.reshape(-1,1),("affine",))[:,1]      # treat-prob per bin-center
        pol_acc["Kallus"][g].append(kp.tolist()); acc["Kallus"][g].append(realized(kp)); obj_acc["Kallus"][g].append(float('nan'))
    print(f"seed{seed} naiveATE={naive_ates[-1]:.3f} | "+" ".join(f"{m}@Γ3={np.mean(acc[m][3.0]):.3f}" for m in METHODS),flush=True)
out["naive_obs_ate"]=float(np.mean(naive_ates))
for m in METHODS:
    out["methods"][m]={"realized_mean":[float(np.mean(acc[m][g])) for g in GAMMAS],
                       "realized_sd":[float(np.std(acc[m][g])) for g in GAMMAS],
                       "policy_mean":[np.mean(np.array(pol_acc[m][g]),axis=0).tolist() for g in GAMMAS],
                       "objective_mean":[float(np.nanmean(obj_acc[m][g])) for g in GAMMAS]}
(NEW/"assets"/"exp_realdata"/"lalonde_results.json").write_text(json.dumps(out,indent=1))
print("\n=== LaLonde done. naive obs ATE=%.3f vs experimental ATE=%.3f ($1000s) ==="%(out["naive_obs_ate"],exp_ate))
for m in METHODS: print(f"  {m:9s} realized@Γ:"+" ".join(f"{v:.3f}"for v in out['methods'][m]['realized_mean']))
print(f"  refs: never-treat={ctrl_val:.3f} treat-all={treatall_val:.3f} best-by-bin={best_val:.3f}")
