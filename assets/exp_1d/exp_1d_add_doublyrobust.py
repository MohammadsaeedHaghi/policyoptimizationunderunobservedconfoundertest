"""Add the DOUBLY-ROBUST family (R-OW-DoublyRobust, R-O-DoublyRobust) to the 1-D 2-arm experiment and
regenerate the method-comparison plots from the npz. REUSES the other 6 methods' stored results (does NOT
recompute R-OW/R-O/IPW/Regret-O/Hajek-OW/Kallus). The DR methods take an ESTIMATED cross-fitted outcome
model μ̂_k(x) (per-arm logistic on the observed binary (X,T,Y) — never the oracle). Same capped LP setup,
same seed-0 data, same deploy/scoring as exp_1d.py (verified bit-identical reconstruction of stored rt).
Usage: python3 exp_1d_add_doublyrobust.py {bern|uni}"""
import sys, time, importlib.util, csv, json
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
MODE=sys.argv[1] if len(sys.argv)>1 else "bern"
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
if MODE=="bern": ASSETS=NEW/"assets"/"exp_1d"; NPZ="exp_1d_data.npz"; CSV="exp_1d.csv"; STITLE="Bernoulli S"
else: ASSETS=NEW/"assets"/"exp_1d_uniformS"; NPZ="exp_1d_uniformS_data.npz"; CSV="exp_1d_uniformS.csv"; STITLE="uniform-11 S"
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
rowdr=load("rowdr","methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py").solve_doublyrobust_o_w_capped
rodr =load("rodr","methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py").solve_doublyrobust_o_x_capped
# ---- DGP (verbatim from exp_1d.py / exp_1d_add_kallus.py) ----
A,C,B=4.0,5.0,1.0; theta0,theta1=0.5,-1.2; K=2; CAP=(1.0,0.5)
GRID=np.round(np.linspace(-1,1,21),6); S_GRID=np.round(np.arange(0,1.0001,0.1),6)
n_tr,n_te=800,2000; seed=0; gt=5.0
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def pa(X,S,k): return np.clip(sig(theta0+0*np.asarray(X,float)) if k==0 else sig(theta1+A*np.asarray(X,float)+C*np.asarray(S,float)),1e-3,1-1e-3)
def generate(nn,gamma,rng):
    X=rng.choice(GRID,size=nn)
    S=(rng.uniform(size=nn)<0.5).astype(int) if MODE=="bern" else rng.choice(S_GRID,size=nn)
    Yp=np.column_stack([(rng.uniform(size=nn)<pa(X,S,0)).astype(float),(rng.uniform(size=nn)<pa(X,S,1)).astype(float)])
    U1=-B*X+gamma*(S-0.5); p1=np.exp(U1)/(1+np.exp(U1)); T=(rng.uniform(size=nn)<p1).astype(int)
    from types import SimpleNamespace
    return SimpleNamespace(X=X.reshape(-1,1),T=T,Y=Yp[np.arange(nn),T],Ypot=Yp)
t0=time.time(); rng=np.random.default_rng(seed); tr=generate(n_tr,gt,rng); te=generate(n_te,gt,rng)
w,_=common.ipw_weights_from_data(tr.X,tr.T,K); D=common.pairwise_distance_matrix(tr.X)
eps=tuple(common.tight_epsilon(D,tr.T,w,K,is_distance=True,c_eps=1.0))
muhat=common.outcome_means(tr.X,tr.T,tr.Y,n_arms=K,cross_fit=True)        # ESTIMATED μ̂ (binary→logistic), NOT oracle
# ---- support mapping + deploy/scoring (EXACT exp_1d.py protocol) ----
uniq={}
for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
ukeys=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in ukeys])
xs=np.sort(ukeys); nn_xs=np.array([int(np.argmin(np.abs(ukeys-x))) for x in xs])
nn_te=np.array([int(np.argmin(np.abs(ukeys-x))) for x in np.round(te.X.ravel(),6)])
nn_tr=np.array([int(np.argmin(np.abs(ukeys-x))) for x in np.round(tr.X.ravel(),6)])
def polx(pi): pi=np.asarray(pi,float); return pi[:,uidx][:,nn_xs]                     # (K,21) grid policy
def rt_test(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_te]; return float((c*te.Ypot.T).sum()/c.shape[1])
def rt_train(pi): pi=np.asarray(pi,float); c=pi[:,uidx][:,nn_tr]; return float((c*tr.Ypot.T).sum()/c.shape[1])
# ---- load stored NPZ; VERIFY data/scoring reproduction against a stored method (gate) ----
d=dict(np.load(ASSETS/NPZ)); GAMMAS=d["GAMMAS"]; mi=4; mG=float(d["matched_Gamma"])
assert np.allclose(xs,d["xs"]), "xs grid mismatch — data reproduction broken"
gi_te=np.array([int(np.argmin(np.abs(xs-x))) for x in te.X.ravel()])
tg=d["RW_treat_grid"]; rt_chk=float(((1-tg)[gi_te]*te.Ypot[:,0]+tg[gi_te]*te.Ypot[:,1]).mean())
assert abs(rt_chk-d["RW_rt"][mi])<1e-9, f"scoring gate FAILED: {rt_chk} vs {d['RW_rt'][mi]}"
print(f"[{MODE}][{time.time()-t0:.0f}s] gate OK (RW rt recon Δ={abs(rt_chk-d['RW_rt'][mi]):.2e}); eps={tuple(round(e,4) for e in eps)} "
      f"μ̂ agree-with-best={float(np.mean((muhat[:,1]>muhat[:,0])==(tr.Ypot[:,1].mean()>0))):.2f}",flush=True)
# ---- run ONLY the two DR methods over the Γ sweep (capped, same cap=(1.0,0.5)) ----
def safe(fn):
    try: return fn()
    except Exception as e: print("  FAIL:",str(e)[:100]); return None
for key,solve in [("RW_DR",lambda G:rowdr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=float(G),cap=CAP,discretize=False,zscore=False,epsilon=eps)),
                  ("RO_DR",lambda G:rodr(tr.X,tr.T,tr.Y,w,muhat,n_arms=K,Gamma=float(G),cap=CAP,discretize=False))]:
    rt_=[]; rttr_=[]; obj_=[]; tgrid=None
    for gi,G in enumerate(GAMMAS):
        sv=safe(lambda:solve(G))
        if sv is None: rt_.append(np.nan); rttr_.append(np.nan); obj_.append(np.nan); continue
        pi=sv.pi; rt_.append(rt_test(pi)); rttr_.append(rt_train(pi)); obj_.append(float(sv.objective_value))
        if gi==mi: tgrid=polx(pi)[1]
        print(f"[{MODE}][{time.time()-t0:.0f}s] {key} Γ={G}: obj={obj_[-1]:+.4f} rt_train={rttr_[-1]:.4f} rt_test={rt_[-1]:.4f}",flush=True)
    d[key+"_rt"]=np.array(rt_); d[key+"_rt_train"]=np.array(rttr_); d[key+"_obj"]=np.array(obj_); d[key+"_treat_grid"]=np.array(tgrid)
np.savez(ASSETS/NPZ,**d)
# ---- regenerate the 4 method-comparison figures (all 8 methods, from npz) ----
METH=["RW","RO","IPW","RegretO","RegretOW","RW_DR","RO_DR","Kallus"]
LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","RegretO":"Regret-O","RegretOW":"Hajek-OW","RW_DR":"R-OW-DR","RO_DR":"R-O-DR","Kallus":"Kallus"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","RegretO":"#1f77b4","RegretOW":"#17becf","RW_DR":"#a01f1f","RO_DR":"#6d4a7d","Kallus":"#7f7f7f"}
MK ={"RW":"o","RO":"s","IPW":"^","RegretO":"v","RegretOW":"D","RW_DR":"P","RO_DR":"*","Kallus":"X"}
G=GAMMAS
# Fig A: realised train | test
fig,(axL,axR)=plt.subplots(1,2,figsize=(12.8,5.3),dpi=140,sharey=True)
for ax,suf,ttl,fi,bm in [(axL,"_rt_train","Train (in-sample)",float(d["full_info_train"]),float(d["best_means_train"])),
                         (axR,"_rt","Test (deployed)",float(d["full_info"]),float(d["best_means"]))]:
    for m in METH: ax.plot(G,d[m+suf],'-',marker=MK[m],color=COL[m],lw=2,ms=5.5,label=LAB[m])
    ax.axhline(fi,ls='--',lw=1.4,color="#111",label="Full info (unconstrained)")
    ax.axhline(bm,ls='--',lw=1.4,color="#2ca02c",alpha=.8,label="Best true-means (unconstrained)")
    ax.axvline(mG,ls=':',color="#888",lw=1); ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_title(ttl); ax.grid(alpha=.25)
axL.set_ylabel("realised outcome  E[Y]")
h,l=axL.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=10,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.03))
fig.suptitle(f"1-D 2-arm · {STITLE} · realised outcome vs Γ — train (in-sample) vs test (deployed)  ·  +DoublyRobust",y=0.99,fontsize=11.5)
fig.tight_layout(rect=[0,0.07,1,1]); fig.savefig(ASSETS/"realized_train_test.png",bbox_inches="tight"); plt.close(fig)
# Fig B: objective vs Γ
fig,ax=plt.subplots(figsize=(8.0,4.9),dpi=140)
for m in METH: ax.plot(G,d[m+"_obj"],'-',marker=MK[m],color=COL[m],lw=2,ms=5.5,label=LAB[m])
ax.axhline(0,color="#999",lw=.8); ax.set_xlabel("Γ"); ax.set_ylabel("worst-case objective")
ax.set_title(f"1-D 2-arm · {STITLE} · worst-case objective vs Γ"); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(ASSETS/"objective_vs_gamma.png"); plt.close(fig)
# Fig C: policy strips
acmap=ListedColormap(["#dfe3ea","#d62728"]); mat=[(d[m+"_treat_grid"]>0.5).astype(int) for m in METH]
fig,ax=plt.subplots(figsize=(8.6,4.4),dpi=140)
im=ax.imshow(np.array(mat),aspect="auto",cmap=acmap,vmin=0,vmax=1,extent=[xs.min(),xs.max(),len(mat)-.5,-.5])
ax.set_yticks(range(len(METH))); ax.set_yticklabels([LAB[m] for m in METH],fontsize=9)
ax.set_xlabel("X"); ax.set_title(f"{STITLE} · chosen arm vs X at matched Γ={mG}  (grey=control, red=treat)")
cb=fig.colorbar(im,ax=ax,ticks=[0,1],pad=.02); cb.ax.set_yticklabels(["control","treat"])
fig.tight_layout(); fig.savefig(ASSETS/"policy_strips.png"); plt.close(fig)
# Fig D: all-methods treat policy
fig,ax=plt.subplots(figsize=(8.4,4.9),dpi=140)
for m in METH: ax.plot(xs,d[m+"_treat_grid"],'-',marker=MK[m],color=COL[m],lw=2,ms=5,label=LAB[m],alpha=.95)
ax.set_xlabel("X"); ax.set_ylabel("π(treat | x)"); ax.set_ylim(-.05,1.05)
ax.set_title(f"{STITLE} · treatment policy π(treat | x) — all methods at matched Γ={mG}")
ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2,loc="upper left")
fig.tight_layout(); fig.savefig(ASSETS/"policy_treat_all.png"); plt.close(fig)
# ---- rewrite CSV (8 methods) + emit JSON for the HTML interactive chart ----
with open(ASSETS/CSV,"w",newline="") as f:
    wri=csv.writer(f); wri.writerow(["method","Gamma","realized_train","realized_test","objective"])
    for m in METH:
        for gi,gg in enumerate(G): wri.writerow([LAB[m],round(float(gg),4),float(d[m+"_rt_train"][gi]),float(d[m+"_rt"][gi]),float(d[m+"_obj"][gi])])
chart={"mode":MODE,"x":[round(float(v),4) for v in xs],"gamma":mG,
       "dr_series":[{"id":k,"label":LAB[k],"color":COL[k],"y":[round(float(v),4) for v in d[k+"_treat_grid"]]} for k in ("RW_DR","RO_DR")]}
(ASSETS/"dr_chart.json").write_text(json.dumps(chart))
summ={"mode":MODE,"matched_Gamma":mG,"full_info":float(d["full_info"]),"best_means":float(d["best_means"]),
      "methods":{LAB[m]:{"rt_train":round(float(d[m+"_rt_train"][mi]),4),"rt_test":round(float(d[m+"_rt"][mi]),4),
                          "obj":round(float(d[m+"_obj"][mi]),4),"treat_rate":round(float(np.mean(d[m+"_treat_grid"]>0.5)),3),
                          "rt_test_sweep":[round(float(v),4) for v in d[m+"_rt"]]} for m in METH}}
(ASSETS/"dr_summary.json").write_text(json.dumps(summ,indent=1))
print(f"[{MODE}][{time.time()-t0:.0f}s] DONE -> npz+csv+json updated, 4 figures regenerated with DR family")
print(f"[{MODE}] @matched Γ rt_test:",{LAB[m]:round(float(d[m+'_rt'][mi]),4) for m in METH})
