"""Add the Kallus (parametric softmax regret-min) method to a 1-D 2-arm experiment and regenerate the
method-comparison plots from the npz. Reuses the other 5 methods' stored results (does NOT recompute them).
Usage: python3 exp_1d_add_kallus.py {bern|uni}
"""
import sys, time, importlib.util, csv
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
MODE=sys.argv[1] if len(sys.argv)>1 else "bern"
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
if MODE=="bern": ASSETS=NEW/"assets"/"exp_1d"; NPZ="exp_1d_data.npz"; CSV="exp_1d.csv"; STITLE="Bernoulli S"
else: ASSETS=NEW/"assets"/"exp_1d_uniformS"; NPZ="exp_1d_uniformS_data.npz"; CSV="exp_1d_uniformS.csv"; STITLE="uniform-11 S"
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
kal=load("kallusmod","methods/Kallus/kallus.py"); fit_kallus=kal.fit_kallus; predict_kallus=kal.predict_kallus
A,C,B=4.0,5.0,1.0; theta0,theta1=0.5,-1.2; K=2
GRID=np.round(np.linspace(-1,1,21),6); S_GRID=np.round(np.arange(0,1.0001,0.1),6)
n_tr,n_te=800,2000; seed=0; gt=5.0; BASIS=("affine",)
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
w,_=common.ipw_weights_from_data(tr.X,tr.T,K)
d=dict(np.load(ASSETS/NPZ)); GAMMAS=d["GAMMAS"]; xs=d["xs"]; mi=4
print(f"[{MODE}][{time.time()-t0:.0f}s] running Kallus over {len(GAMMAS)} Γ (parametric softmax, odds set, uncapped)",flush=True)
ktr=[]; kte=[]; kobj=[]; ktreat=None
for gi,G in enumerate(GAMMAS):
    res=fit_kallus(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=float(G),maximize=True,wasserstein=False,basis=BASIS,
                   n_iters=20,n_restarts=4,seed=0)
    th=res.theta
    pi_tr=predict_kallus(th,tr.X,BASIS); pi_te=predict_kallus(th,te.X,BASIS)
    ktr.append(float((pi_tr*tr.Ypot).sum(1).mean())); kte.append(float((pi_te*te.Ypot).sum(1).mean()))
    kobj.append(float(res.objective_value))
    if gi==mi: ktreat=predict_kallus(th,xs.reshape(-1,1),BASIS)[:,1]
    print(f"[{MODE}][{time.time()-t0:.0f}s] Γ={G}: regret={res.objective_value:+.4f} rt_train={ktr[-1]:.4f} rt_test={kte[-1]:.4f}",flush=True)
d["Kallus_rt"]=np.array(kte); d["Kallus_obj"]=np.array(kobj); d["Kallus_rt_train"]=np.array(ktr); d["Kallus_treat_grid"]=np.array(ktreat)
np.savez(ASSETS/NPZ,**d)
# ---------- regenerate the 4 method-comparison figures (all 6 methods, from npz) ----------
METH=["RW","RO","IPW","RegretO","RegretOW","Kallus"]
LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","RegretO":"Regret-O","RegretOW":"Hajek-OW","Kallus":"Kallus"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","RegretO":"#1f77b4","RegretOW":"#17becf","Kallus":"#7f7f7f"}
MK ={"RW":"o","RO":"s","IPW":"^","RegretO":"v","RegretOW":"D","Kallus":"X"}
G=GAMMAS; mG=float(d["matched_Gamma"])
# Fig A: realised train | test
fig,(axL,axR)=plt.subplots(1,2,figsize=(12.6,5.2),dpi=140,sharey=True)
for ax,suf,ttl,fi,bm in [(axL,"_rt_train","Train (in-sample)",float(d["full_info_train"]),float(d["best_means_train"])),
                         (axR,"_rt","Test (deployed)",float(d["full_info"]),float(d["best_means"]))]:
    for m in METH: ax.plot(G,d[m+suf],'-',marker=MK[m],color=COL[m],lw=2,ms=5,label=LAB[m])
    ax.axhline(fi,ls='--',lw=1.4,color="#111",label="Full info (unconstrained)")
    ax.axhline(bm,ls='--',lw=1.4,color="#2ca02c",alpha=.8,label="Best true-means (unconstrained)")
    ax.axvline(mG,ls=':',color="#888",lw=1); ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_title(ttl); ax.grid(alpha=.25)
axL.set_ylabel("realised outcome  E[Y]")
h,l=axL.get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=8,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle(f"1-D 2-arm · {STITLE} · realised outcome vs Γ — train (in-sample) vs test (deployed)  ·  +Kallus",y=0.99,fontsize=11.5)
fig.tight_layout(rect=[0,0.06,1,1]); fig.savefig(ASSETS/"realized_train_test.png",bbox_inches="tight"); plt.close(fig)
# Fig B: objective vs Γ
fig,ax=plt.subplots(figsize=(7.8,4.8),dpi=140)
for m in METH: ax.plot(G,d[m+"_obj"],'-',marker=MK[m],color=COL[m],lw=2,ms=5,label=LAB[m])
ax.axhline(0,color="#999",lw=.8); ax.set_xlabel("Γ"); ax.set_ylabel("worst-case objective")
ax.set_title(f"1-D 2-arm · {STITLE} · worst-case objective vs Γ"); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(ASSETS/"objective_vs_gamma.png"); plt.close(fig)
# Fig C: policy strips (chosen arm = treat if treat_grid>0.5)
acmap=ListedColormap(["#dfe3ea","#d62728"]); mat=[(d[m+"_treat_grid"]>0.5).astype(int) for m in METH]
fig,ax=plt.subplots(figsize=(8.4,3.9),dpi=140)
im=ax.imshow(np.array(mat),aspect="auto",cmap=acmap,vmin=0,vmax=1,extent=[xs.min(),xs.max(),len(mat)-.5,-.5])
ax.set_yticks(range(len(METH))); ax.set_yticklabels([LAB[m] for m in METH],fontsize=9)
ax.set_xlabel("X"); ax.set_title(f"{STITLE} · chosen arm vs X at matched Γ={mG}  (grey=control, red=treat)")
cb=fig.colorbar(im,ax=ax,ticks=[0,1],pad=.02); cb.ax.set_yticklabels(["control","treat"])
fig.tight_layout(); fig.savefig(ASSETS/"policy_strips.png"); plt.close(fig)
# Fig D: all-methods treat policy
fig,ax=plt.subplots(figsize=(8.2,4.8),dpi=140)
for m in METH: ax.plot(xs,d[m+"_treat_grid"],'-',marker=MK[m],color=COL[m],lw=2,ms=5,label=LAB[m],alpha=.95)
ax.set_xlabel("X"); ax.set_ylabel("π(treat | x)"); ax.set_ylim(-.05,1.05)
ax.set_title(f"{STITLE} · treatment policy π(treat | x) — all methods at matched Γ={mG}")
ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2,loc="upper left")
fig.tight_layout(); fig.savefig(ASSETS/"policy_treat_all.png"); plt.close(fig)
# ---------- rewrite CSV with all 6 methods ----------
with open(ASSETS/CSV,"w",newline="") as f:
    wri=csv.writer(f); wri.writerow(["method","Gamma","realized_train","realized_test","objective"])
    for m in METH:
        for gi,gg in enumerate(G): wri.writerow([LAB[m],round(float(gg),4),float(d[m+"_rt_train"][gi]),float(d[m+"_rt"][gi]),float(d[m+"_obj"][gi])])
print(f"[{MODE}][{time.time()-t0:.0f}s] DONE -> npz+csv updated, 4 figures regenerated with Kallus",flush=True)
print(f"[{MODE}] Kallus @matched Γ: rt_train={ktr[mi]:.4f} rt_test={kte[mi]:.4f} regret={kobj[mi]:+.4f} treat_rate={np.mean(ktreat>0.5):.3f}")
