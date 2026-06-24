"""1-D discrete 2-arm DGP where R-OW wins (cap-enabled Wasserstein edge): Γ-sweep + plots."""
import sys, time, importlib.util, csv
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0,str(NEW)); import common
ASSETS=NEW/"assets"/"exp_1d"; ASSETS.mkdir(parents=True,exist_ok=True)
def load(n,rel):
    sp=importlib.util.spec_from_file_location(n,str(NEW/rel)); m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
A,C,B=4.0,5.0,1.0; theta0,theta1=0.5,-1.2; K=2; CAP=(1.0,0.5)
GRID=np.round(np.linspace(-1,1,21),6); n_tr,n_te=800,2000; seed=0; gt=5.0
GAMMAS=[1.0,3.0,6.0,9.0,round(float(np.exp(gt/2)),4),16.0]; mi=4
def sig(z): z=np.asarray(z,float); return 1/(1+np.exp(-np.clip(z,-40,40)))
def pa(X,S,k): return np.clip(sig(theta0+0*X) if k==0 else sig(theta1+A*X+C*np.asarray(S,float)),1e-3,1-1e-3)
def generate(n,gamma,rng):
    X=rng.choice(GRID,size=n); S=(rng.uniform(size=n)<0.5).astype(int)
    Yp=np.column_stack([(rng.uniform(size=n)<pa(X,S,0)).astype(float),(rng.uniform(size=n)<pa(X,S,1)).astype(float)])
    U1=-B*X+gamma*(S-0.5); p1=np.exp(U1)/(1+np.exp(U1)); T=(rng.uniform(size=n)<p1).astype(int)
    mu=np.column_stack([0.5*pa(X,0,0)+0.5*pa(X,1,0),0.5*pa(X,0,1)+0.5*pa(X,1,1)])
    from types import SimpleNamespace
    return SimpleNamespace(X=X.reshape(-1,1),S=S,T=T,Y=Yp[np.arange(n),T],Ypot=Yp,mu=mu,n_arms=K)
t0=time.time(); rng=np.random.default_rng(seed)
tr=generate(n_tr,gt,rng); te=generate(n_te,gt,rng)
w,_=common.ipw_weights_from_data(tr.X,tr.T,K); D=common.pairwise_distance_matrix(tr.X)
eps=tuple(common.tight_epsilon(D,tr.T,w,K,is_distance=True,c_eps=1.0))
print(f"[{time.time()-t0:.0f}s] setup; arms {np.bincount(tr.T).tolist()} eps={tuple(round(e,4) for e in eps)}",flush=True)
row=load("row","methods/IPW-O-W/Capped/ipw_o_w_capped.py").solve_ipw_o_w_capped
ro =load("ro","methods/IPW-O-X/Capped/ipw_o_x_capped.py").solve_ipw_o_x_capped
ipw=load("ipw","methods/IPW-X-X/Capped/ipw_x_x_capped.py").solve_ipw_x_x_capped
rgo=load("rgo","methods/Hajek-O-X/Capped/hajek_o_x_capped.py").solve_hajek_o_x_capped
rgw=load("rgw","methods/Hajek-O-W/Capped/hajek_o_w_capped.py").solve_hajek_o_w_capped
uniq={};
for j,xv in enumerate(np.round(tr.X.ravel(),6)): uniq.setdefault(float(xv),j)
ukeys=np.array(list(uniq.keys())); uidx=np.array([uniq[k] for k in ukeys])
nn_te=np.array([np.argmin(np.abs(ukeys-x)) for x in np.round(te.X.ravel(),6)])
xs=np.sort(ukeys); nn_xs=np.array([np.argmin(np.abs(ukeys-x)) for x in xs])
def rt(pi): pi=np.asarray(pi,float); cols=pi[:,uidx][:,nn_te]; return float((cols*te.Ypot.T).sum()/cols.shape[1])
def polx(pi): pi=np.asarray(pi,float); return pi[:,uidx][:,nn_xs]
METH=["RW","RO","IPW","RegretO","RegretOW"]; LAB={"RW":"R-OW","RO":"R-O","IPW":"IPW","RegretO":"Regret-O","RegretOW":"Hajek-OW"}
COL={"RW":"#d62728","RO":"#9467bd","IPW":"#2ca02c","RegretO":"#1f77b4","RegretOW":"#17becf"}
res={m:dict(obj=[],rt=[],pi=[]) for m in METH}
di=ipw(tr.X,tr.T,tr.Y,w,n_arms=K,cap=CAP,discretize=False)
for G in GAMMAS:
    sv={"RW":row(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,discretize=False,zscore=False,epsilon=eps),
        "RO":ro(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,discretize=False),"IPW":di,
        "RegretO":rgo(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,maximize=True,discretize=False),
        "RegretOW":rgw(tr.X,tr.T,tr.Y,w,n_arms=K,Gamma=G,cap=CAP,maximize=True,discretize=False,zscore=False,epsilon=eps)}
    for m in METH: res[m]["obj"].append(sv[m].objective_value); res[m]["rt"].append(rt(sv[m].pi)); res[m]["pi"].append(np.asarray(sv[m].pi))
    print(f"[{time.time()-t0:.0f}s] Γ={G} done",flush=True)
full_info=float(np.mean(te.Ypot.max(axis=1))); best_means=float(np.mean(te.Ypot[np.arange(n_te),np.argmax(te.mu,axis=1)]))
# Fig1 realised vs Γ
fig,ax=plt.subplots(figsize=(7.8,4.8),dpi=140)
for m in METH: ax.plot(GAMMAS,res[m]["rt"],'-o',color=COL[m],lw=2,ms=5,label=LAB[m])
ax.axhline(full_info,ls='--',lw=1.4,color="#111",label="Full info (unconstrained)")
ax.axhline(best_means,ls='--',lw=1.4,color="#2ca02c",alpha=.8,label="Best true-means (unconstrained)")
ax.axvline(GAMMAS[mi],ls=':',color="#888",lw=1); ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_ylabel("realised test outcome  E[Y]")
ax.set_title("1-D discrete 2-arm (R-OW wins) · n=800 — realised test outcome vs Γ"); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(ASSETS/"realized_vs_gamma.png"); plt.close(fig)
# Fig2 objective vs Γ
fig,ax=plt.subplots(figsize=(7.8,4.8),dpi=140)
for m in METH: ax.plot(GAMMAS,res[m]["obj"],'-o',color=COL[m],lw=2,ms=5,label=LAB[m])
ax.axhline(0,color="#999",lw=.8); ax.set_xlabel("Γ"); ax.set_ylabel("worst-case objective")
ax.set_title("1-D discrete 2-arm · worst-case objective vs Γ"); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(ASSETS/"objective_vs_gamma.png"); plt.close(fig)
# Fig3 policy strips
acmap=ListedColormap(["#dfe3ea","#d62728"]); mat=[np.argmax(polx(res[m]["pi"][mi]),axis=0) for m in METH]
fig,ax=plt.subplots(figsize=(8.4,3.4),dpi=140)
im=ax.imshow(np.array(mat),aspect="auto",cmap=acmap,vmin=0,vmax=1,extent=[xs.min(),xs.max(),len(mat)-.5,-.5])
ax.set_yticks(range(len(METH))); ax.set_yticklabels([LAB[m] for m in METH],fontsize=9)
ax.set_xlabel("X"); ax.set_title(f"1-D · chosen arm vs X at matched Γ={GAMMAS[mi]}  (grey=control, red=treat)")
cb=fig.colorbar(im,ax=ax,ticks=[0,1],pad=.02); cb.ax.set_yticklabels(["control","treat"])
fig.tight_layout(); fig.savefig(ASSETS/"policy_strips.png"); plt.close(fig)
# Fig4 R-OW π_k(x)
pol=polx(res["RW"]["pi"][mi]); ac=["#bdbdbd","#d62728"]
fig,ax=plt.subplots(figsize=(7.8,4.4),dpi=140)
for k in range(K): ax.plot(xs,pol[k],'-o',color=ac[k],lw=2,ms=4,label=("control" if k==0 else "treat"))
ax.set_xlabel("X"); ax.set_ylabel("π_k(x)"); ax.set_ylim(-.05,1.05)
ax.set_title(f"1-D R-OW policy π_k(x) at matched Γ={GAMMAS[mi]}"); ax.grid(alpha=.25); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ASSETS/"policy_detail_rw.png"); plt.close(fig)
np.savez(ASSETS/"exp_1d_data.npz",GAMMAS=np.array(GAMMAS),xs=xs,full_info=full_info,best_means=best_means,
         **{f"{m}_rt":np.array(res[m]["rt"]) for m in METH},**{f"{m}_obj":np.array(res[m]["obj"]) for m in METH})
with open(ASSETS/"exp_1d.csv","w",newline="") as f:
    wri=csv.writer(f); wri.writerow(["method","Gamma","realized_test","objective"])
    for m in METH:
        for gi,G in enumerate(GAMMAS): wri.writerow([LAB[m],round(G,4),res[m]["rt"][gi],res[m]["obj"][gi]])
print(f"[{time.time()-t0:.0f}s] DONE -> {ASSETS}",flush=True)
print("ceilings: Full=%.4f Best-means=%.4f"%(full_info,best_means))
print("realized @ matched Γ:",{LAB[m]:round(res[m]['rt'][mi],4) for m in METH})
