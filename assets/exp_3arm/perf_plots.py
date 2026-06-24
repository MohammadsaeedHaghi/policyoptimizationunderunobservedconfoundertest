"""Aggregate the 5 seeds and draw the performance figures (all static, no dropdowns):
  realized_vs_gamma.png  — realised value vs Γ (train + test), 8 methods + 2 ceilings, ±SD
  objective_vs_gamma.png — worst-case in-sample objective vs Γ
  policy.png             — policy-strip grid at matched Γ (chosen arm vs X, per method)
  realized_bar.png       — realised value bar chart at Γ=3 (peak) and matched Γ
  capacity_usage.png     — per-arm allocation vs Γ vs the cap (arms 1,2)
Saves exp3arm_results.json."""
import sys, json, glob, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location("dgp",str(HERE/"dgp.py")); dgp=importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
seeds=[json.loads(open(f).read()) for f in sorted(glob.glob(str(HERE/"_multiseed"/"seed*.json")))]
nS=len(seeds); G=seeds[0]["gammas"]; mi=dgp.mi; gpeak=G.index(3.0); mG=G[mi]
METHODS=["R-OW","R-OW-DR","R-O","R-O-DR","Hajek-OW","Regret-O","AIPW","IPW"]
COL={"R-OW":"#d62728","R-OW-DR":"#a01f1f","R-O":"#9467bd","R-O-DR":"#6d4a7d","Hajek-OW":"#17becf","Regret-O":"#1f77b4","IPW":"#2ca02c","AIPW":"#ff7f0e"}
MK={"R-OW":"o","R-OW-DR":"o","R-O":"s","R-O-DR":"s","Hajek-OW":"^","Regret-O":"^","IPW":"v","AIPW":"D"}
def agg(m,fld):
    A=np.array([[s["methods"][m][str(gi)][fld] for gi in range(len(G))] for s in seeds]); return A.mean(0),A.std(0)
cei={k:float(np.mean([s["ceilings"][k] for s in seeds])) for k in seeds[0]["ceilings"]}
cap=np.mean([s["cap"] for s in seeds],0)
out={"gammas":G,"matched_idx":mi,"peak_idx":gpeak,"cap":[float(c) for c in cap],"ceilings":cei,
     "methods":{m:{f:[list(a) for a in agg(m,f)] for f in ["rt_train","rt_test","exp_train","exp_test","obj"]} for m in METHODS}}
(HERE/"exp3arm_results.json").write_text(json.dumps(out,indent=1))

# ---- (1) realised value vs Γ : train + test ----
fig,ax=plt.subplots(1,2,figsize=(13.4,5.0),sharey=True)
for j,(fld,ttl) in enumerate([("rt_train","Train (in-sample)"),("rt_test","Test (deployed)")]):
    a=ax[j]
    for m in METHODS:
        y,e=agg(m,fld); ls="--" if m in("IPW","AIPW") else "-"
        a.plot(G,y,ls+MK[m],color=COL[m],lw=1.9,ms=4,label=m+(" (Γ-indep.)" if m in("IPW","AIPW") else ""))
        if m not in("IPW","AIPW"): a.fill_between(G,y-e,y+e,color=COL[m],alpha=.08)
    ck="full_info_"+("train" if "train" in fld else "test"); bk="best_means_"+("train" if "train" in fld else "test")
    a.axhline(cei[ck],ls=":",c="#111",lw=1.4,label="Full-info ceiling %.3f"%cei[ck])
    a.axhline(cei[bk],ls="--",c="#111",lw=1.2,label="Best-means ceiling %.3f"%cei[bk])
    a.axvline(mG,color="#bbb",ls=":",lw=1.2); a.axvline(3.0,color="#d62728",ls="--",lw=1.1,alpha=.6)
    a.set_xlabel("Γ (assumed sensitivity)"); a.set_title(ttl,fontsize=11); a.grid(alpha=.25)
ax[0].set_ylabel("realised success rate (mean ± SD, %d seeds)"%nS); ax[1].legend(fontsize=7.6,ncol=2,loc="lower right")
fig.suptitle("3-arm exp_a (γ=5), capacity-constrained — realised value vs Γ   (dotted = matched Γ≈12.18, red dashed = Γ=3 peak)",fontsize=11.5)
fig.tight_layout(); fig.savefig(HERE/"realized_vs_gamma.png",dpi=120); plt.close(fig); print("realized_vs_gamma.png")

# ---- (2) worst-case objective vs Γ ----
fig,ax=plt.subplots(figsize=(8.6,4.7))
for m in METHODS:
    y,e=agg(m,"obj"); ls="--" if m in("IPW","AIPW") else "-"
    ax.plot(G,y,ls+MK[m],color=COL[m],lw=1.8,ms=4,label=m);
    if m not in("IPW","AIPW"): ax.fill_between(G,y-e,y+e,color=COL[m],alpha=.08)
ax.axhline(0,ls=":",c="#999",lw=1); ax.axvline(mG,color="#bbb",ls=":",lw=1.2)
ax.set_xlabel("Γ"); ax.set_ylabel("worst-case in-sample objective (mean ± SD)")
ax.set_title("Worst-case objective vs Γ — value methods report a worst-case value, regret methods a worst-case regret ≤ 0",fontsize=10.3)
ax.legend(fontsize=8.2,ncol=2,loc="best"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(HERE/"objective_vs_gamma.png",dpi=120); plt.close(fig); print("objective_vs_gamma.png")

# ---- (3) policy-strip grid at matched Γ (seed 0) ----
pg=seeds[0]["policy_grid"]; gridX=np.array(seeds[0]["grid"]); rows=["Full-info","Best-means"]+METHODS
M=np.array([np.argmax(np.array(pg[r]),axis=0) for r in rows])     # (n_rows, 21) argmax arm
cmap=ListedColormap(dgp.ARM_COL)
fig,ax=plt.subplots(figsize=(10.0,4.6))
im=ax.imshow(M,aspect="auto",cmap=cmap,vmin=0,vmax=2,extent=[gridX.min(),gridX.max(),len(rows)-0.5,-0.5])
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows,fontsize=9); ax.set_xlabel("X")
ax.set_title("Chosen arm vs X at the matched Γ≈12.18 (seed 0)  —  colour = argmax arm (0 blue, 1 orange, 2 red)",fontsize=10.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=dgp.ARM_COL[k],label="arm %d"%k) for k in range(3)],fontsize=9,loc="upper center",ncol=3,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(); fig.savefig(HERE/"policy.png",dpi=120); plt.close(fig); print("policy.png")

# ---- (4) realised bar at Γ=3 (peak) and matched Γ ----
fig,ax=plt.subplots(1,2,figsize=(12.6,4.7),sharey=True)
for j,(gi,ttl) in enumerate([(gpeak,"Γ=3 (R-OW's peak)"),(mi,"matched Γ≈12.18")]):
    a=ax[j]; ys=[agg(m,"rt_test")[0][gi] for m in METHODS]; es=[agg(m,"rt_test")[1][gi] for m in METHODS]
    order=sorted(range(len(METHODS)),key=lambda i:-ys[i]); ms=[METHODS[i] for i in order]
    a.bar(range(len(ms)),[ys[i] for i in order],yerr=[es[i] for i in order],capsize=3,color=[COL[m] for m in ms],alpha=.88,edgecolor="#333")
    a.axhline(cei["full_info_test"],ls=":",c="#111",lw=1.4,label="Full-info %.3f"%cei["full_info_test"])
    a.axhline(cei["best_means_test"],ls="--",c="#111",lw=1.2,label="Best-means %.3f"%cei["best_means_test"])
    a.set_xticks(range(len(ms))); a.set_xticklabels(ms,rotation=20,fontsize=8.5); a.set_title(ttl,fontsize=11); a.grid(alpha=.25,axis="y")
ax[0].set_ylabel("realised test value (mean ± SD, %d seeds)"%nS); ax[0].set_ylim(0.6,cei["full_info_test"]+0.01); ax[1].legend(fontsize=8.5,loc="lower right")
fig.suptitle("Realised test value per method — capacity-constrained 3-arm exp_a (γ=5)",fontsize=11.5)
fig.tight_layout(); fig.savefig(HERE/"realized_bar.png",dpi=120); plt.close(fig); print("realized_bar.png")

# ---- (5) capacity usage vs Γ (arms 1,2) ----
def use(m): return np.array([[s["methods"][m][str(gi)]["use"] for gi in range(len(G))] for s in seeds]).mean(0)  # (nG,K)
fig,ax=plt.subplots(1,2,figsize=(12.6,4.5),sharey=True)
for j,k in enumerate([1,2]):
    a=ax[j]
    for m in ["R-OW","R-O","R-OW-DR","IPW","AIPW"]:
        a.plot(G,use(m)[:,k],("--" if m in("IPW","AIPW") else "-")+MK[m],color=COL[m],lw=1.7,ms=4,label=m)
    a.axhline(cap[k],ls=":",c="#111",lw=1.6,label="cap_%d = %.3f"%(k,cap[k]))
    a.axvline(mG,color="#bbb",ls=":",lw=1.2); a.set_xlabel("Γ"); a.set_title("Arm %d allocation"%k,fontsize=11); a.grid(alpha=.25)
ax[0].set_ylabel(r"$(1/n)\sum_i \pi_k(x_i)$ (train, mean over seeds)"); ax[1].legend(fontsize=8.5,loc="best")
fig.suptitle("Per-arm capacity usage vs Γ — arms 1,2 are capped at their train share (arm 0 uncapped)",fontsize=11.2)
fig.tight_layout(); fig.savefig(HERE/"capacity_usage.png",dpi=120); plt.close(fig); print("capacity_usage.png")
print("\n=== realised TEST value @ Γ=3 (peak) and matched Γ (mean over %d seeds) ==="%nS)
for m in METHODS: print("  %-10s Γ=3 %.3f   matched %.3f"%(m,agg(m,"rt_test")[0][gpeak],agg(m,"rt_test")[0][mi]))
print("  ceilings: full-info %.3f  best-means %.3f | cap=(%.2f,%.3f,%.3f)"%(cei["full_info_test"],cei["best_means_test"],*cap))
