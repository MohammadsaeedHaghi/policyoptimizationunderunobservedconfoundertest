"""Plots for the real-data experiments (IHDP + LaLonde). Reads ihdp_results.json + lalonde_results.json.
Saves: ihdp_bars.png, lalonde_confounding.png, lalonde_bias.png, lalonde_realized.png, lalonde_policy.png."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")/"assets"/"exp_realdata"
COL={"R-OW":"#d62728","R-O":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","R-OW-DR":"#a01f1f","Kallus":"#7f7f7f"}
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":.25})

# ---------------- IHDP: fraction of oracle gain captured ----------------
ih=json.loads((HERE/"ihdp_results.json").read_text()); nz=ih["normalized"]
ms=[m for m in ["AIPW","R-OW-DR","R-OW","R-O","Kallus","IPW"]]
fig,ax=plt.subplots(figsize=(7.6,4.2))
y=[nz[m]["frac_mean"]*100 for m in ms]; e=[nz[m]["frac_sd"]*100 for m in ms]
ax.bar(range(len(ms)),y,yerr=e,capsize=4,color=[COL[m] for m in ms],alpha=.85,edgecolor="#333")
ax.axhline(nz["_treatall_frac"]*100,ls="--",c="#444",lw=1.4,label="treat-everyone baseline (%.0f%%)"%(nz["_treatall_frac"]*100))
ax.axhline(100,ls=":",c="green",lw=1.2,label="oracle (100%)")
ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_ylim(60,105)
ax.set_ylabel("% of oracle gain captured\n(method−ctrl)/(oracle−ctrl)")
ax.set_title("IHDP (10 reps): every method ≈ treat-everyone, near the oracle\nweak confounding (max |corr(T,x)|≈%.2f) does not separate the methods"%ih["conf_maxcorr"])
ax.legend(fontsize=9,loc="lower right"); fig.tight_layout(); fig.savefig(HERE/"ihdp_bars.png",dpi=120); plt.close(fig)
print("ihdp_bars.png")

# ---------------- LaLonde ----------------
la=json.loads((HERE/"lalonde_results.json").read_text())
# (1) confounding diagnostic: treated(NSW) vs control(PSID), hidden earnings highlighted
diag=la["diag"]; covs=list(diag.keys())
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
earn=["re74","re75"]; xe=np.arange(len(earn)); wbar=.36
a1.bar(xe-wbar/2,[diag[c][0] for c in earn],wbar,label="treated (NSW trainees)",color="#d62728",alpha=.85,edgecolor="#333")
a1.bar(xe+wbar/2,[diag[c][1] for c in earn],wbar,label="control (PSID)",color="#7f7f7f",alpha=.85,edgecolor="#333")
a1.set_xticks(xe); a1.set_xticklabels(["re74 (1974 $)","re75 (1975 $)"]); a1.set_ylabel("mean prior earnings ($)")
a1.set_title("HIDDEN confounders: ~10× earnings gap\n(treated are disadvantaged; controls are general-population)")
a1.legend(fontsize=9)
other=["age","education","black","married","nodegree"]; xo=np.arange(len(other))
a2.bar(xo-wbar/2,[diag[c][0] for c in other],wbar,label="treated (NSW)",color="#d62728",alpha=.85,edgecolor="#333")
a2.bar(xo+wbar/2,[diag[c][1] for c in other],wbar,label="control (PSID)",color="#7f7f7f",alpha=.85,edgecolor="#333")
a2.set_xticks(xo); a2.set_xticklabels(other,rotation=20); a2.set_title("Observed covariates also differ\n(only education is shown to the methods)"); a2.legend(fontsize=9)
fig.tight_layout(); fig.savefig(HERE/"lalonde_confounding.png",dpi=120); plt.close(fig); print("lalonde_confounding.png")

# (2) the famous bias: naive observational ATE vs experimental truth
fig,ax=plt.subplots(figsize=(6.2,4.3))
vals=[la["naive_obs_ate"],la["exp_ate"]]; cols=["#7f7f7f","#2ca02c"]
b=ax.bar([0,1],vals,color=cols,alpha=.85,edgecolor="#333",width=.6)
ax.axhline(0,c="#333",lw=1); ax.set_xticks([0,1])
ax.set_xticklabels(["naive observational\n(NSW vs PSID, confounded)","experimental truth\n(randomised NSW)"])
ax.set_ylabel("estimated ATE on re78 ($1000s)")
for i,v in enumerate(vals): ax.text(i,v+(.4 if v>0 else -.9),f"${v*1000:,.0f}",ha="center",fontweight="bold")
ax.set_title("The famous LaLonde bias: hiding prior earnings makes the\nbeneficial program look catastrophically harmful")
fig.tight_layout(); fig.savefig(HERE/"lalonde_bias.png",dpi=120); plt.close(fig); print("lalonde_bias.png")

# (3) realized value vs Γ per method, with reference lines
G=la["gammas"]
fig,ax=plt.subplots(figsize=(8.4,4.8))
for m in ["R-OW","R-OW-DR","R-O","AIPW","IPW","Kallus"]:
    md=la["methods"][m]; y=md["realized_mean"]; e=md["realized_sd"]
    ax.plot(G,y,"-o",color=COL[m],label=m,lw=1.8,ms=4)
    ax.fill_between(G,np.array(y)-np.array(e),np.array(y)+np.array(e),color=COL[m],alpha=.10)
ax.axhline(la["treatall"],ls="--",c="#2ca02c",lw=1.4,label="treat-everyone (truth-optimal ≈ %.2f)"%la["treatall"])
ax.axhline(la["ctrl"],ls=":",c="#444",lw=1.4,label="never-treat (ctrl = %.2f)"%la["ctrl"])
ax.set_xlabel("Γ (assumed sensitivity)"); ax.set_ylabel("realised policy value on randomised NSW ($1000s)")
ax.set_title("LaLonde: train on confounded observational data, test on the randomised experiment\nAll methods hedge to never-treat — strong confounding defeats observational policy learning",fontsize=10.5)
ax.legend(fontsize=8.5,ncol=2,loc="best"); fig.tight_layout(); fig.savefig(HERE/"lalonde_realized.png",dpi=120); plt.close(fig); print("lalonde_realized.png")

# (4) policy π(treat|education) at a mid Γ + experimental ATE-by-bin
gi=G.index(3.0) if 3.0 in G else len(G)//2
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
xb=np.arange(la["bins"]); lab=["≤8","9","10","11","12","≥13"][:la["bins"]]
for m in ["R-OW","AIPW","IPW","Kallus"]:
    a1.plot(xb,la["methods"][m]["policy_mean"][gi],"-o",color=COL[m],label=m,lw=1.7,ms=5)
a1.set_xticks(xb); a1.set_xticklabels(lab); a1.set_ylim(-.05,1.05)
a1.set_xlabel("education (years)"); a1.set_ylabel("P(treat | education)")
a1.set_title("Learned policy at Γ=%g — methods mostly refuse to treat"%G[gi]); a1.legend(fontsize=9)
ab=la["ate_by_bin"]; a2.bar(xb,ab,color=["#d62728" if v<0 else "#2ca02c" for v in ab],alpha=.8,edgecolor="#333")
a2.axhline(0,c="#333",lw=1); a2.set_xticks(xb); a2.set_xticklabels(lab)
a2.set_xlabel("education (years)"); a2.set_ylabel("experimental ATE in bin ($1000s)")
a2.set_title("Ground-truth benefit gradient (randomised NSW):\ntreatment helps most at high education")
fig.tight_layout(); fig.savefig(HERE/"lalonde_policy.png",dpi=120); plt.close(fig); print("lalonde_policy.png")
print("all real-data plots done")
