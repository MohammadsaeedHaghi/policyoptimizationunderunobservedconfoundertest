"""Bars plot for the 4-binary-feature IHDP variant. Reads ihdp4_results.json -> ihdp4_bars.png."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
HERE=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")/"assets"/"exp_realdata"
COL={"R-OW":"#d62728","R-O":"#9467bd","IPW":"#2ca02c","AIPW":"#ff7f0e","R-OW-DR":"#a01f1f","Kallus":"#7f7f7f"}
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":.25})
ih=json.loads((HERE/"ihdp4_results.json").read_text()); nz=ih["normalized"]
ms=sorted(["AIPW","R-OW-DR","R-OW","R-O","Kallus","IPW"],key=lambda m:-nz[m]["frac_mean"])
fig,ax=plt.subplots(figsize=(7.8,4.4))
y=[nz[m]["frac_mean"]*100 for m in ms]; e=[nz[m]["frac_sd"]*100 for m in ms]
ax.bar(range(len(ms)),y,yerr=e,capsize=4,color=[COL[m] for m in ms],alpha=.85,edgecolor="#333")
ax.axhline(nz["_treatall_frac"]*100,ls="--",c="#444",lw=1.4,label="treat-everyone baseline (%.0f%%)"%(nz["_treatall_frac"]*100))
ax.axhline(100,ls=":",c="green",lw=1.2,label="oracle (100%)")
ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_ylim(60,105)
ax.set_ylabel("% of oracle gain captured\n(method−ctrl)/(oracle−ctrl)")
ax.set_title("IHDP — 4 binary features observed (%s), 21 hidden\nstill clustered near treat-everyone; hidden-confounding stays mild (≈%.2f)"%(
    ", ".join(ih["obs_features"]), ih["conf_maxcorr_hidden"]),fontsize=10.5)
ax.legend(fontsize=9,loc="lower right"); fig.tight_layout(); fig.savefig(HERE/"ihdp4_bars.png",dpi=120); plt.close(fig)
print("ihdp4_bars.png written")
