"""Post-process IHDP: the raw realised values are dominated by cross-rep outcome-SCALE variation (rep means span 8..50),
so a raw mean±SD across reps is misleading. Recompute per-rep ctrl/oracle (cheap — just mu0/mu1, no solver, same split
as ihdp_run.py) and report each method as the FRACTION OF THE ORACLE GAIN it captures: (method−ctrl)/(oracle−ctrl)∈[0,1].
This is scale-free and comparable across reps. Augments ihdp_results.json with a 'normalized' block."""
import json, numpy as np, pandas as pd
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
DATA=NEW/"assets"/"exp_realdata"/"data"; NB=10; OBS=6
cols=['treat','yf','ycf','mu0','mu1']+['x%d'%i for i in range(1,26)]
res=json.loads((NEW/"assets"/"exp_realdata"/"ihdp_results.json").read_text())
METHODS=list(res["methods"].keys()); frac={m:[] for m in METHODS}
ctrls=[];orcs=[];tas=[]
for ri,rep in enumerate(range(1,11)):
    d=pd.read_csv(DATA/f"ihdp_{rep}.csv",header=None,names=cols)
    mu0=d['mu0'].values.astype(float); mu1=d['mu1'].values.astype(float)
    rng=np.random.default_rng(rep); idx=rng.permutation(len(d)); ntr=int(0.6*len(d)); tei=idx[ntr:]
    m0,m1=mu0[tei],mu1[tei]; c=float(m0.mean()); o=float(np.maximum(m0,m1).mean()); ta=float(m1.mean())
    ctrls.append(c);orcs.append(o);tas.append(ta)
    for m in METHODS:
        v=res["methods"][m]["per_rep"][ri]; frac[m].append((v-c)/(o-c) if o>c else np.nan)
res["normalized"]={m:{"frac_mean":float(np.nanmean(frac[m])),"frac_sd":float(np.nanstd(frac[m])),
                      "frac_per_rep":[float(x) for x in frac[m]]} for m in METHODS}
res["normalized"]["_treatall_frac"]=float(np.nanmean([(tas[i]-ctrls[i])/(orcs[i]-ctrls[i]) for i in range(10)]))
(NEW/"assets"/"exp_realdata"/"ihdp_results.json").write_text(json.dumps(res,indent=1))
print("IHDP — fraction of oracle gain captured (mean±SD over 10 reps), scale-free:")
for m in sorted(METHODS,key=lambda m:-res["normalized"][m]["frac_mean"]):
    print(f"  {m:9s} {res['normalized'][m]['frac_mean']*100:5.1f}% ± {res['normalized'][m]['frac_sd']*100:.1f}%")
print(f"  (treat-everyone baseline captures {res['normalized']['_treatall_frac']*100:.1f}% — methods cluster around it)")
print("  spread between best & worst method per rep:",
      [round((max(res['methods'][m]['per_rep'][i] for m in METHODS)-min(res['methods'][m]['per_rep'][i] for m in METHODS)),3) for i in range(10)])
