#!/usr/bin/env python3
"""Summarise the realised overlap e(x,u) across every prepared population.

The index stores only e_min / e_max, and those are a poor summary here: they are extremes over
25 populations x 20k rows = 500k units, and at gamma=0 -- where there is NO confounding and e
should sit near 1/2 -- the minimum is already 0.0014. That is one outlier row in the standardised
UCI covariates (some features are heavy-tailed, so lambda'x can reach several units even with
lambda_j ~ U(-0.1, 0.1)), not a property of the design.

Percentiles show what the assignment actually looks like. Writes prepared/_overlap.json so the
report builder does not have to load 84 MB of npz at build time.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "prepared" / "_overlap.json"

acc = {}
for f in sorted((HERE / "prepared").glob("*_pop.npz")):
    # filename is {dataset}_g{gamma}_d{seed}_pop.npz with gamma written by "%g"
    stem = f.stem[:-4]
    g = float(stem.rsplit("_d", 1)[0].rsplit("_g", 1)[1])
    e = np.load(f)["e"].ravel()
    acc.setdefault(g, []).append(e)

rows = {}
for g, es in sorted(acc.items()):
    e = np.concatenate(es)
    rows["%g" % g] = {
        "gamma": g, "Gamma": float(np.exp(2 * g)), "n_pop": len(es), "n_units": int(e.size),
        "min": float(e.min()), "p01": float(np.percentile(e, 1)),
        "p05": float(np.percentile(e, 5)), "p50": float(np.percentile(e, 50)),
        "p95": float(np.percentile(e, 95)), "p99": float(np.percentile(e, 99)),
        "max": float(e.max()),
        "frac_below_01": float((e < 0.01).mean()), "frac_above_99": float((e > 0.99).mean()),
    }
    r = rows["%g" % g]
    print("gamma=%-4g Gamma=%8.1f | e: min %.4f  [p1 %.3f, p50 %.3f, p99 %.3f]  max %.4f | "
          "frac<0.01 %.4f  frac>0.99 %.4f"
          % (g, r["Gamma"], r["min"], r["p01"], r["p50"], r["p99"], r["max"],
             r["frac_below_01"], r["frac_above_99"]))

OUT.write_text(json.dumps(rows, indent=1))
print("\nwrote %s (%d gamma levels)" % (OUT, len(rows)))
