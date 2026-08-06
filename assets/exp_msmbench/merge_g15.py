#!/usr/bin/env python3
"""Merge the Gamma=15 column into kmz_main_ce1.0.json (backup kept as *_pre15.json).

The Gamma=15 run writes its own full-shape file; this folds its surface/best columns into the
main campaign file so every consumer (the KMZ report widget, the semisynth KMZ tab) sees one
consistent grid. Refs are asserted equal before touching anything -- same DGP, same seeds, so
oracle/never/all must match to rounding.
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "kmz_main_ce1.0.json"
G15 = HERE / "kmz_main_g15only_ce1.0.json"

m = json.loads(MAIN.read_text())
g = json.loads(G15.read_text())

for k in ("oracle", "never_treat", "all_treat"):
    assert abs(m[k] - g[k]) < 1e-6, "refs differ on %s: %s vs %s" % (k, m[k], g[k])

if "15" in m["gammas"]:
    print("Gamma=15 already merged; nothing to do")
    raise SystemExit

shutil.copy(MAIN, HERE / "kmz_main_ce1.0_pre15.json")
gk = [k for k in g["surface"]["IPW-O-W"].keys()]
assert gk == ["15"], gk
for meth in m["methods"]:
    m["surface"][meth]["15"] = g["surface"][meth]["15"]
    b_new = max((g["surface"][meth]["15"][L], L) for L in g["Lgrid"])
    if b_new[0] > m["best"][meth]["value"]:
        m["best"][meth] = {"gamma": "15", "L": b_new[1], "value": round(b_new[0], 4)}
m["gammas"] = m["gammas"] + ["15"]
for key in ("policies_by_seed", "policies_support_by_seed"):
    if key in m and key in g:
        for sd in m[key]:
            for meth in m[key][sd]:
                if isinstance(m[key][sd][meth], dict) and meth in g[key].get(sd, {}):
                    m[key][sd][meth].update({kk: vv for kk, vv in g[key][sd][meth].items()
                                             if kk == "15"})
MAIN.write_text(json.dumps(m))
print("merged Gamma=15 into", MAIN.name, "| gammas now", m["gammas"])
