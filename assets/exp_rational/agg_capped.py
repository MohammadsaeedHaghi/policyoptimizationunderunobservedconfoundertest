#!/usr/bin/env python3
"""Aggregate the capped cells, one table per (config, cap)."""
import json, glob, collections
import numpy as np

OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]

byk = collections.defaultdict(list)
for f in glob.glob("capped/*_s*.json"):
    d = json.load(open(f)); byk[(d["tag"], d["cap"])].append(d["cell"])

def nz(v, c): return (v - c["bc"]) / c["sc"]

for (tag, cap), cs in sorted(byk.items()):
    rows = {}
    for m in OX + OW:
        cand = []
        for ce in cs[0]["grid"].get(m, {}):
            for lk in cs[0]["grid"][m][ce]:
                vs = [nz(c["grid"][m][ce][lk], c) for c in cs if lk in c["grid"][m].get(ce, {})]
                if len(vs) == len(cs): cand.append((float(np.mean(vs)), float(np.std(vs)), ce, lk))
        if cand: rows[m] = max(cand)
    for m in XX:
        cand = []
        for lk in cs[0]["xx"].get(m, {}):
            vs = [nz(c["xx"][m][lk], c) for c in cs if lk in c["xx"].get(m, {})]
            if len(vs) == len(cs): cand.append((float(np.mean(vs)), float(np.std(vs)), "-", lk))
        if cand: rows[m] = max(cand)
    nv = [nz(c["naive"], c) for c in cs]
    rows["naive"] = (float(np.mean(nv)), float(np.std(nv)), "-", "-")

    marg = {}
    for a, b in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
        dd = []
        for ce in cs[0]["grid"].get(a, {}):
            for lk in cs[0]["grid"][a][ce]:
                pa = [nz(c["grid"][a][ce][lk], c) for c in cs if lk in c["grid"][a].get(ce, {})]
                pb = [nz(c["grid"][b]["1"][lk], c) for c in cs if lk in c["grid"][b].get("1", {})]
                if len(pa) == len(pb) == len(cs): dd.append(float(np.mean(np.array(pa) - np.array(pb))))
        if dd: marg[a] = (float(np.mean(dd)), float(np.max(dd)))

    ow = max(rows[m][0] for m in OW if m in rows)
    gaps = {k: ow - rows[k][0] for k in OX + XX + ["naive"] if k in rows}
    print("=" * 78)
    print("%s   cap=%.0f%%   (%d seeds, n=400)   headroom %.3f"
          % (tag, 100 * cap, len(cs), float(np.mean([c["sc"] for c in cs]))))
    for m in OW + OX + XX + ["naive"]:
        if m in rows:
            v, sd, ce, lk = rows[m]
            print("   %-20s %7.3f +-%.3f   c_eps %-4s L %-4s" % (m, v, sd, ce, lk))
    print("   transport margin (same L, c_eps): %s"
          % {k: "mean %+.3f / max %+.3f" % (v[0], v[1]) for k, v in marg.items()})
    print("   O-W best %.3f, min gap %+.3f %s"
          % (ow, min(gaps.values()), "<-- PASS" if min(gaps.values()) > 0 else ""))
