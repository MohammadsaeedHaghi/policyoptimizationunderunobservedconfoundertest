#!/usr/bin/env python3
"""Aggregate the per-seed confirmation cells into one table per candidate config."""
import json, glob, collections
import numpy as np

OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]

byt = collections.defaultdict(list)
for f in glob.glob("confirm/*_s*.json"):
    d = json.load(open(f)); byt[d["tag"]].append(d)

def nz(v, c): return (v - c["bc"]) / c["sc"]

out = []
for tag, ds in sorted(byt.items()):
    cells = [d["cell"] for d in ds]
    cfg = ds[0]["cfg"]
    rows = {}
    for m in OX + OW:
        best = None
        for ce in cells[0]["grid"].get(m, {}):
            for lk in cells[0]["grid"][m][ce]:
                vs = [nz(c["grid"][m][ce][lk], c) for c in cells
                      if lk in c["grid"].get(m, {}).get(ce, {})]
                if len(vs) < len(cells): continue
                if best is None or np.mean(vs) > best[0]:
                    best = (float(np.mean(vs)), ce, lk, float(np.std(vs)))
        if best: rows[m] = best
    for m in XX:
        best = None
        for lk in cells[0]["xx"].get(m, {}):
            vs = [nz(c["xx"][m][lk], c) for c in cells if lk in c["xx"].get(m, {})]
            if len(vs) < len(cells): continue
            if best is None or np.mean(vs) > best[0]:
                best = (float(np.mean(vs)), "-", lk, float(np.std(vs)))
        if best: rows[m] = best
    nvs = [nz(c["naive"], c) for c in cells]
    rows["naive"] = (float(np.mean(nvs)), "-", "-", float(np.std(nvs)))

    # transport margin at IDENTICAL (c_eps, L), paired across seeds
    marg = {}
    for a, b in (("IPW-O-W", "IPW-O-X"), ("DoublyRobust-O-W", "DoublyRobust-O-X")):
        dd = []
        for ce in cells[0]["grid"].get(a, {}):
            for lk in cells[0]["grid"][a][ce]:
                pa = [nz(c["grid"][a][ce][lk], c) for c in cells if lk in c["grid"][a].get(ce, {})]
                pb = [nz(c["grid"][b]["1"][lk], c) for c in cells if lk in c["grid"][b].get("1", {})]
                if len(pa) == len(pb) == len(cells): dd.append(np.mean(np.array(pa) - np.array(pb)))
        if dd: marg[a] = (float(np.max(dd)), float(np.mean(dd)))

    ow = max(rows[m][0] for m in OW if m in rows)
    gaps = {k: ow - rows[k][0] for k in OX + XX + ["naive"] if k in rows}
    hr = float(np.mean([c["sc"] for c in cells]))
    out.append((tag, cfg, len(cells), hr, rows, marg, ow, gaps, min(gaps.values())))

out.sort(key=lambda r: r[-1], reverse=True)
for tag, cfg, ns, hr, rows, marg, ow, gaps, mg in out:
    print("=" * 84)
    print("%s  (%d seeds, n=400)  a=%g alpha=%g delta=%g beta0=%g bsx=%g | headroom %.3f"
          % (tag, ns, cfg["a"], cfg["alpha"], cfg["delta"], cfg["beta0"], cfg.get("bsx", 0), hr))
    for m in OW + OX + XX + ["naive"]:
        if m in rows:
            v, ce, lk, sd = rows[m]
            print("   %-20s %7.3f +-%.3f   c_eps %-4s L %-4s" % (m, v, sd, ce, lk))
    print("   transport margin (same L, c_eps): %s"
          % {k: "max %+.3f / mean %+.3f" % v for k, v in marg.items()})
    print("   O-W best %.3f, min gap over all rivals %+.3f %s"
          % (ow, mg, "<-- PASS" if mg > 0 and hr > 0.05 else ""))
