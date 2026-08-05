#!/usr/bin/env python3
"""Rank the smoke-sweep configs by whether the O-W family actually wins, and by how much."""
import json, glob
import numpy as np

rows = []
for f in sorted(glob.glob("smoke/cfg*.json")):
    d = json.load(open(f))
    r = d["rows"]
    ow = max((r[m]["norm"] for m in ("IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W") if m in r), default=-9)
    ox = max((r[m]["norm"] for m in ("IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X") if m in r), default=-9)
    xx = max((r[m]["norm"] for m in ("IPW-X-X", "DoublyRobust-X-X", "Direct-X-X") if m in r), default=-9)
    nv = r.get("naive", {}).get("norm", -9)
    marg = d.get("margin", {}).get("IPW-O-W", {}).get("mean", float("nan"))
    rows.append({"tag": d["tag"], "cfg": d["cfg"], "hr": d["headroom"], "ow": ow, "ox": ox,
                 "xx": xx, "naive": nv, "gap_ox": ow - ox, "gap_xx": ow - xx,
                 "gap_naive": ow - nv, "margin": marg, "pass": d.get("pass", False)})

if not rows:
    print("no results yet"); raise SystemExit

rows.sort(key=lambda r: (r["pass"], min(r["gap_ox"], r["gap_xx"], r["gap_naive"])), reverse=True)
print("%-7s %5s %5s %5s %5s %6s | %6s %6s %6s %6s %7s %4s"
      % ("cfg", "alpha", "delta", "beta0", "a", "headrm", "O-W", "O-X", "X-X", "naive",
         "minGap", "pass"))
print("-" * 92)
for r in rows[:24]:
    c = r["cfg"]
    print("%-7s %5g %5g %5g %5g %6.3f | %6.3f %6.3f %6.3f %6.3f %7.3f %4s"
          % (r["tag"], c["alpha"], c["delta"], c["beta0"], c["a"], r["hr"], r["ow"], r["ox"],
             r["xx"], r["naive"], min(r["gap_ox"], r["gap_xx"], r["gap_naive"]),
             "YES" if r["pass"] else ""))
npass = sum(1 for r in rows if r["pass"])
print("\n%d/%d configs complete, %d PASS (O-W beats O-X, X-X and naive, headroom > 0.05)"
      % (len(rows), 54, npass))
