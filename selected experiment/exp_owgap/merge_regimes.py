#!/usr/bin/env python3
# Merge two single-regime result files (uncap, cap) into one standard per-epsilon file.
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
out = dict(a)
out["regimes"] = dict(a.get("regimes", {})); out["regimes"].update(b.get("regimes", {}))
json.dump(out, open(sys.argv[3], "w"), indent=2)
print("merged -> %s  regimes=%s" % (sys.argv[3], list(out["regimes"].keys())))
