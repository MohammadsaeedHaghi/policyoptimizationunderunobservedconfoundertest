#!/usr/bin/env python3
"""The knob grid the smoke sweep walks. Printed one config per line so an sbatch array can index it."""
import json, sys

GRID = []
for alpha in (0.0, 1.0, 2.0):          # coupling corr(x, S)
    for delta in (0.0, 1.0, 2.0):      # hidden signal's effect on the CATE
        for beta0 in (0.0, 2.0, 5.0):  # hidden signal's effect on the baseline (prognostic)
            for a in (1.0, 3.0):       # strength of the rational x-response
                GRID.append({"alpha": alpha, "delta": delta, "beta0": beta0, "a": a})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-n":
        print(len(GRID))
    else:
        i = int(sys.argv[1])
        c = GRID[i]
        print(" ".join("--%s %g" % (k, v) for k, v in c.items()))
