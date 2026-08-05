#!/usr/bin/env python3
"""The sweep's own top configs, which the first confirmation grid predates.

All three pair beta0=5 (strong prognostic confounding) with delta=2 (the hidden signal also
drives the CATE) -- the combination the 54-config sweep found most separating, and the one my
analytic pre-screen predicted would push the naive decision boundary furthest off.
"""
import sys

CAND = [
    {"tag": "sw34", "a": 1.0, "alpha": 1.0, "delta": 2.0, "beta0": 5.0},
    {"tag": "sw35", "a": 3.0, "alpha": 1.0, "delta": 2.0, "beta0": 5.0},
    {"tag": "sw52", "a": 2.0, "alpha": 2.0, "delta": 2.0, "beta0": 5.0},
]
SEEDS = 10

if __name__ == "__main__":
    if sys.argv[1] == "-n":
        print(len(CAND) * SEEDS)
    else:
        i = int(sys.argv[1]); c = CAND[i // SEEDS]; sd = i % SEEDS
        args = " ".join("--%s %g" % (k, v) for k, v in c.items() if k != "tag")
        print("%s|%s|%d" % (c["tag"], args, sd))
