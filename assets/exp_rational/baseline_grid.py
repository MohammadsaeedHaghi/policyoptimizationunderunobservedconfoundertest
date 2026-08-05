#!/usr/bin/env python3
"""Configs to score the fixed Kallus / Hess baselines on: every candidate the confirmation ran."""
import sys

CAND = [
    {"tag": "nocouple", "a": 2.0, "alpha": 0.0, "delta": 1.0, "beta0": 2.0},   # RECOMMENDED
    {"tag": "base",     "a": 2.0, "alpha": 1.0, "delta": 1.0, "beta0": 2.0},
    {"tag": "sw35",     "a": 3.0, "alpha": 1.0, "delta": 2.0, "beta0": 5.0},
    {"tag": "sw52",     "a": 2.0, "alpha": 2.0, "delta": 2.0, "beta0": 5.0},
    {"tag": "sw34",     "a": 1.0, "alpha": 1.0, "delta": 2.0, "beta0": 5.0},
    {"tag": "delta2",   "a": 2.0, "alpha": 1.0, "delta": 2.0, "beta0": 2.0},
    {"tag": "a15",      "a": 1.5, "alpha": 1.0, "delta": 1.0, "beta0": 2.0},
    {"tag": "a3",       "a": 3.0, "alpha": 1.0, "delta": 1.0, "beta0": 2.0},
    {"tag": "b0_5",     "a": 2.0, "alpha": 1.0, "delta": 1.0, "beta0": 5.0},
    {"tag": "cfg44",    "a": 1.0, "alpha": 2.0, "delta": 1.0, "beta0": 2.0},
    {"tag": "bsx4",     "a": 2.0, "alpha": 1.0, "delta": 1.0, "beta0": 2.0, "bsx": 4.0},
]
SEEDS = 10

if __name__ == "__main__":
    if sys.argv[1] == "-n":
        print(len(CAND) * SEEDS)
    else:
        i = int(sys.argv[1]); c = CAND[i // SEEDS]; sd = i % SEEDS
        args = " ".join("--%s %g" % (k, v) for k, v in c.items() if k != "tag")
        print("%s|%s|%d" % (c["tag"], args, sd))
