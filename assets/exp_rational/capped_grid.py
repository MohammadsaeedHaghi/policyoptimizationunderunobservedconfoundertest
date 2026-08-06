#!/usr/bin/env python3
"""Capped runs on configs screened for BOTH ranking corruption and signal-to-noise.

The first attempt raised beta0 to 26 to corrupt the ranking and destroyed the SNR doing it: the
apparent bias is beta0*Delta(x) while the outcome noise is beta0*sd(S), so beta0 buys corruption
and noise in equal measure and cannot improve the ratio. The lever that CAN is `alpha`, which makes
Delta(x) vary with x -- where Pr(S=+1|x) saturates there is no S variation and hence no bias, so
the bias becomes a localised bump rather than a level shift. Both configs below hold SNR near the
0.42 of the uncapped winner while cutting the naive top-30% overlap to 0.17 and 0.25.
"""
import sys

CAND = [
    {"tag": "cap_primary", "a": 2.0, "alpha": 6.0,  "delta": 1.0, "beta0": 4.0},
    {"tag": "cap_second",  "a": 2.0, "alpha": 10.0, "delta": 1.0, "beta0": 6.0},
]
CAPS = [0.3, 0.2]
SEEDS = 10

if __name__ == "__main__":
    if sys.argv[1] == "-n":
        print(len(CAND) * len(CAPS) * SEEDS)
    else:
        i = int(sys.argv[1])
        c = CAND[i // (len(CAPS) * SEEDS)]
        r = i % (len(CAPS) * SEEDS)
        cap = CAPS[r // SEEDS]; sd = r % SEEDS
        args = " ".join(("--%s %s" % (k, v)) if isinstance(v, str) else ("--%s %g" % (k, v))
                        for k, v in c.items() if k != "tag")
        print("%s|%s --cap %g|%d|%g" % (c["tag"], args, cap, sd, cap))
