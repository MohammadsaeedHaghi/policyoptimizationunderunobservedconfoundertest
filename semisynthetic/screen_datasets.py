#!/usr/bin/env python3
"""Apply the paper's dataset-admissibility screen to the UCI candidates.

A dataset is admissible iff its cross-validated logistic log-loss lies in [0.35, log 2].
Below 0.35 the label is already almost perfectly explained by a linear logistic model, so
baselines that ignore propensity would suffice and nothing can be separated; above log 2 the
label is pure noise.

Also reports two quantities the paper does not, but which matter for US:
  headroom   oracle minus the better constant policy, at gamma = 0. If this is ~0 the benchmark
             cannot separate policy learners no matter how the confounding is set (the failure
             mode measured on IHDP/Twins/IST).
  corr(x,U)  how trackable the hidden confounder is from the covariates. Since U IS the label,
             this is determined by exactly what the log-loss screen controls -- worth seeing.
"""
import json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semisynthetic_dgp import (UCI_CANDIDATES, load_uci_dataset, generate_semisynthetic,
                               SCREEN_LO, SCREEN_HI)

HERE = Path(__file__).resolve().parent
rows = []
print("%-20s %7s %5s %10s %6s %10s %10s  %s"
      % ("dataset", "n", "d", "CV logloss", "adm?", "headroom", "corr(x,U)", "note"))
print("-" * 96)
for name in UCI_CANDIDATES:
    try:
        X, y, info = load_uci_dataset(name, enforce_screen=False)
    except Exception as ex:
        print("%-20s %s" % (name, ("LOAD FAILED: " + str(ex)[:60])))
        continue
    hd = cu = float("nan")
    try:                       # gamma = 0 probe: measures the benchmark's intrinsic headroom
        nt = min(1500, max(200, len(y) // 2))
        _o, _orc, meta = generate_semisynthetic(X, y, gamma=0.0, seed=0,
                                                n_train=nt, n_test=min(500, len(y) - nt))
        hd, cu = meta.diagnostics["headroom"], meta.diagnostics["corr_x_U"]
    except Exception as ex:
        pass
    ok = SCREEN_LO <= info["cv_logloss"] <= SCREEN_HI
    rows.append({**info, "headroom": hd, "corr_x_U": cu})
    print("%-20s %7d %5d %10.4f %6s %10.4f %10.3f  %s"
          % (name, info["n"], info["d"], info["cv_logloss"], "YES" if ok else "no", hd, cu,
             "" if ok else ("too separable" if info["cv_logloss"] < SCREEN_LO else "too noisy")))

adm = [r for r in rows if SCREEN_LO <= r["cv_logloss"] <= SCREEN_HI]
adm_ok = [r for r in adm if r["headroom"] == r["headroom"] and r["headroom"] > 0.15]
print("\n%d/%d admissible by the paper's screen; %d of those also have headroom > 0.15"
      % (len(adm), len(rows), len(adm_ok)))
if adm_ok:
    pick = max(adm_ok, key=lambda r: min(r["n"], 20000))     # prefer a large-but-not-huge pool
    print("\nRECOMMENDED: %s  (n=%d, d=%d, CV log-loss %.4f, headroom %+.4f)"
          % (pick["dataset"], pick["n"], pick["d"], pick["cv_logloss"], pick["headroom"]))
json.dump(rows, open(HERE / "screen.json", "w"), indent=1)
print("\nwrote screen.json")
