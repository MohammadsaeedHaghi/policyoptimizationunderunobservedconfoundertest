#!/usr/bin/env python3
"""Recompute the published-baseline entries of existing capped cells with the PAPER-EXACT Hess.

The LP arms are untouched -- re-solving them would buy nothing. Draws are deterministic in
(cfg, seed), so the budgeted evaluation is reproduced exactly: rank test units by the method's
own deployed score, treat the top cap fraction."""
import sys, json, argparse
from pathlib import Path
import numpy as np

ROOT = Path("/home1/haghim/code 1.1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "methods" / "SharpHess"))
sys.path.insert(0, str(HERE))
from dgp_rational import Cfg, draw
import sharp_hess as H
import importlib.util


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--file", required=True)
    a = ap.parse_args()
    p = Path(a.file); d = json.loads(p.read_text())
    cfg = Cfg(**d["cfg"]); sd = int(d["seed"]); cap = float(d["cap"])
    tr = draw(d["n_train"], 1000 + sd, cfg); te = draw(4000, 90000 + sd, cfg)
    X, T, Yraw = tr["x"].reshape(-1, 1), tr["T"], tr["Y"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)
    Y0t, Y1t = te["Y0"], te["Y1"]
    n_te = len(Y1t); k = int(np.floor(cap * n_te))

    def eval_capped(score):
        sc = np.asarray(score, float).ravel(); pi = np.zeros(n_te)
        if k > 0: pi[np.argsort(-sc)[:k]] = 1.0
        return float(np.mean(pi * Y1t + (1 - pi) * Y0t))

    bl = d["cell"].setdefault("baselines", {})
    if "baselines_prev" not in d["cell"]:
        d["cell"]["baselines_prev"] = dict(bl)          # keep the custom-trainer numbers
    pol = H.hess_paper(X, T, Y, Gamma=float(cfg.Gstar), seed=sd, maximize=True)
    bl["SharpHess"] = eval_capped(H.apply_hess_paper(pol, te["x"].reshape(-1, 1)))
    p.write_text(json.dumps(d))
    print("%s: SharpHess(paper) capped -> %+.4f" % (p.name, bl["SharpHess"]), flush=True)


if __name__ == "__main__":
    main()
