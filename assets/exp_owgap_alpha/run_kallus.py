#!/usr/bin/env python3
"""Kallus & Zhou parametric baseline on an owgap-contract DGP (box-only, NO Gurobi session).

Runs the parametric softmax Kallus regret-minimiser (methods/Kallus/kallus.py) with the
marginal-sensitivity box ONLY (wasserstein=False): the inner worst case is the pure-numpy
Dinkelbach root-find (common.hajek_regret.selfnorm_box_dinkelbach), so no Gurobi model is
ever built. fit_kallus() still calls configure_gurobi_license() -- which starts+disposes a
WLS test session -- so we monkeypatch it to a no-op here; that makes this runner safe to run
CONCURRENTLY with the 2-worker LP sbatch chain (the WLS baseline-2 limit is account-wide).

Kallus is FLAT (uncapped only; a smooth softmax cannot enforce a hard capacity), so the
output has a single "uncap" regime. JSON shape mirrors run_experiment_parallel.py:
  {N_train, seeds, gammas, oracle, grid, regimes:{uncap:{mean,sd,obj,obj_sd,policy_seed0,policy_by_seed}}}
with the one method key "Kallus".

Usage:
  python3 assets/exp_owgap_alpha/run_kallus.py --dgp assets/exp_owgap/dgp.py \
      --out assets/exp_owgap/owgap_kallus_20seed.json --n 600 --seeds 20 --gammas 1,1.5,2,2.5,3,4,5,6,8
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import common


def _load_mod(path, name):
    sp = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m


def _gk(g):
    g = float(g); return str(int(g)) if g == int(g) else str(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--gammas", default="1,1.5,2,2.5,3,4,5,6,8")
    ap.add_argument("--eval", default="grid", choices=["grid", "draws"], dest="evalmode",
                    help="grid: exact_value on the uniform LEVELS grid (owgap-style DGPs). "
                         "draws: realized value on a fresh d.generate() test draw -- required "
                         "when the DGP's X-marginal is NOT uniform on LEVELS (e.g. diabetes).")
    ap.add_argument("--n-test", type=int, default=4000, dest="ntest")
    a = ap.parse_args()
    gammas = [float(x) for x in a.gammas.split(",")]; seeds = list(range(a.seeds))

    dgp_path = str((ROOT / a.dgp).resolve()) if not os.path.isabs(a.dgp) else a.dgp
    d = _load_mod(dgp_path, "kallus_dgp")
    kal = _load_mod(ROOT / "methods" / "Kallus" / "kallus.py", "kallus_mod")
    kal.configure_gurobi_license = lambda: None   # box-only path builds no Gurobi model (see docstring)

    LV = np.asarray(d.LEVELS, float); K = d.K
    if a.evalmode == "grid":
        orc = d.exact_value(d.grid_truth()["oracle"])
    per = {}                                       # seedkey -> {"val":[by gamma], "obj":[...], "pol":{gk:grid}}
    t0 = time.time()
    orc_draws = []
    for sd in seeds:
        obs, _ = d.generate(a.n, sd)
        if a.evalmode == "draws":
            te, ft = d.generate(a.ntest, sd + 1000)
            Xte, Y1t, Y0t = te["X"], ft["Y1"], ft["Y0"]
            po = d.oracle_policy(Xte.ravel())
            orc_draws.append(float(np.mean(po * Y1t + (1 - po) * Y0t)))
        wraw, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K, normalize=False)
        out = {"val": [], "obj": [], "pol": {}}
        for g in gammas:
            res = kal.fit_kallus(obs["X"], obs["T"], obs["Y"], wraw, n_arms=K, Gamma=g,
                                 maximize=True, wasserstein=False, seed=sd)
            pi = kal.predict_kallus(res.theta, LV.reshape(-1, 1))[:, 1]
            if a.evalmode == "draws":
                pe = kal.predict_kallus(res.theta, Xte)[:, 1]
                out["val"].append(round(float(np.mean(pe * Y1t + (1 - pe) * Y0t)), 4))
            else:
                out["val"].append(round(float(d.exact_value(pi)), 4))
            out["obj"].append(round(float(res.objective_value), 4))
            out["pol"][_gk(g)] = [round(float(v), 4) for v in pi]
        per[str(sd)] = out
        print("seed %d done (%.1f min elapsed)  val=%s" % (sd, (time.time() - t0) / 60, out["val"]), flush=True)

    mean = [round(float(np.mean([per[str(s)]["val"][gi] for s in seeds])), 4) for gi in range(len(gammas))]
    sd_ = [round(float(np.std([per[str(s)]["val"][gi] for s in seeds])), 4) for gi in range(len(gammas))]
    objm = [round(float(np.mean([per[str(s)]["obj"][gi] for s in seeds])), 4) for gi in range(len(gammas))]
    objsd = [round(float(np.std([per[str(s)]["obj"][gi] for s in seeds])), 4) for gi in range(len(gammas))]
    polS = {str(s): per[str(s)]["pol"] for s in seeds}
    polS["avg"] = {_gk(g): [round(float(np.mean([polS[str(s)][_gk(g)][j] for s in seeds])), 4)
                            for j in range(len(LV))] for g in gammas}
    if a.evalmode == "draws":
        orc = float(np.mean(orc_draws))
    out = {"N_train": a.n, "seeds": seeds, "gammas": gammas, "oracle": round(float(orc), 4),
           "grid": [float(v) for v in LV],
           "regimes": {"uncap": {"mean": {"Kallus": mean}, "sd": {"Kallus": sd_},
                                 "obj": {"Kallus": objm}, "obj_sd": {"Kallus": objsd},
                                 "policy_seed0": {"Kallus": polS["0"]},
                                 "policy_by_seed": {"Kallus": polS}}}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("saved %s  oracle=%.3f  Kallus mean=%s  (%.1f min)" % (a.out, orc, mean, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
