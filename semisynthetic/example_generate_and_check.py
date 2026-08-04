#!/usr/bin/env python3
"""Invariant checks on the semi-synthetic construction. Exits non-zero if anything fails."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from semisynthetic_dgp import load_uci_dataset, generate_semisynthetic

DATASET = sys.argv[1] if len(sys.argv) > 1 else "bank_marketing"
fails = []


def check(cond, msg):
    print("  [%s] %s" % ("ok " if cond else "FAIL", msg))
    if not cond: fails.append(msg)


X, y, info = load_uci_dataset(DATASET, enforce_screen=True)
print("dataset %s  n=%d d=%d  CV log-loss %.4f  (paper band [0.35, %.4f])\n"
      % (DATASET, info["n"], info["d"], info["cv_logloss"], np.log(2)))

for gamma in (0.0, 1.0, 1.5, 2.0):
    D, O, meta = generate_semisynthetic(X, y, gamma=gamma, seed=42, n_train=1500, n_test=500)
    d = meta.diagnostics
    print("gamma = %.1f   Gamma_MSM = e^{2g} = %.4f" % (gamma, meta.Gamma_MSM))

    # --- spec invariants ---
    check(np.array_equal(O["U"], O["Y0"]), "Y0 == U elementwise")
    check(set(np.unique(O["U"]).tolist()) <= {-1, 1}, "U and Y0 are in {-1,+1}")
    c0 = D["T"] == 0
    check(np.allclose(D["Y"][c0], O["Y0"][c0]), "Y == Y0 where T = 0")
    check(np.allclose(D["Y"][~c0], O["Y1"][~c0]), "Y == Y1 where T = 1")
    check(not any(k in D for k in ("U", "Y0", "Y1", "tau", "pi0", "e_star")),
          "D_obs leaks no oracle field")

    # --- confounding direction: adverse Y0 = -1 must be treated MORE (pi0 lower) ---
    p_good = float(np.mean(D["T"][O["Y0"] == 1] == 0))     # P(T=0 | Y0=+1)
    p_bad = float(np.mean(D["T"][O["Y0"] == -1] == 0))     # P(T=0 | Y0=-1)
    if gamma > 0:
        check(p_good > p_bad,
              "P(T=0|Y0=+1)=%.4f > P(T=0|Y0=-1)=%.4f  (confounding points the right way)"
              % (p_good, p_bad))
    else:
        check(abs(p_good - p_bad) < 0.05,
              "gamma=0: P(T=0|.) nearly equal (%.4f vs %.4f)" % (p_good, p_bad))

    # --- the exact odds-ratio identity that makes Gamma = e^{2 gamma} exact ---
    check(d["odds_ratio_max_abs_err"] < 1e-9,
          "odds ratio between U states == e^{2g} exactly (max abs err %.2e)"
          % d["odds_ratio_max_abs_err"])
    check(0.01 < d["e_min"] and d["e_max"] < 0.99,
          "positivity holds without clipping: e in [%.4f, %.4f]" % (d["e_min"], d["e_max"]))

    # --- the check the paper's screen does NOT do: is there anything to learn? ---
    check(d["headroom"] > 0.05,
          "headroom (oracle - best constant) = %+.4f > 0.05" % d["headroom"])
    check(0.2 < d["frac_tau_pos"] < 0.8,
          "CATE changes sign: frac(tau>0) = %.3f" % d["frac_tau_pos"])
    print("     propensity imbalance P(T=1)=%.3f  |  oracle %.4f  never %.4f  all %.4f\n"
          % (d["P_T1"], d["oracle"], d["never"], d["all"]))

# --- ALSO validate the PREPARED population, which is what the campaign actually consumes.
# The per-gamma checks above re-draw their own (a,b,c) and so have their own headroom; the
# prepared file is a different draw and is the one that matters.
_pre = HERE / "prepared" / "_index.json"
if _pre.exists():
    import json as _json
    recs = [r for r in _json.loads(_pre.read_text()) if r["dataset"] == DATASET]
    if recs:
        print("prepared population (what the runner consumes):")
        for r in recs:
            check(r["headroom"] > 0.05,
                  "g=%.1f prepared headroom = %+.4f > 0.05" % (r["gamma"], r["headroom"]))
            check(r["odds_ratio_max_abs_err"] < 1e-9,
                  "g=%.1f prepared odds-ratio err %.1e" % (r["gamma"], r["odds_ratio_max_abs_err"]))
        print()

# --- the uncentred variant must FAIL the headroom check: that is the point of centring ---
_D, _O, m0 = generate_semisynthetic(X, y, gamma=1.0, seed=42, n_train=1500, n_test=500,
                                    centre_tau=False)
print("negative control (uncentred tau, the source spec's form):")
check(m0.diagnostics["frac_tau_pos"] > 0.999,
      "uncentred tau is strictly positive: frac(tau>0) = %.4f" % m0.diagnostics["frac_tau_pos"])
check(m0.diagnostics["headroom"] < 0.01,
      "and so it has no headroom: %+.5f (oracle == all-treat)" % m0.diagnostics["headroom"])

print("\n%s (%d failures)" % ("ALL CHECKS PASS" if not fails else "FAILURES", len(fails)))
sys.exit(1 if fails else 0)
