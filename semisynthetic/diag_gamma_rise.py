#!/usr/bin/env python3
"""Why does IPW-X-X IMPROVE as gamma grows, when confounding is getting worse?

Measured: IPW-X-X normalised goes 0.749 (gamma=0) -> 0.139 (gamma=1.5) -> 0.255 (gamma=4), and
IPW-O-X and Direct-X-X track it. The references are identical across gamma, so this is a real
comparison, not a normalisation artefact.

HYPOTHESIS. In this DGP the hidden confounder IS the untreated outcome, U = Y0 in {-1,+1}, and
logit pi0 = lam'x + gamma*u with NO clipping. As gamma grows, assignment becomes almost perfectly
determined by u:  u=+1 -> never treated,  u=-1 -> always treated. So inside the TREATED arm, Y0
becomes CONSTANT (= -1), and the observed outcome there is

    Y1 = Y0 + tau(x) + eps  ->  -1 + tau(x) + eps,

which varies with x ONLY through tau. The Y0 heterogeneity that normally swamps the CATE signal is
selected away. Strong confounding wrecks the arm LEVELS (so the value estimate is badly biased) but
it makes the SHAPE of tau(x) easier to read -- and a policy only needs the shape.

This script measures the four quantities that hypothesis implies, per gamma:
  P(T=1)              treated share, should approach P(Y0=-1)
  sd(Y0 | T=1)        within-treated dispersion of the confounder, should fall to ~0
  corr(Y_obs, tau)    within the treated arm, should RISE toward 1
  signal-to-noise     sd(tau | T=1) / sd(Y_obs | T=1), should rise
plus, from the saved results, how far the LEARNED IPW-X-X policy curve sits from the oracle rule.
"""
import sys, json, glob, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bank_marketing")
    ap.add_argument("--dgp-seed", type=int, default=0)
    a = ap.parse_args()

    print("%s (dgp draw %d) -- population diagnostics per gamma\n" % (a.data, a.dgp_seed))
    print("%-6s %9s %8s %11s %12s %10s %10s"
          % ("gamma", "Gamma", "P(T=1)", "sd(Y0|T=1)", "corr(Y,tau)", "SNR|T=1", "P(Y0=-1)"))
    print("-" * 74)
    rows = {}
    for g in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0):
        f = HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, g, a.dgp_seed))
        if not f.exists():
            continue
        z = np.load(f)
        e, Y0, Y1, tau = z["e"].ravel(), z["Y0"].ravel(), z["Y1"].ravel(), z["tau"].ravel()
        rng = np.random.default_rng(10_000)          # same law the runner draws T from
        T = (rng.uniform(size=len(e)) < e).astype(int)
        Yo = np.where(T == 1, Y1, Y0)
        m = T == 1
        sdY0 = float(np.std(Y0[m]))
        cc = float(np.corrcoef(Yo[m], tau[m])[0, 1]) if np.std(Yo[m]) > 1e-12 else float("nan")
        snr = float(np.std(tau[m]) / (np.std(Yo[m]) + 1e-12))
        rows[g] = dict(pt=float(m.mean()), sdY0=sdY0, corr=cc, snr=snr)
        print("%-6.1f %9.1f %8.3f %11.3f %12.3f %10.3f %10.3f"
              % (g, np.exp(2 * g), m.mean(), sdY0, cc, snr, float((Y0 < 0).mean())))

    # how close is the LEARNED IPW-X-X policy to the oracle rule, as gamma grows?
    print("\nlearned IPW-X-X policy vs the oracle rule (from the saved per-cell curves)")
    print("%-6s %14s %14s %12s" % ("gamma", "agree w/ oracle", "frac treated", "oracle frac"))
    print("-" * 50)
    PG = np.linspace(-1.0, 1.0, 41)
    for g in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0):
        # the SLURM array names files from a shell string ("1.0") while "%g" gives "1" --
        # accept both spellings rather than depending on one formatter (same trap as the aggregator)
        fs = sorted(set(glob.glob(str(HERE / "results" / ("%s_g%g_d%d_s*.json" % (a.data, g, a.dgp_seed)))))
                    | set(glob.glob(str(HERE / "results" / ("%s_g%s_d%d_s*.json" % (a.data, g, a.dgp_seed))))))
        if not fs:
            continue
        zf = HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, g, a.dgp_seed))
        z = np.load(zf); xs, taus = z["x"].ravel(), z["tau"].ravel()
        # oracle decision along the same grid the curves are stored on
        orc = np.array([1.0 if np.mean(taus[np.abs(xs - t) < 0.05]) > 0 else 0.0 for t in PG])
        ag, ft = [], []
        for f in fs:
            d = json.loads(Path(f).read_text())
            cur = d.get("xx", {}).get("IPW-X-X", {})
            if not cur: continue
            best = max(cur.values(), key=lambda o: o["value"] if o["value"] == o["value"] else -9e9)
            c = np.asarray(best.get("curve", []), float)
            if c.size != len(PG): continue
            ag.append(float(np.mean((c > 0.5) == (orc > 0.5)))); ft.append(float(np.mean(c > 0.5)))
        if ag:
            print("%-6.1f %14.3f %14.3f %12.3f"
                  % (g, np.mean(ag), np.mean(ft), float(orc.mean())))


if __name__ == "__main__":
    main()
