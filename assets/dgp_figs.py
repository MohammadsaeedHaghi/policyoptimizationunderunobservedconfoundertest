#!/usr/bin/env python3
"""Shared DGP-explanation figures for the synthetic benchmark reports (gstar, KMZ).

Both DGP modules expose the same four functions -- p_s1(x), propensity(x, S), mu0(x, S),
mu1(x, S) -- so every figure here is computed from those alone and works unchanged on either
benchmark. Everything is EXACT (analytic marginalisation over the binary hidden state), not
Monte-Carlo, so the curves carry no simulation noise.

The four figures answer four separate questions about a confounded DGP:

  arm_means      What is the average outcome under each treatment?  E[Y(1)|x] and E[Y(0)|x],
                 marginalised over the hidden S. This is the quantity a policy is choosing
                 between, and its crossing point IS the oracle's decision boundary.
  observed_vs_true  What would you SEE in the data, per arm?  E[Y|x,T=t] against E[Y(t)|x].
                 The gap is the confounding bias, shown arm-by-arm rather than only in the
                 contrast -- which is where it is easiest to misread, because both arms can be
                 badly biased while the contrast looks mild (or the reverse).
  weight_box     NOT rendered by all_figs (removed from the reports on request). Kept because it
                 is the only view that shows the declared-nominal vs fitted-marginal gap behind
                 the operative-Gamma result; call it directly to bring it back.
  value_bars     NOT rendered by all_figs (removed from the reports on request). Call it
                 directly to bring the headroom bars back.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "semi experiments"))
from report_common import linechart, bars

C_T1, C_T0 = "#d62728", "#1f77b4"          # arm 1 / arm 0, matching the house estimator palette
C_SP, C_SM = "#2ca02c", "#9467bd"          # hidden state S = +1 / S = -1
C_NOM = "#334155"


def _marginals(d, xg):
    """Exact marginalisation over the binary hidden state S."""
    p = np.asarray(d.p_s1(xg), float) * np.ones_like(xg)     # P(S = +1 | x)
    one = np.ones_like(xg)
    e1, e0 = d.propensity(xg, one), d.propensity(xg, -one)   # P(T=1 | x, S)
    m1p, m1m = d.mu1(xg, one), d.mu1(xg, -one)
    m0p, m0m = d.mu0(xg, one), d.mu0(xg, -one)
    EY1 = p * m1p + (1 - p) * m1m                            # E[Y(1) | x]
    EY0 = p * m0p + (1 - p) * m0m                            # E[Y(0) | x]
    eT = p * e1 + (1 - p) * e0                               # P(T=1 | x)  (the NOMINAL propensity)
    # observed arm means: reweight each S by how likely that arm is for it
    obs1 = (p * e1 * m1p + (1 - p) * e0 * m1m) / np.maximum(eT, 1e-12)
    obs0 = (p * (1 - e1) * m0p + (1 - p) * (1 - e0) * m0m) / np.maximum(1 - eT, 1e-12)
    return dict(p=p, e1=e1, e0=e0, eT=eT, EY1=EY1, EY0=EY0, obs1=obs1, obs0=obs0,
                m1p=m1p, m1m=m1m, m0p=m0p, m0m=m0m)


def arm_means(d, xg, xlab="x"):
    M = _marginals(d, xg)
    cross = M["EY1"] - M["EY0"]
    sign_change = np.where(np.diff(np.sign(cross)) != 0)[0]
    hl = [("0", "#888", 0.0, "4 3")] if (M["EY1"].min() < 0 or M["EY0"].min() < 0) else None
    fig = linechart([("E[Y(1) | x]  -- treat", C_T1, list(xg), list(M["EY1"]), "", "n"),
                     ("E[Y(0) | x]  -- do not treat", C_T0, list(xg), list(M["EY0"]), "", "n")],
                    title="Average outcome under each treatment", xlab=xlab,
                    ylab="E[Y(a) | x]", hlines=hl)
    xs = ", ".join("%.2f" % xg[i] for i in sign_change) if len(sign_change) else "never"
    cap = ("Averaged over the hidden state S, so this is what the treatment is actually worth at "
           "each x. The two curves cross at x = %s &mdash; that crossing IS the oracle's decision "
           "boundary, and every method is trying to find it from confounded data." % xs)
    return fig, cap


def observed_vs_true(d, xg, xlab="x"):
    M = _marginals(d, xg)
    fig = linechart([("E[Y(1) | x]  (truth)", C_T1, list(xg), list(M["EY1"]), "", "n"),
                     ("E[Y | x, T=1]  (observed)", C_T1, list(xg), list(M["obs1"]), "5 4", "n"),
                     ("E[Y(0) | x]  (truth)", C_T0, list(xg), list(M["EY0"]), "", "n"),
                     ("E[Y | x, T=0]  (observed)", C_T0, list(xg), list(M["obs0"]), "5 4", "n")],
                    title="What you would see, against the truth", xlab=xlab, ylab="outcome")
    b1 = float(np.mean(np.abs(M["obs1"] - M["EY1"])))
    b0 = float(np.mean(np.abs(M["obs0"] - M["EY0"])))
    cap = ("Dashed = the arm means visible in infinite CONFOUNDED data; solid = the true "
           "potential-outcome means. Mean absolute bias %.2f in the treated arm and %.2f in the "
           "control arm. Both arms are biased in the SAME direction here, which is why the "
           "contrast can look far milder than either arm on its own." % (b1, b0))
    return fig, cap


def weight_box(d, xg, gstar, xlab="x"):
    """True 1/e(x,S) against the MSM box, plus the nominal the PIPELINE actually converges to.

    Which nominal the box is built around matters and the two can differ. A DGP may declare a
    nominal propensity e_nom(x) and place the true e(x,S) exactly on the resulting MSM boundary
    (Kallus-Mao-Zhou do); but an estimator fitted to (x, T) converges to the S-MARGINAL
    P(T=1|x), which need not equal e_nom. Where they differ, the box the solver actually gets is
    not the box the DGP was built to satisfy -- so both are drawn.
    """
    M = _marginals(d, xg)
    declared = getattr(d, "e_nom", None)
    e_ref = np.asarray(declared(xg), float) if declared is not None else M["eT"]
    w_nom = 1.0 / np.maximum(e_ref, 1e-12)
    lo = 1.0 + (w_nom - 1.0) / gstar
    hi = 1.0 + gstar * (w_nom - 1.0)
    w_p, w_m = 1.0 / np.maximum(M["e1"], 1e-12), 1.0 / np.maximum(M["e0"], 1e-12)
    w_marg = 1.0 / np.maximum(M["eT"], 1e-12)

    series = [("box upper  1 + G(w-1)", "#94a3b8", list(xg), list(hi), "5 4", "n"),
              ("box lower  1 + (w-1)/G", "#94a3b8", list(xg), list(lo), "5 4", "n"),
              ("nominal 1/e(x)", C_NOM, list(xg), list(w_nom), "", "n"),
              ("true 1/e(x, S=+1)", C_SP, list(xg), list(w_p), "", "n"),
              ("true 1/e(x, S=-1)", C_SM, list(xg), list(w_m), "", "n")]
    drift = float(np.max(np.abs(M["eT"] - e_ref))) if declared is not None else 0.0
    if drift > 1e-3:
        series.append(("1/P(T=1|x)  (what an estimator learns)", "#b45309", list(xg),
                       list(w_marg), "2 3"))

    def _inside(ref):
        w = 1.0 / np.maximum(ref, 1e-12)
        l, h = 1.0 + (w - 1.0) / gstar, 1.0 + gstar * (w - 1.0)
        return float(np.mean((w_p >= l - 1e-9) & (w_p <= h + 1e-9) &
                             (w_m >= l - 1e-9) & (w_m <= h + 1e-9)))

    fig = linechart(series, title="The MSM box at Gamma* = %g, and the weights it must cover"
                    % gstar, xlab=xlab, ylab="treated-arm weight 1/e")
    cap = ("Grey dashed = the interval the adversary may search at the matched Gamma* = %g. "
           "Green/purple = the weights the DGP actually uses for each hidden state. "
           "%.0f%% of the grid has BOTH inside the box." % (gstar, 100 * _inside(e_ref)))
    if drift > 1e-3:
        cap += (" <b>But that box is built around the DGP's DECLARED nominal.</b> An estimator "
                "fitted to (x, T) converges instead to the marginal P(T=1|x) (amber), which "
                "differs by up to %.3f here &mdash; and the box built around THAT contains both "
                "true weights on only %.0f%% of the grid. This is why the operative Gamma a "
                "fitted pipeline needs exceeds the declared Gamma* on this benchmark."
                % (drift, 100 * _inside(M["eT"])))
    return fig, cap


def value_bars(d, xg, cap_frac=None):
    """Oracle against the two constant policies, integrated over a uniform x."""
    M = _marginals(d, xg)
    orc = (M["EY1"] > M["EY0"]).astype(float)
    v_orc = float(np.mean(orc * M["EY1"] + (1 - orc) * M["EY0"]))
    v_all, v_nev = float(np.mean(M["EY1"])), float(np.mean(M["EY0"]))
    cats = ["oracle", "treat all", "treat none"]
    vals = [v_orc, v_all, v_nev]
    cols = ["#111111", C_T1, C_T0]
    fig = bars(cats, vals, cols, "How much is there to win? (analytic, infinite data)",
               "average outcome")
    best_const = max(v_all, v_nev)
    cap = ("Computed analytically over a uniform x, so these are the infinite-data values; the measured references elsewhere in this report differ by Monte-Carlo error. Headroom = oracle minus the better constant policy = <b>%+.3f</b> (oracle %.3f, "
           "treat-all %.3f, treat-none %.3f). The oracle treats %.0f%% of the covariate range. "
           "This is the gap every method in the results is competing over."
           % (v_orc - best_const, v_orc, v_all, v_nev, 100 * float(np.mean(orc))))
    return fig, cap


def coupling(d, xg, xlab="x"):
    """P(S=+1 | x): how much the OBSERVED covariate knows about the hidden state."""
    M = _marginals(d, xg)
    fig = linechart([("P(S=+1 | x)", C_NOM, list(xg), list(M["p"]), "", "n")],
                    title="Hidden-state coupling", xlab=xlab, ylab="P(S=+1 | x)",
                    hlines=[("0.5", "#888", 0.5, "4 3")], legend=False)
    rng = float(M["p"].max() - M["p"].min())
    cap = ("The single number that decides whether a transport / balance term can help at all. "
           "Range %.2f here. Flat at 0.5 means the covariate carries NO information about the "
           "confounder, and no X-balancing device can constrain it; a steep curve means the "
           "confounder is x-trackable." % rng)
    return fig, cap


def propensity_fig(d, xg, xlab="x"):
    M = _marginals(d, xg)
    series = [("P(T=1 | x, S=+1)", C_SP, list(xg), list(M["e1"]), "", "n"),
              ("P(T=1 | x, S=-1)", C_SM, list(xg), list(M["e0"]), "", "n"),
              ("P(T=1 | x)  marginal", C_NOM, list(xg), list(M["eT"]), "5 4", "n")]
    declared = getattr(d, "e_nom", None)
    if declared is not None:
        en = np.asarray(declared(xg), float) * np.ones_like(xg)
        if float(np.max(np.abs(en - M["eT"]))) > 1e-3:
            series.append(("declared nominal e(x)", "#b45309", list(xg), list(en), "2 3", "n"))
    fig = linechart(series, title="Propensity: how treatment was assigned", xlab=xlab,
                    ylab="P(T=1 | x, S)")
    lo = float(min(M["e1"].min(), M["e0"].min())); hi = float(max(M["e1"].max(), M["e0"].max()))
    cap = ("The vertical gap between the green and purple curves IS the confounding: two units at "
           "the same x get treated at different rates purely because of the hidden state. Range "
           "[%.3f, %.3f], no clipping, so overlap holds everywhere and the odds ratio between the "
           "two states is exactly Gamma* at every x." % (lo, hi))
    return fig, cap


def cate_fig(d, xg, xlab="x"):
    M = _marginals(d, xg)
    c = M["EY1"] - M["EY0"]
    orc = (c > 0).astype(float)
    fig = linechart([("CATE(x) = E[Y(1)-Y(0) | x]", C_T1, list(xg), list(c), "", "n")],
                    title="True treatment effect", xlab=xlab, ylab="CATE",
                    hlines=[("0", "#888", 0.0, "4 3")], legend=False)
    cap = ("Where this is positive the oracle treats: %.0f%% of the covariate range. This is the "
           "signal every method is trying to recover; everything else on this page is about what "
           "stands between the data and this curve." % (100 * float(np.mean(orc))))
    return fig, cap


def mu_by_state(d, xg, xlab="x"):
    """Outcome means split by BOTH arm and hidden state -- the raw ingredients of the DGP."""
    M = _marginals(d, xg)
    fig = linechart([("mu1(x, S=+1)", C_T1, list(xg), list(M["m1p"]), "", "n"),
                     ("mu0(x, S=+1)", C_T0, list(xg), list(M["m0p"]), "", "n"),
                     ("mu1(x, S=-1)", C_T1, list(xg), list(M["m1m"]), "5 4", "n"),
                     ("mu0(x, S=-1)", C_T0, list(xg), list(M["m0m"]), "5 4", "n")],
                    title="Outcome means by arm AND hidden state", xlab=xlab, ylab="mu_t(x, S)")
    d_state = float(np.mean(np.abs(M["m1p"] - M["m1m"])))
    d_arm = float(np.mean(np.abs(M["m1p"] - M["m0p"])))
    cap = ("Solid = S=+1, dashed = S=-1; red = treated, blue = control. The hidden state moves "
           "outcomes by %.1f on average while the treatment moves them by %.1f &mdash; the "
           "confounder is the larger effect, which is exactly what makes the benchmark hard."
           % (d_state, d_arm))
    return fig, cap


def contrast_inversion(d, xg, xlab="x"):
    """True CATE against the contrast an unconfounded analysis would report."""
    M = _marginals(d, xg)
    true_c = M["EY1"] - M["EY0"]
    obs_c = M["obs1"] - M["obs0"]
    fig = linechart([("true CATE", "#111111", list(xg), list(true_c), "", "n"),
                     ("observed contrast E[Y|x,T=1] - E[Y|x,T=0]", "#1f77b4", list(xg),
                      list(obs_c), "2 3", "n")],
                    title="What confounding does to the contrast", xlab=xlab, ylab="contrast",
                    hlines=[("0", "#888", 0.0, "4 3")])
    flip = float(np.mean((true_c > 0) != (obs_c > 0)))
    cap = ("Black = the truth, blue = what infinite CONFOUNDED data would show. The vertical gap "
           "is pure bias. It matters only where it crosses zero: the two disagree about whether "
           "to treat on <b>%.0f%%</b> of the covariate range, and that disagreement is the entire "
           "opportunity for a sensitivity-model method." % (100 * flip))
    return fig, cap


def all_figs(d, gstar, xg=None, xlab="x", figcap=None):
    """-> HTML for the seven-figure DGP set, two per row.

    Identical construction for every benchmark: the DGP module is the only input, so two reports
    built with this are directly comparable figure-for-figure.
    """
    if xg is None: xg = np.linspace(-1.0, 1.0, 401)
    if figcap is None:
        def figcap(f, c): return f'<div>{f}<p class="muted figcap">{c}</p></div>'
    blocks = [figcap(*coupling(d, xg, xlab)),
              figcap(*propensity_fig(d, xg, xlab)),
              figcap(*cate_fig(d, xg, xlab)),
              figcap(*mu_by_state(d, xg, xlab)),
              figcap(*arm_means(d, xg, xlab)),
              figcap(*observed_vs_true(d, xg, xlab)),
              figcap(*contrast_inversion(d, xg, xlab))]
    out = []
    for i in range(0, len(blocks), 2):
        out.append('<div class="figrow">' + "".join(blocks[i:i + 2]) + "</div>")
    return "".join(out)
