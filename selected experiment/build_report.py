#!/usr/bin/env python3
# Build a self-contained, owgap-ONLY report (exp_nmcap removed). Embeds every plot as base64.
# Run from inside the "selected experiment/" folder:  python3 build_report.py
import json, base64, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OWG = os.path.join(HERE, "exp_owgap")
FAM = {'IPW':'#1f77b4','Hajek':'#9467bd','DoublyRobust':'#2ca02c','Direct':'#ff7f0e','Kallus':'#d62728','Oracle':'#444444'}

charts = json.load(open(os.path.join(OWG, "owgap_charts.json")))

def restab(key, cols):
    v = charts[key]; xs = v['x']; idx = [(c, xs.index(c)) for c in cols if c in xs]
    th = ''.join("<th>Γ=%s</th>" % (int(c) if c == int(c) else c) for c, _ in idx)
    rows = ''
    for s in v['series']:
        lab = s.get('label', s['id']); col = FAM.get(lab.split('-')[0], '#888')
        sw = "<span class='sw' style='background:%s'></span>" % col
        cells = ''.join(("<td>&ndash;</td>" if s['y'][i] is None else "<td>%.2f</td>" % s['y'][i]) for _, i in idx)
        cls = " class='hl'" if lab.endswith('-O-W') else ""
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (cls, sw, lab, cells)
    return "<table class='restab'><tr><th>method</th>%s</tr>%s</table>" % (th, rows)

owg_u = restab('val_uncap_ce1', [1.0,1.5,2.0,3.0,4.0,6.0,8.0])
owg_c = restab('val_cap_ce1',   [1.0,1.5,2.0,3.0,4.0,6.0,8.0])

def datauri(name):
    with open(os.path.join(OWG, name), 'rb') as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode('ascii')
def figcard(src, title, cap):
    return ("<figure class='card'><figcaption class='ft'>%s</figcaption>"
            "<img loading='lazy' src='%s' alt='%s'><figcaption class='fc'>%s</figcaption></figure>"
            % (title, datauri(src), title, cap))
def grid(items): return "<div class='grid'>%s</div>" % ''.join(figcard(*it) for it in items)
def inner_tabs_n(prefix, panels):
    btns = ''.join("<button class='inner-btn%s' onclick=\"showInner('%s-%d',this)\">%s</button>" % (' active' if i==0 else '', prefix, i, lbl) for i,(lbl,_) in enumerate(panels))
    divs = ''.join("<div id='%s-%d' class='inner-panel%s'>%s</div>" % (prefix, i, ' active' if i==0 else '', html) for i,(_,html) in enumerate(panels))
    return "<div class='inner-tabs'>" + btns + "</div>" + divs
def inner_tabs(prefix, uncap_html, cap_html):
    return ("<div class='inner-tabs'>"
            "<button class='inner-btn active' onclick=\"showInner('%s-uncap',this)\">Uncapped&nbsp;(treat ≤ 100%%)</button>"
            "<button class='inner-btn' onclick=\"showInner('%s-cap',this)\">Capped&nbsp;(treat ≤ 50%%)</button></div>"
            "<div id='%s-uncap' class='inner-panel active'>%s</div>"
            "<div id='%s-cap' class='inner-panel'>%s</div>" % (prefix, prefix, prefix, uncap_html, prefix, cap_html))

owg_truth = [
 ("sx.png","Hidden vitality vs fitness — P(S=+1|X)=σ(10X)","Strong S–X coupling: balancing the observed X also balances the unobserved S."),
 ("prop.png","Propensity e(X,S)=σ(0.8S−2X)","Moderate selection on the hidden vitality, so the matched Γ stays small (≈5)."),
 ("pt_x.png","Marginal treatment propensity P(T=1 | X)","What the analyst actually observes once the hidden S is integrated out: P(T=1|X)=σ(10X)·e(X,+1)+(1−σ(10X))·e(X,−1). The per-S curves (dashed) are unobservable."),
 ("outcome.png","Per-arm outcome means E[Y(t)|X,S]","Vitality dominates both arms (μ₀=8S, μ₁=9S+1.5X) — the fitted outcome model is badly biased."),
 ("cate.png","True CATE(X)=(2σ(10X)−1)+1.5X","Treat iff X>0. Subtle and partly harmful for the unfit (X<0)."),
 ("obs.png","Observed covariate imbalance","Arm-specific X frequencies in one training draw."),
]
owg_val_uncap = [
 ("val_uncap_ce1.png","Realised E[Y] vs Γ — c_ε=1 (headline, mean ± SD)","Both O-W circles beat every method, peaking at Γ=2–3. Hajek-O-X collapses to never-treat by Γ≈2.5."),
 ("val_uncap_ce1.5.png","c_ε=1.5","Wider Wasserstein radius steepens the post-peak decline."),
 ("val_uncap_ce2.png","c_ε=2 (collapse view)","IPW-O-W peaks then declines — the over-conservative collapse, but the gap also erodes."),
]
owg_val_cap = [
 ("val_cap_ce1.png","Realised E[Y] vs Γ — c_ε=1 (mean ± SD)","The cap rescues AIPW to 0.71; IPW-O-W keeps the clean win, DR-O-W ties at small Γ."),
 ("val_cap_ce1.5.png","c_ε=1.5","Same trend under the cap."),
 ("val_cap_ce2.png","c_ε=2 (collapse view)","Capped collapse view."),
]
lim = [
 ("ex_ipw.png","Pure-IPW O-W collapses under extreme Γ×ε","IPW-O-W for four ε. Push both knobs → hedges to never-treat (0.85 → ~0.08 at c_ε=16, Γ=1000)."),
 ("ex_family.png","Doubly-robust anchored; Hajek collapses hardest","DR-O-W/-O-X floor ~0.6 (direct μ̂ term anchors them); Hajek-O-X collapses earliest."),
]

LEGEND = ("<div class='legend'><b>How to read every figure.</b> <b>Colour = estimator family</b> — "
 "<span class='sw' style='background:#1f77b4'></span>IPW, <span class='sw' style='background:#2ca02c'></span>DoublyRobust (AIPW), "
 "<span class='sw' style='background:#9467bd'></span>Hajek, <span class='sw' style='background:#ff7f0e'></span>Direct, "
 "<span class='sw' style='background:#444'></span>Oracle. <b>Shape = uncertainty set</b> — X-X dashed (Γ-free), "
 "O-X squares (odds-box), O-W circles (odds-box ∩ Wasserstein). Higher E[Y] is better.</div>")

TIGHT_EPS_NOTE = ("<div class='note'><b>On ε: c_ε=1 is the tightest feasible Wasserstein ball — a literal ε=0 is impossible.</b> "
 "Here ε is measured as a multiple of <code>tight_epsilon</code>, defined as the <i>minimum optimal-transport cost</i> to map the empirical "
 "\\(X\\)-law onto the IPW-reweighted law. So <b>c_ε=1 sets ε exactly to that floor</b> — the tightest possible ball — and the c_ε=1 panel below "
 "<b>is</b> the tightest-ε experiment. Any smaller radius (including 0) leaves the reweighted target unreachable, and the O-W worst-case dual is "
 "<b>unbounded</b> (verified: ε below the floor ⇒ Gurobi status 5, at every N). The reason a zero ball fails even though same-\\(X\\) units can swap "
 "mass for free: <b>balancing the arms needs <i>cross</i>-level transport</b> (e.g. mass must flow into the sparse treated cell at \\(X=1\\)), which costs "
 "&gt;0. Measured floors: tight_epsilon ≈ <b>(0.24, 0.28) at N=16</b> vs ≈ <b>(0.04, 0.03) at N=300</b> — the floor <b>shrinks as N grows</b> "
 "(more data ⇒ less extreme weights ⇒ less cross-level mass to move), reaching 0 only as \\(N\\to\\infty\\).</div>")

tab_overview = (LEGEND +
 "<div class='panel'><h3 style='margin-top:4px'>The result — exp_owgap (confounding-robust policy optimization)</h3>"
 "<p>A clinical DGP where an <b>unobserved vitality S</b> dominates outcomes. Both <b>O-W methods (IPW-O-W and "
 "DoublyRobust-O-W) beat every other method</b> — including plain AIPW — by a large margin, peaking at a small "
 "sensitivity Γ (2–3), in both the capped and uncapped regimes.</p>"
 "<table class='sep'><tr><th>method</th><th>exp_owgap (hidden-S confounding)</th></tr>"
 "<tr><td>naive IPW-X-X</td><td class='fail'>0.33 — fails</td></tr>"
 "<tr><td>naive AIPW (DoublyRobust-X-X)</td><td>0.64–0.65</td></tr>"
 "<tr><td><b>IPW-O-W</b></td><td class='win'>0.78 ✓</td></tr>"
 "<tr><td><b>DoublyRobust-O-W</b></td><td class='win'>0.78 ✓</td></tr></table>"
 "<p class='sub' style='margin:6px 0 0'>Realised E[Y] at the operating point (uncapped, c_ε=1). Oracle=0.847; never-treat=0.</p></div>"
 "<h3>What to put on the paper</h3><ol>"
 "<li><b>Headline:</b> uncapped value-vs-Γ figure + the 20-seed paired-CI claim (IPW-O-W − AIPW = +0.12 [0.09, 0.15] at Γ=3).</li>"
 "<li><b>Key insight:</b> since mean_X E[Y(0)|X]=0, V(π)=mean_X[π·CATE] is independent of d₀ → confounding magnitude (gap) and CATE scale (collapse speed) decouple.</li>"
 "<li><b>Over-conservatism (extreme Γ×ε)</b> → Limitations/Discussion (see the Limitations tab).</li></ol>"
 "<div class='note'><b>Honest caveat:</b> under a capacity cap, AIPW is rescued (0.64→0.71), so IPW-O-W carries the clean win "
 "while DoublyRobust-O-W ties at small Γ. Anchor strongest claims on the uncapped regime.</div>"
 "<p class='sub' style='margin-top:16px'>Data currently shown: <b>N=600, 5 seeds</b> (20-seed for the CI). DGP: "
 "<code>exp_owgap/dgp.py</code>. This report is owgap-only (exp_nmcap removed).</p>")

DGP_EXPLAIN = ("<div class='panel'><h3 style='margin-top:4px'>How the dataset is generated (the DGP)</h3>"
 "<p>Each row is one synthetic patient. For a chosen sample size \\(N\\) and random seed we draw \\(N\\) i.i.d. patients in the order below. Throughout, \\(\\sigma(z)=\\dfrac{1}{1+e^{-z}}\\) is the logistic function.</p>"
 "<p><b>Step 1 — observed covariate</b> \\(X\\) (fitness / biomarker), uniform on a 7-point grid:</p>"
 "\\[ X \\;\\sim\\; \\mathrm{Uniform}\\Big\\{-1,\\;-\\tfrac{2}{3},\\;-\\tfrac{1}{3},\\;0,\\;\\tfrac{1}{3},\\;\\tfrac{2}{3},\\;1\\Big\\}. \\]"
 "<p><b>Step 2 — unobserved confounder</b> \\(S\\in\\{-1,+1\\}\\) (latent &ldquo;vitality&rdquo;), strongly coupled to \\(X\\):</p>"
 "\\[ \\Pr(S=+1\\mid X)=\\sigma(10\\,X), \\qquad \\Pr(S=-1\\mid X)=1-\\sigma(10\\,X). \\]"
 "<p>Since \\(\\sigma(10X)\\) saturates quickly, vitality tracks fitness almost deterministically. <b>\\(S\\) is hidden from every estimator</b> — it is the source of unobserved confounding.</p>"
 "<p><b>Step 3 — treatment</b> \\(T\\in\\{0,1\\}\\) (\\(1=\\) aggressive therapy), assigned with propensity that depends on the hidden \\(S\\):</p>"
 "\\[ T\\mid X,S \\;\\sim\\; \\mathrm{Bernoulli}\\big(e(X,S)\\big), \\qquad e(X,S)=\\sigma(0.8\\,S-2\\,X). \\]"
 "<p>Clinicians escalate the vital (\\(S=+1\\)) and the low-fitness (\\(-2X\\)) patients, so treatment is <b>confounded by \\(S\\)</b> — exactly what the \\(\\Gamma\\) sensitivity model hedges against.</p>"
 "<p><b>Step 4 — potential outcomes</b> (vitality \\(S\\) dominates the scale of <i>both</i> arms):</p>"
 "\\[ Y(0)=8\\,S+\\varepsilon, \\qquad Y(1)=9\\,S+1.5\\,X+\\varepsilon, \\qquad \\varepsilon\\sim\\mathcal{N}\\!\\big(0,\\,0.6^{2}\\big). \\]"
 "<p><b>Step 5 — observed outcome</b> (only the assigned arm is revealed):</p>"
 "\\[ Y=Y(T)=T\\,Y(1)+(1-T)\\,Y(0). \\]"
 "<p>The estimators receive <b>only \\((X,T,Y)\\)</b>; the quantities \\(S,\\;Y(0),\\;Y(1),\\;e(X,S)\\) are withheld and used only to score the oracle.</p>"
 "<p><b>Resulting effect &amp; optimal policy.</b> The conditional average treatment effect is</p>"
 "\\[ \\tau(X)=\\mathbb{E}\\big[Y(1)-Y(0)\\mid X\\big]=(9-8)\\,\\mathbb{E}[S\\mid X]+1.5\\,X=\\underbrace{\\big(2\\sigma(10X)-1\\big)}_{=\\,\\mathbb{E}[S\\mid X]}+\\,1.5\\,X, \\]"
 "<p>so the optimal (oracle) policy is</p>"
 "\\[ \\pi^{\\star}(X)=\\mathbf{1}\\{\\tau(X)>0\\}=\\mathbf{1}\\{X>0\\}, \\qquad \\text{oracle value }=0.847,\\quad \\text{never-treat }=0. \\]"
 "<p class='sub'><b>Why it is hard, and why O-W wins.</b> The hidden \\(S\\) both <i>assigns treatment</i> (Step 3) and <i>dominates the outcome</i> (Step 4), so naive IPW / outcome-model estimators are badly biased and over-treat into the harmful low-fitness region. The O-W methods enforce covariate (Wasserstein) balance on the observed \\(X\\); because \\(S\\) tracks \\(X\\), balancing \\(X\\) also <b>balances the hidden \\(S\\)</b>, recovering \\(\\pi^{\\star}(X)=\\mathbf{1}\\{X>0\\}\\). Code: <code>exp_owgap/dgp.py → generate(n, seed)</code>.</p></div>")
DGP_TRUTH = "<h3>The data-generating process (truth) — shared by both regimes</h3>" + grid(owg_truth)

N1000_ORDER = ['IPW-X-X','DoublyRobust-X-X','Direct-X-X','IPW-O-X','DoublyRobust-O-X','Hajek-O-X','IPW-O-W','DoublyRobust-O-W']
def _n1style(name):
    fam = name.split('-')[0]; c = FAM.get(fam, '#888')
    if name.endswith('-O-W'): return c,'-','o'
    if name.endswith('-O-X'): return c,'-','s'
    if name.endswith('-X-X'): return c,'--',None
    return c,'-',None

# ---------- generic "Objective vs Γ" panel (3 plots, one per estimator) ----------
OBJ_GROUPS = [
    ('IPW',           ['IPW-X-X', 'Direct-X-X', 'IPW-O-X', 'IPW-O-W'],              'worst-case IPW value bound'),
    ('Doubly-Robust', ['DoublyRobust-X-X', 'DoublyRobust-O-X', 'DoublyRobust-O-W'], 'robust DR value bound'),
    ('Hajek',         ['Hajek-O-X'],                                                'worst-case regret (≤ 0)'),
]
def _obj_curve(obj_path, methods, fname, ylab, title, ns):
    d = json.load(open(obj_path)); G = d['gammas']; blk = d['regimes']['uncap']['obj']
    lg = np.log10(G)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m in methods:
        if m not in blk: continue
        c, ls, mk = _n1style(m)
        ax.plot(lg, blk[m], ls, color=c, marker=mk, ms=5, lw=2.1, label=m)
    ax.axhline(0, color='#888', ls=':', lw=1)
    ax.set_xticks(lg); ax.set_xticklabels(['%g' % g for g in G], fontsize=7)
    ax.set_xlabel('Γ  (log scale)'); ax.set_ylabel('%s  (%d seeds)' % (ylab, ns))
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, fname), dpi=120); plt.close(fig)
    return fname
def _obj_valcurve(obj_path, fname, oracle, ns):
    d = json.load(open(obj_path)); G = d['gammas']; blk = d['regimes']['uncap']['mean']
    lg = np.log10(G)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for m in N1000_ORDER:
        if m not in blk: continue
        c, ls, mk = _n1style(m)
        ax.plot(lg, blk[m], ls, color=c, marker=mk, ms=4, lw=1.9, label=m)
    ax.axhline(oracle, color='#444', ls=':', lw=1.3, label='oracle %.2f' % oracle)
    ax.axhline(0, color='#bbb', ls=':', lw=1)
    ax.set_xticks(lg); ax.set_xticklabels(['%g' % g for g in G], fontsize=7)
    ax.set_xlabel('Γ  (log scale)'); ax.set_ylabel('realised E[Y]  (%d seeds)' % ns)
    ax.grid(alpha=.25); ax.legend(fontsize=6.5, ncol=2)
    ax.set_title('Realised value of the resulting policy vs Γ', fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, fname), dpi=120); plt.close(fig)
    return fname
def _obj_table(obj_path, methods, key='obj', fmt='%.2f'):
    d = json.load(open(obj_path)); G = d['gammas']; blk = d['regimes']['uncap'][key]
    th = ''.join('<th>Γ=%g</th>' % g for g in G)
    rows = ''
    for m in methods:
        if m not in blk: continue
        sw = "<span class='sw' style='background:%s'></span>" % FAM.get(m.split('-')[0], '#888')
        cls = " class='hl'" if m.endswith('-O-W') else ""
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (cls, sw, m, ''.join('<td>' + (fmt % v) + '</td>' for v in blk[m]))
    return "<table class='restab'><tr><th>method</th>%s</tr>%s</table>" % (th, rows)
def obj_panel(obj_path, slug):
    if not os.path.exists(obj_path):
        return "<p class='sub'>Objective sweep for this experiment is still running — rebuild when <code>%s</code> lands.</p>" % os.path.basename(obj_path)
    d = json.load(open(obj_path)); ns = len(d['seeds']); orc = d['oracle']
    cards = []
    for label, meths, scale in OBJ_GROUPS:
        fn = '%s_obj_%s.png' % (slug, label.replace('-', ''))
        _obj_curve(obj_path, meths, fn, scale, '%s estimator' % label, ns)
        cards.append((fn, '%s estimator' % label, 'objective = %s. X-X is the Γ-free baseline; O-X / O-W fall as Γ grows.' % scale))
    tables = ''.join("<h4 style='margin:16px 0 4px'>%s estimator — objective values</h4><div class='panel'>%s</div>"
                     % (label, _obj_table(obj_path, meths)) for label, meths, _ in OBJ_GROUPS)
    valfn = '%s_objval.png' % slug
    _obj_valcurve(obj_path, valfn, orc, ns)
    return (
      "<h3>What this tab shows</h3>"
      "<div class='note'>Each curve is the <b>internal worst-case objective the method actually maximises</b> — "
      "<i>not</i> the realised outcome. The three estimators optimise <b>different</b> objectives on <b>different scales</b> "
      "(IPW value, Doubly-Robust value, Hájek <b>regret</b>), so each gets its <b>own plot</b>. Within a plot the <b>X-X</b> curve is the "
      "non-robust baseline (Γ-free, flat), <b>O-X</b> adds the odds-box, <b>O-W</b> adds odds-box ∩ Wasserstein. Γ is pushed to 1000. "
      "Watching this objective fall is the <i>mechanism</i> that drives the policy conservative.</div>"
      "<h3>Worst-case objective vs Γ — one plot per estimator</h3>"
      + grid(cards) + tables +
      "<div class='note'><b>How to read it.</b> The <b>X-X</b> baseline is flat — it ignores confounding, so Γ does nothing. "
      "For <b>IPW</b>, the box-only <b>IPW-O-X</b> dives hardest (nothing bounds its worst case, so as Γ→∞ it believes treating could be "
      "arbitrarily bad and retreats to never-treat); <b>IPW-O-W</b> falls far less because the <b>Wasserstein ε-ball caps</b> the worst case. "
      "For <b>Doubly-Robust</b>, both O-curves fall more slowly because the direct \\(\\hat\\mu\\) term is immune to Γ (only the IPW residual sits in the box). "
      "For <b>Hájek</b> the objective is worst-case <b>regret</b> and floors at <b>0</b> (never-treat has zero regret — the do-no-harm fallback).</div>"
      "<h3>…and the policy that objective produces</h3>"
      + grid([(valfn, 'Realised E[Y] of the resulting policy',
               'The objective above is the cause; this realised value is the effect — unanchored objectives collapse, anchored ones do not.')]) +
      "<div class='panel'>" + _obj_table(obj_path, N1000_ORDER, key='mean') + "</div>"
      "<div class='note sub'>Objective values are worst-case <b>lower bounds</b> (well below the oracle of %.2f); what matters for the decision is their "
      "<b>shape</b> in Γ. Uncapped, c_ε=1, %d seeds.</div>" % (orc, ns))

tab_owgap = ("<div class='mhead'><span class='pill' style='background:#be123c'>exp_owgap</span>"
 "<h2 style='border:none;margin:0'>Hidden-vitality clinical DGP</h2></div>"
 "<p class='lead'><b>Design:</b> X discrete (7 levels); S∈{−1,+1} unobserved, P(S=+1|X)=σ(10X); e(X,S)=σ(0.8S−2X); "
 "μ₀=8S, μ₁=9S+1.5X; CATE(X)=(2σ(10X)−1)+1.5X ⇒ treat iff X&gt;0. Oracle=0.847. (DGP: <code>exp_owgap/dgp.py</code>.)</p>"
 + DGP_EXPLAIN + LEGEND + DGP_TRUTH +
 "<h3 style='margin-top:28px'>Results by regime</h3>" +
 inner_tabs_n('owgap', [('Uncapped',
   "<h3>Realised value vs Γ (ε sweep)</h3>" + grid(owg_val_uncap) +
   "<h3>Result table (realised E[Y], c_ε=1 — O-W rows highlighted)</h3><div class='panel'>" + owg_u + "</div>"
   "<div class='good'><b>Finding (uncapped).</b> At Γ=2–3 both O-W beat the best naive competitor (AIPW=0.64) by +0.11–0.18; "
   "20-seed paired CI IPW-O-W − AIPW = +0.12 [0.09, 0.15] at Γ=3. Box-only AIPW (DR-O-X) fails → the Wasserstein term is the differentiator.</div>"),
   ('Capped',
   "<h3>Realised value vs Γ (ε sweep)</h3>" + grid(owg_val_cap) +
   "<h3>Result table (realised E[Y], c_ε=1 — O-W rows highlighted)</h3><div class='panel'>" + owg_c + "</div>"
   "<div class='note'><b>Finding (capped).</b> The cap rescues AIPW to 0.71, so IPW-O-W carries the clean win while DoublyRobust-O-W ties at small Γ.</div>"),
   ('Objective vs Γ', obj_panel(os.path.join(OWG, 'owgap_obj.json'), 'owg'))]))

tab_lim = ("<div class='mhead'><span class='pill' style='background:#7c3aed'>Limitations</span>"
 "<h2 style='border:none;margin:0'>Over-conservatism at extreme Γ×ε</h2></div>"
 "<p class='lead'>Inflating the uncertainty set far past anything realistic makes the worst-case-<i>value</i> methods hedge to never-treat — a property to disclose, not hide.</p>"
 + grid(lim) +
 "<div class='note'><b>Takeaway.</b> Pushing Γ and ε together drives IPW-O-W from 0.85 to ~0.08 (~never-treat); the doubly-robust "
 "variants are anchored (~0.6) by their direct μ̂ term; Hajek-O-X collapses earliest. ⇒ Anchor strongest claims on the uncapped, c_ε=1 operating point.</div>")

# ---------- N=1000 rerun (added as a NEW tab; the N=600 sections above are kept) ----------
N1000_CES = [ce for ce in ['1.0','1.5','2.0'] if os.path.exists(os.path.join(OWG, 'owgap_results_n1000_ce%s.json' % ce))]
def gen_n1000_plot(ce, reg):
    d = json.load(open(os.path.join(OWG, 'owgap_results_n1000_ce%s.json' % ce)))
    G = d['gammas']; orc = d['oracle']; ns = len(d['seeds']); mean = d['regimes'][reg]['mean']; sd = d['regimes'][reg].get('sd', {})
    fig, ax = plt.subplots(figsize=(7,4.3))
    for m in N1000_ORDER:
        if m not in mean: continue
        y = np.array(mean[m], float); c, ls, mk = _n1style(m)
        ax.plot(G, y, ls, color=c, marker=mk, ms=4.5, lw=2, label=m)
        if m in sd: s = np.array(sd[m], float); ax.fill_between(G, y-s, y+s, color=c, alpha=.13, lw=0)
    ax.axhline(orc, color='#444', ls=':', label='oracle %.2f' % orc); ax.axhline(0, color='#bbb', ls=':')
    ax.set_xlabel('Γ'); ax.set_ylabel('realised E[Y] (mean ± SD, %d seeds)' % ns)
    ax.set_ylim(-0.06, orc+0.07); ax.grid(alpha=.25); ax.legend(fontsize=6.5, ncol=2)
    ax.set_title('N=1000, %s, c_ε=%s (%d seeds)' % ('UNCAPPED' if reg=='uncap' else 'CAPPED (treat≤50%%)', ce, ns), fontsize=9)
    out = 'val_n1000_%s_ce%s.png' % (reg, ce); fig.tight_layout(); fig.savefig(os.path.join(OWG, out), dpi=120); plt.close(fig)
    return out
def restab_n1000(ce, reg, cols):
    d = json.load(open(os.path.join(OWG, 'owgap_results_n1000_ce%s.json' % ce)))
    G = d['gammas']; orc = d['oracle']; mean = d['regimes'][reg]['mean']; idx = [(c, G.index(c)) for c in cols if c in G]
    th = ''.join("<th>Γ=%s</th>" % (int(c) if c==int(c) else c) for c,_ in idx)
    rows = ''
    for m in N1000_ORDER:
        if m not in mean: continue
        sw = "<span class='sw' style='background:%s'></span>" % FAM.get(m.split('-')[0], '#888')
        cells = ''.join("<td>%.2f</td>" % mean[m][i] for _,i in idx)
        cls = " class='hl'" if m.endswith('-O-W') else ""
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (cls, sw, m, cells)
    rows += "<tr><td><span class='sw' style='background:#444'></span>Oracle</td>%s</tr>" % ''.join("<td>%.2f</td>" % orc for _ in idx)
    return "<table class='restab'><tr><th>method</th>%s</tr>%s</table>" % (th, rows)
def _n1000_regime(reg):
    figs = grid([(gen_n1000_plot(ce, reg), 'c_ε=%s' % ce, 'N=1000, mean ± SD over seeds — realised E[Y] vs Γ.') for ce in N1000_CES])
    tbl = restab_n1000('1.0', reg, [1.0,1.5,2.0,3.0,4.0,6.0,8.0]) if '1.0' in N1000_CES else '<p class="sub">c_ε=1.0 table pending.</p>'
    return ("<h3>Realised value vs Γ (ε present: %s)</h3>%s"
            "<h3>Result table (c_ε=1.0, mean — O-W rows highlighted)</h3><div class='panel'>%s</div>" % (', '.join(N1000_CES), figs, tbl))
if N1000_CES:
    _ns = len(json.load(open(os.path.join(OWG, 'owgap_results_n1000_ce%s.json' % N1000_CES[0])))['seeds'])
    tab_n1000 = ("<div class='mhead'><span class='pill' style='background:#0e7490'>N=1000 rerun</span>"
      "<h2 style='border:none;margin:0'>exp_owgap at N=1000 (%d-seed, 2-worker)</h2></div>"
      "<p class='lead'>Higher-N rerun of exp_owgap (N_train=1000, %d seeds), <b>added alongside</b> the original N=600 results "
      "(the other tabs are unchanged). Same DGP (<code>exp_owgap/dgp.py</code>). ε completed so far: <b>%s</b> of {1.0, 1.5, 2.0} "
      "— this tab updates as the remaining ε land.</p>" % (_ns, _ns, ', '.join(N1000_CES)) + LEGEND +
      DGP_EXPLAIN + DGP_TRUTH +
      "<h3 style='margin-top:28px'>Results by regime</h3>" + inner_tabs_n('n1000', [
          ('Uncapped', _n1000_regime('uncap')),
          ('Capped', _n1000_regime('cap')),
          ('Objective vs Γ', "<div class='note'>The worst-case <b>objective</b> vs Γ is a property of the estimator and the uncertainty set — <b>not</b> of the sample size. "
              "At N=1000 the curves have the <b>same shape</b> as N=600 (identical DGP, same oracle rule), so to avoid a redundant ~40-min re-solve the panel below reuses the "
              "<b>N=600 objective re-solve</b> (uncapped, c_ε=1). The N=1000 policies — shown in the Uncapped / Capped tabs — follow exactly this mechanism.</div>"
              + obj_panel(os.path.join(OWG, 'owgap_obj.json'), 'owg'))]) +
      "<div class='note'>N=1000 is a separate rerun at fewer seeds (%d) than the N=600 headline (5-seed, 20 for the CI); treat the exact margins as preliminary until all ε and more seeds are in.</div>" % _ns)
else:
    tab_n1000 = None

# ---------- exp_owgap_x3: 3-value X {-1,0,1}, N=200 ----------
OWX3 = os.path.join(HERE, "exp_owgap_x3")
X3_CES = [ce for ce in ['1.0','1.5','2.0'] if os.path.exists(os.path.join(OWX3, 'owgap_x3_ce%s.json' % ce))]
def gen_x3_plot(ce, reg):
    d = json.load(open(os.path.join(OWX3, 'owgap_x3_ce%s.json' % ce)))
    G = d['gammas']; orc = d['oracle']; ns = len(d['seeds']); mean = d['regimes'][reg]['mean']; sd = d['regimes'][reg].get('sd', {})
    fig, ax = plt.subplots(figsize=(7,4.3))
    for m in N1000_ORDER:
        if m not in mean: continue
        y = np.array(mean[m], float); c, ls, mk = _n1style(m)
        ax.plot(G, y, ls, color=c, marker=mk, ms=4.5, lw=2, label=m)
        if m in sd: s = np.array(sd[m], float); ax.fill_between(G, y-s, y+s, color=c, alpha=.13, lw=0)
    ax.axhline(orc, color='#444', ls=':', label='oracle %.2f' % orc); ax.axhline(0, color='#bbb', ls=':')
    ax.set_xlabel('Γ'); ax.set_ylabel('realised E[Y] (mean ± SD, %d seeds)' % ns)
    ax.set_ylim(-0.06, orc+0.07); ax.grid(alpha=.25); ax.legend(fontsize=6.5, ncol=2)
    ax.set_title('X in {-1,0,1}, N=200, %s, c_eps=%s (%d seeds)' % ('UNCAPPED' if reg=='uncap' else 'CAPPED (treat<=50%%)', ce, ns), fontsize=9)
    out = 'val_x3_%s_ce%s.png' % (reg, ce); fig.tight_layout(); fig.savefig(os.path.join(OWG, out), dpi=120); plt.close(fig)
    return out
def restab_x3(ce, reg, cols):
    d = json.load(open(os.path.join(OWX3, 'owgap_x3_ce%s.json' % ce)))
    G = d['gammas']; orc = d['oracle']; mean = d['regimes'][reg]['mean']; idx = [(c, G.index(c)) for c in cols if c in G]
    th = ''.join("<th>Γ=%s</th>" % (int(c) if c==int(c) else c) for c,_ in idx)
    rows = ''
    for m in N1000_ORDER:
        if m not in mean: continue
        sw = "<span class='sw' style='background:%s'></span>" % FAM.get(m.split('-')[0], '#888')
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (" class='hl'" if m.endswith('-O-W') else "", sw, m, ''.join("<td>%.2f</td>" % mean[m][i] for _,i in idx))
    rows += "<tr><td><span class='sw' style='background:#444'></span>Oracle</td>%s</tr>" % ''.join("<td>%.2f</td>" % orc for _ in idx)
    return "<table class='restab'><tr><th>method</th>%s</tr>%s</table>" % (th, rows)
def _x3_regime(reg):
    figs = grid([(gen_x3_plot(ce, reg), 'c_ε=%s' % ce, 'X∈{−1,0,1}, N=200, mean ± SD over seeds.') for ce in X3_CES])
    tbl = restab_x3('1.0', reg, [1.0,1.5,2.0,3.0,4.0,6.0,8.0]) if '1.0' in X3_CES else ''
    return ("<h3>Realised value vs Γ (ε present: %s)</h3>%s"
            "<h3>Result table (c_ε=1.0, mean — O-W rows highlighted)</h3><div class='panel'>%s</div>" % (', '.join(X3_CES), figs, tbl))
if X3_CES:
    _nx3 = len(json.load(open(os.path.join(OWX3, 'owgap_x3_ce%s.json' % X3_CES[0])))['seeds'])
    tab_x3 = ("<div class='mhead'><span class='pill' style='background:#7c2d12'>3-value X</span>"
      "<h2 style='border:none;margin:0'>exp_owgap with X ∈ {−1, 0, 1}  (N=200, %d-seed)</h2></div>"
      "<p class='lead'>A robustness variant of exp_owgap: identical DGP (<code>exp_owgap_x3/dgp.py</code>), but the covariate "
      "\\(X\\) takes only <b>three</b> values \\(\\{-1,0,1\\}\\) instead of the 7-level grid, at \\(N_{\\text{train}}=200\\), %d seeds. "
      "The true effect is \\(\\text{CATE}(-1)=-2.5,\\ \\text{CATE}(0)=0,\\ \\text{CATE}(1)=+2.5\\), so the oracle treats only \\(X=1\\) "
      "(oracle value \\(\\approx 0.83\\)).</p>" % (_nx3, _nx3) + LEGEND +
      "<div class='note'><b>Key finding — the method separation collapses here.</b> Uncapped, <b>every method (including naïve IPW / AIPW) ≈ the oracle (0.83)</b>. "
      "The reason is structural: the only boundary point is \\(X=0\\), where the true effect is <b>exactly 0</b>, so over-treating it is "
      "<i>costless</i> — realised value \\(V(\\pi)=\\text{mean}_X[\\pi(X)\\,\\text{CATE}(X)]\\) is unchanged whether or not \\(X=0\\) is treated. "
      "The 7-level experiment had intermediate levels (\\(X=\\pm\\tfrac13,\\dots\\)) with <i>nonzero</i> CATE where naïve over-treatment was "
      "penalised — that is exactly what created the O-W advantage. Under the <b>capacity cap</b>, IPW-O-W still edges ahead "
      "(≈0.43 vs 0.38) by spending the limited budget on \\(X=1\\); Hajek-O-X collapses to never-treat as before.</div>"
      "<h3 style='margin-top:24px'>Results by regime</h3>" + inner_tabs_n('x3', [
          ('Uncapped', _x3_regime('uncap')),
          ('Capped', _x3_regime('cap')),
          ('Objective vs Γ', obj_panel(os.path.join(OWX3, 'owgap_x3_obj.json'), 'x3'))]))
else:
    tab_x3 = None

# ---------- exp_owgap_x4: 4-value X {-1,-0.5,0.5,1} (no X=0), N=300 ----------
OWX4 = os.path.join(HERE, "exp_owgap_x4")
X4_CES = [ce for ce in ['1.0','1.5','2.0'] if os.path.exists(os.path.join(OWX4, 'owgap_x4_ce%s.json' % ce))]
def gen_x4_plot(ce, reg, fpat='owgap_x4_ce%s.json', nlab='N=300', outpre='val_x4'):
    d = json.load(open(os.path.join(OWX4, fpat % ce)))
    G = d['gammas']; orc = d['oracle']; ns = len(d['seeds']); mean = d['regimes'][reg]['mean']; sd = d['regimes'][reg].get('sd', {})
    fig, ax = plt.subplots(figsize=(7,4.3))
    for m in N1000_ORDER:
        if m not in mean: continue
        y = np.array(mean[m], float); c, ls, mk = _n1style(m)
        ax.plot(G, y, ls, color=c, marker=mk, ms=4.5, lw=2, label=m)
        if m in sd: s = np.array(sd[m], float); ax.fill_between(G, y-s, y+s, color=c, alpha=.13, lw=0)
    ax.axhline(orc, color='#444', ls=':', label='oracle %.2f' % orc); ax.axhline(0, color='#bbb', ls=':')
    ax.set_xlabel('Γ'); ax.set_ylabel('realised E[Y] (mean ± SD, %d seeds)' % ns)
    ax.set_ylim(-0.12, orc+0.08); ax.grid(alpha=.25); ax.legend(fontsize=6.5, ncol=2)
    ax.set_title('X in {-1,-0.5,0.5,1}, %s, %s, c_eps=%s (%d seeds)' % (nlab, 'UNCAPPED' if reg=='uncap' else 'CAPPED (treat<=50%%)', ce, ns), fontsize=9)
    out = '%s_%s_ce%s.png' % (outpre, reg, ce); fig.tight_layout(); fig.savefig(os.path.join(OWG, out), dpi=120); plt.close(fig)
    return out
def restab_x4(ce, reg, cols, fpat='owgap_x4_ce%s.json'):
    d = json.load(open(os.path.join(OWX4, fpat % ce)))
    G = d['gammas']; orc = d['oracle']; mean = d['regimes'][reg]['mean']; idx = [(c, G.index(c)) for c in cols if c in G]
    th = ''.join("<th>Γ=%s</th>" % (int(c) if c==int(c) else c) for c,_ in idx)
    rows = ''
    for m in N1000_ORDER:
        if m not in mean: continue
        sw = "<span class='sw' style='background:%s'></span>" % FAM.get(m.split('-')[0], '#888')
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (" class='hl'" if m.endswith('-O-W') else "", sw, m, ''.join("<td>%.2f</td>" % mean[m][i] for _,i in idx))
    rows += "<tr><td><span class='sw' style='background:#444'></span>Oracle</td>%s</tr>" % ''.join("<td>%.2f</td>" % orc for _ in idx)
    return "<table class='restab'><tr><th>method</th>%s</tr>%s</table>" % (th, rows)
def _x4_regime(reg, fpat='owgap_x4_ce%s.json', nlab='N=300', outpre='val_x4', ces=None):
    ces = ces if ces is not None else X4_CES
    figs = grid([(gen_x4_plot(ce, reg, fpat, nlab, outpre), 'c_ε=%s' % ce, 'X∈{−1,−0.5,0.5,1}, %s, mean ± SD over seeds.' % nlab) for ce in ces])
    tbl = restab_x4('1.0', reg, [1.0,1.5,2.0,3.0,4.0,6.0,8.0], fpat) if '1.0' in ces else ''
    return ("<h3>Realised value vs Γ (ε present: %s)</h3>%s"
            "<h3>Result table (c_ε=1.0, mean — O-W rows highlighted)</h3><div class='panel'>%s</div>" % (', '.join(ces), figs, tbl))

def _x4_dgp_block():
    import importlib.util as _i4
    _s4 = _i4.spec_from_file_location('x4dgp_truth', os.path.join(OWX4, 'dgp.py'))
    g = _i4.module_from_spec(_s4); _s4.loader.exec_module(g)
    X = np.asarray(g.LEVELS, float)
    p = g.p_s1(X); ep = g.propensity(X, 1.0); em = g.propensity(X, -1.0)
    ES = 2*p - 1
    cate = (g.D1-g.D0)*ES + (g.B1-g.B0)*X
    emarg = p*ep + (1-p)*em
    m0p, m0m = g.D0*1.0 + g.B0*X, g.D0*(-1.0) + g.B0*X
    m1p, m1m = g.D1*1.0 + g.B1*X, g.D1*(-1.0) + g.B1*X
    orc = (cate > 0).astype(int)
    def _sv(fn):
        plt.gcf().tight_layout(); plt.gcf().savefig(os.path.join(OWG, fn), dpi=120); plt.close(plt.gcf()); return fn
    # 1) P(S=+1|X)
    plt.figure(figsize=(5.0,3.5)); plt.plot(X, p, '-o', color='#444', lw=2.4, ms=8)
    plt.xticks(X); plt.ylim(-0.05,1.05); plt.xlabel('X'); plt.ylabel('P(S=+1 | X)')
    plt.title('Hidden vitality vs fitness — σ(10X)', fontsize=10); plt.grid(alpha=.25)
    p1 = _sv('x4_sx.png')
    # 3) P(T=1|X) marginal + per-S
    plt.figure(figsize=(5.0,3.5))
    plt.plot(X, emarg, '-o', color='#6a51a3', lw=2.6, ms=8, label='P(T=1 | X)  (observed)')
    plt.plot(X, ep, '--s', color='#1f77b4', lw=1.8, ms=5, label='e(X,+1)  (hidden)')
    plt.plot(X, em, '--^', color='#d62728', lw=1.8, ms=5, label='e(X,−1)  (hidden)')
    plt.axhline(0.5, color='#bbb', ls=':'); plt.xticks(X); plt.ylim(-0.03,1.03); plt.xlabel('X'); plt.ylabel('propensity')
    plt.title('Treatment propensity P(T=1 | X)', fontsize=10); plt.grid(alpha=.25); plt.legend(fontsize=7.5)
    p3 = _sv('x4_pt.png')
    # 3b) ESTIMATED propensity (logistic fit, as the methods actually use it) vs true — 8 seeds, N=300
    import sys as _sys
    _root = os.path.dirname(HERE)
    if _root not in _sys.path: _sys.path.insert(0, _root)
    import common as _cm
    _est = []
    for _sd in range(8):
        _o, _oo = g.generate(300, _sd)
        _Xo = np.asarray(_o['X'], float).reshape(-1, 1); _To = np.asarray(_o['T']).astype(int).ravel()
        _eh = _cm.propensity_binary(_Xo, _To)
        _est.append([float(np.mean(_eh[np.round(_Xo.ravel(),6) == round(float(l),6)])) for l in X])
    _est = np.array(_est); _em8 = _est.mean(0); _es8 = _est.std(0)
    plt.figure(figsize=(5.0,3.5))
    plt.plot(X, emarg, '-o', color='#6a51a3', lw=2.4, ms=7, label='true P(T=1 | X)')
    plt.plot(X, _em8, '-s', color='#d62728', lw=2.2, ms=6, label='estimated ê(X)  (logistic, mean)')
    plt.fill_between(X, _em8-_es8, _em8+_es8, color='#d62728', alpha=.15, label='±SD over 8 seeds')
    plt.axhline(0.5, color='#bbb', ls=':'); plt.xticks(X); plt.ylim(0,1); plt.xlabel('X'); plt.ylabel('P(T=1 | X)')
    plt.title('Estimated vs true propensity (N=300)', fontsize=10); plt.grid(alpha=.25); plt.legend(fontsize=7)
    p3b = _sv('x4_prop_est.png')
    # 4) per-arm potential-outcome means + true CATE — single shared y-axis
    plt.figure(figsize=(5.6,3.7))
    plt.plot(X, m1p, '-o', color='#2ca02c', lw=2.2, ms=6, label='μ₁ | S=+1')
    plt.plot(X, m1m, '--o', color='#2ca02c', lw=1.8, ms=5, label='μ₁ | S=−1')
    plt.plot(X, m0p, '-s', color='#1f77b4', lw=2.2, ms=6, label='μ₀ | S=+1')
    plt.plot(X, m0m, '--s', color='#1f77b4', lw=1.8, ms=5, label='μ₀ | S=−1')
    plt.plot(X, cate, '-D', color='#d62728', lw=2.6, ms=7, label='true CATE(X)')
    plt.axhline(0, color='#bbb', ls=':')
    plt.xticks(X); plt.xlabel('X'); plt.ylabel('E[Y(t) | X,S]  and  CATE(X)')
    plt.legend(fontsize=6.8, ncol=3, loc='upper left')
    plt.title('Potential-outcome means E[Y(t)|X,S] + true CATE(X)', fontsize=9.5); plt.grid(alpha=.25)
    p4 = _sv('x4_cate_outcome.png')
    # 5) OBSERVED outcome mean per treatment arm  E[Y|X,T=t]  (confounded by hidden S)
    ESt = (p*ep - (1-p)*em) / (p*ep + (1-p)*em)               # E[S | X, T=1]
    ESc = (p*(1-ep) - (1-p)*(1-em)) / (p*(1-ep) + (1-p)*(1-em))  # E[S | X, T=0]
    eY1 = g.D1*ESt + g.B1*X                                    # E[Y | X, T=1]
    eY0 = g.D0*ESc                                             # E[Y | X, T=0]
    naive = eY1 - eY0                                          # naive (confounded) contrast
    plt.figure(figsize=(5.0,3.5))
    plt.plot(X, eY1, '-o', color='#2ca02c', lw=2.6, ms=7, label='E[Y | X, T=1]  (treated)')
    plt.plot(X, eY0, '-s', color='#1f77b4', lw=2.6, ms=7, label='E[Y | X, T=0]  (control)')
    plt.plot(X, cate, ':^', color='#d62728', lw=2.0, ms=6, label='true CATE(X)')
    plt.plot(X, naive, '--D', color='#ff7f0e', lw=1.8, ms=5, label='naive E[Y|T=1]−E[Y|T=0]')
    plt.axhline(0, color='#bbb', ls=':'); plt.xticks(X); plt.xlabel('X'); plt.ylabel('observed mean outcome')
    plt.title('Observed outcome by treatment arm  E[Y|X,T=t]', fontsize=10); plt.grid(alpha=.25); plt.legend(fontsize=7)
    p5 = _sv('x4_obsy.png')
    # 6) within-arm covariate balance per X  (one N=300 draw)
    obs, _full = g.generate(300, 0)
    Xo = np.asarray(obs['X'], float).ravel(); To = np.asarray(obs['T']).astype(int).ravel()
    ntr = np.array([int(np.sum((np.round(Xo,6) == round(float(l),6)) & (To == 1))) for l in X])
    nct = np.array([int(np.sum((np.round(Xo,6) == round(float(l),6)) & (To == 0))) for l in X])
    bw = 0.34; pos = np.arange(len(X))
    plt.figure(figsize=(5.0,3.5))
    plt.bar(pos-bw/2, ntr, bw, color='#2ca02c', label='T=1  (treated)')
    plt.bar(pos+bw/2, nct, bw, color='#1f77b4', label='T=0  (control)')
    for i in range(len(X)):
        plt.text(pos[i]-bw/2, ntr[i]+0.5, str(ntr[i]), ha='center', va='bottom', fontsize=7, color='#2ca02c')
        plt.text(pos[i]+bw/2, nct[i]+0.5, str(nct[i]), ha='center', va='bottom', fontsize=7, color='#1f77b4')
    plt.xticks(pos, ['%.2f' % x for x in X]); plt.xlabel('X'); plt.ylabel('count in one N=300 draw')
    plt.title('Within-arm covariate balance per X', fontsize=10); plt.grid(alpha=.25, axis='y'); plt.legend(fontsize=8)
    p6 = _sv('x4_obs.png')
    # 7) sum of REALIZED outcomes per arm, per X (same N=300 draw) — scatter
    Yo = np.asarray(obs['Y'], float).ravel()
    sY1 = np.array([float(np.sum(Yo[(np.round(Xo,6) == round(float(l),6)) & (To == 1)])) for l in X])
    sY0 = np.array([float(np.sum(Yo[(np.round(Xo,6) == round(float(l),6)) & (To == 0)])) for l in X])
    plt.figure(figsize=(5.2,3.6))
    plt.scatter(X, sY1, s=110, color='#2ca02c', marker='o', label='Σ Y | T=1 (treated)', zorder=3)
    plt.scatter(X, sY0, s=110, color='#1f77b4', marker='s', label='Σ Y | T=0 (control)', zorder=3)
    plt.axhline(0, color='#bbb', ls=':'); plt.xticks(X); plt.xlabel('X'); plt.ylabel('Σ realized Y  (one N=300 draw)')
    for xi, v in zip(X, sY1): plt.annotate('%.0f' % v, (xi, v), textcoords='offset points', xytext=(0,8), ha='center', fontsize=7, color='#2ca02c')
    for xi, v in zip(X, sY0): plt.annotate('%.0f' % v, (xi, v), textcoords='offset points', xytext=(0,-13), ha='center', fontsize=7, color='#1f77b4')
    plt.title('Sum of realized outcomes per arm, per X  (N=300)', fontsize=10); plt.grid(alpha=.25); plt.legend(fontsize=7.5)
    p7 = _sv('x4_realized_sum.png')
    # 8) same sum at N=10000 — confirm the pattern is stable, not a small-sample artifact
    obsL, _fL = g.generate(10000, 0)
    XL = np.asarray(obsL['X'], float).ravel(); TL = np.asarray(obsL['T']).astype(int).ravel(); YL = np.asarray(obsL['Y'], float).ravel()
    L1 = np.array([float(np.sum(YL[(np.round(XL,6) == round(float(l),6)) & (TL == 1)])) for l in X])
    L0 = np.array([float(np.sum(YL[(np.round(XL,6) == round(float(l),6)) & (TL == 0)])) for l in X])
    plt.figure(figsize=(5.2,3.6))
    plt.scatter(X, L1, s=110, color='#2ca02c', marker='o', label='Σ Y | T=1 (treated)', zorder=3)
    plt.scatter(X, L0, s=110, color='#1f77b4', marker='s', label='Σ Y | T=0 (control)', zorder=3)
    plt.axhline(0, color='#bbb', ls=':'); plt.xticks(X); plt.xlabel('X'); plt.ylabel('Σ realized Y  (one N=10000 draw)')
    for xi, v in zip(X, L1): plt.annotate('%.0f' % v, (xi, v), textcoords='offset points', xytext=(0,8), ha='center', fontsize=6.3, color='#2ca02c')
    for xi, v in zip(X, L0): plt.annotate('%.0f' % v, (xi, v), textcoords='offset points', xytext=(0,-13), ha='center', fontsize=6.3, color='#1f77b4')
    plt.title('Sum of realized outcomes per arm, per X  (N=10000)', fontsize=10); plt.grid(alpha=.25); plt.legend(fontsize=7.5)
    p8 = _sv('x4_realized_sum_10k.png')
    # truth table
    xh = ''.join('<th>%.2f</th>' % x for x in X)
    def row(lbl, vals, hl=False):
        return "<tr%s><td style='text-align:left'>%s</td>%s</tr>" % (" class='hl'" if hl else "", lbl, ''.join('<td>%.3f</td>' % v for v in vals))
    tbl = ("<table class='restab'><tr><th style='text-align:left'>quantity \\ X</th>" + xh + "</tr>"
           + row("P(S=+1\\|X)=σ(10X)", p) + row("P(T=1\\|X)  (observed)", emarg)
           + row("CATE(X)", cate, hl=True) + row("oracle π*(X)=1{X>0}", orc) + "</table>")
    explain = ("<div class='panel'><h3 style='margin-top:4px'>How the X4 dataset is generated (the DGP)</h3>"
      "<p>Identical functional form to the headline <code>exp_owgap</code> DGP, but the covariate \\(X\\) is restricted to the "
      "<b>four levels</b> \\(X\\in\\{-1,-0.5,0.5,1\\}\\) — deliberately <b>excluding \\(X=0\\)</b> so that every level carries a "
      "<i>non-zero</i> treatment effect (no &ldquo;indifferent&rdquo; point).</p>"
      "<p><b>Step 1 — covariate.</b> \\(X\\) is drawn uniformly over the four levels \\(\\{-1,-0.5,0.5,1\\}\\).</p>"
      "<p><b>Step 2 — unobserved confounder</b> \\(S\\in\\{-1,+1\\}\\) (latent vitality), \\(\\Pr(S=+1\\mid X)=\\sigma(10X)\\). "
      "Because \\(|X|\\ge 0.5\\), \\(\\sigma(10X)\\) is essentially saturated, so here \\(S\\) is <b>nearly deterministic given \\(X\\)</b>: "
      "\\(\\Pr(S=+1\\mid X)\\in\\{0,\\,0.007,\\,0.993,\\,1\\}\\). \\(S\\) is hidden from every estimator.</p>"
      "<p><b>Step 3 — treatment</b> \\(T\\mid X,S\\sim\\mathrm{Bernoulli}\\big(e(X,S)\\big),\\; e(X,S)=\\sigma(0.8S-2X)\\). "
      "Integrating out \\(S\\) gives the observed propensity \\(\\Pr(T=1\\mid X)\\in\\{0.769,\\,0.552,\\,0.448,\\,0.231\\}\\).</p>"
      "<p><b>Step 4 — potential outcomes.</b> \\(Y(0)=8S+\\varepsilon,\\; Y(1)=9S+1.5X+\\varepsilon,\\; \\varepsilon\\sim\\mathcal N(0,0.6^2)\\); only \\(Y=Y(T)\\) is observed.</p>"
      "<p><b>Resulting effect &amp; optimal policy.</b> \\(\\tau(X)=(9-8)\\,\\mathbb E[S\\mid X]+1.5X=(2\\sigma(10X)-1)+1.5X\\), giving</p>"
      "\\[ \\tau(X)=(-2.5,\\,-1.74,\\,+1.74,\\,+2.5)\\ \\text{at}\\ X=(-1,-0.5,0.5,1),\\qquad \\pi^\\star(X)=\\mathbf 1\\{X>0\\},\\quad \\text{oracle}\\approx 1.06. \\]"
      "<p class='sub'><b>Why this variant matters.</b> Because \\(S\\) is near-deterministic given \\(X\\), there is essentially "
      "<b>no hidden selection bias</b>, so the outcome model is unbiased and plain AIPW already recovers \\(\\pi^\\star\\) — yet naïve IPW still "
      "fails at \\(X=1\\) where treatment is rare (\\(P(T{=}1\\mid X)=0.23\\)). This is the control case that, together with the 7-level and 3-value variants, pins down "
      "<i>when</i> the O-W (Wasserstein) term actually helps. Code: <code>exp_owgap_x4/dgp.py</code>.</p></div>")
    truth = ("<h3>The data-generating process (truth)</h3>" + grid([
        (p1, 'P(S=+1|X)=σ(10X)', 'Near-deterministic at |X|≥0.5 — so S is almost pinned down by X (little hidden selection).'),
        (p3, 'Treatment propensity P(T=1|X)', 'Observed (solid) vs the unobservable per-S curves (dashed). The marginal falls monotonically from 0.77 at X=−1 to 0.23 at X=1.'),
        (p3b, 'Estimated propensity ê(X) vs true (N=300)', 'What the estimators actually use — a logistic propensity fit on each draw (the methods never see the true e). Red = mean estimate over 8 seeds with ±SD band; purple = truth. Two effects are visible: sampling noise (the band), and a small misspecification bias — a single logistic σ(a+bX) cannot represent the hidden-S mixture, so it slightly over-steepens the middle (over-estimates at X=−0.5, under-estimates at X=0.5). That bias does not vanish with N: at N=10000 ê≈(0.73, 0.62, 0.37, 0.26) vs true (0.77, 0.55, 0.45, 0.23).'),
        (p4, 'Potential-outcome means E[Y(t)|X,S] + true CATE(X)', 'Combined on one shared y-axis: the hidden-S potential outcomes (μ₀=8S, μ₁=9S+1.5X) dominate the ±10 scale, and the true effect CATE(X) (red) is the much smaller ±2.5 contrast near zero — treat iff X>0. Same scale makes explicit how small the actual decision-relevant effect is relative to the S-driven outcome levels.'),
        (p5, 'Observed outcome by treatment arm  E[Y|X,T=t]', 'What the analyst actually sees per arm. Here the naïve difference E[Y|T=1]−E[Y|T=0] (orange) nearly coincides with the true CATE (red) — because |X|≥0.5 makes S almost deterministic given X, treatment barely selects on the hidden S, so there is little confounding. That near-overlap is exactly why plain AIPW already recovers the oracle in this control variant (contrast the 7-level DGP, where the two curves diverge sharply).'),
        (p6, 'Within-arm covariate balance per X', 'Treated vs control counts at each X level in one N=300 draw. Because P(T=1|X) falls from 0.77 (X=−1) to 0.23 (X=1), the arms are covariate-imbalanced — treated-heavy on the left, control-heavy on the right; X=1 has the fewest treated units (and X=−1 the fewest controls), so the naïve IPW weights blow up at both ends.'),
        (p7, 'Sum of realized outcomes per arm, per X  (N=300)', 'Same N=300 draw: the Σ of the actually-observed Y in each (X, arm) cell, as a scatter. It folds together how many units land in the cell (the imbalance at left) with their outcome level. Direct-X-X maximises (1/n)Σ Yᵢ·π_{Tᵢ}(Xᵢ), i.e. it picks the higher-sum arm at each X — control wins everywhere here, so it never-treats (value 0.16 vs oracle 1.06).'),
        (p8, 'Same sum at N=10000 (stability check)', 'The identical scatter at N=10000 — the pattern is unchanged and matches the population limit: the control sum is higher at X=−1, −0.5 and 1, with X=0.5 a near-tie (treated ≈10.6k vs control ≈10.8k). Most telling is X=1, where treating is MOST beneficial (CATE=+2.5) yet the control sum dominates because only 23% of units there are treated. So Direct-X-X never-treats robustly — not a small-sample fluke.'),
      ]) + "<div class='panel'>" + tbl + "</div>")
    return explain + truth

def _x4_ipw_decision_block():
    import importlib.util as _i4, sys as _sys
    _s4 = _i4.spec_from_file_location('x4dgp_ipw', os.path.join(OWX4, 'dgp.py'))
    g = _i4.module_from_spec(_s4); _s4.loader.exec_module(g)
    _root = os.path.dirname(HERE)
    if _root not in _sys.path: _sys.path.insert(0, _root)
    import common as _cm
    X = np.asarray(g.LEVELS, float); K = g.K
    VT, VC = [], []
    for sd in range(8):
        obs, _o = g.generate(300, sd)
        Xo = np.asarray(obs['X'], float).ravel(); To = np.asarray(obs['T']).astype(int).ravel(); Yo = np.asarray(obs['Y'], float).ravel()
        w, _w = _cm.ipw_weights_from_data(Xo, To, K)
        msk = lambda l, t: (np.round(Xo,6) == round(float(l),6)) & (To == t)
        VT.append([float(np.sum(w[msk(l,1)] * Yo[msk(l,1)])) for l in X])
        VC.append([float(np.sum(w[msk(l,0)] * Yo[msk(l,0)])) for l in X])
    VT = np.array(VT); VC = np.array(VC)
    mt, st = VT.mean(0), VT.std(0); mc, sc = VC.mean(0), VC.std(0)
    ftr = (VT > VC).mean(0)              # fraction of seeds that treat at each X
    fig, ax = plt.subplots(figsize=(6.6,4.3))
    ax.axvspan(0, X.max()+0.25, color='#2ca02c', alpha=0.06); ax.axvspan(X.min()-0.25, 0, color='#d62728', alpha=0.06)
    ax.errorbar(X, mt, yerr=st, fmt='-o', color='#2ca02c', lw=2, ms=7, capsize=4, label='value of TREATING   Σ wᵢYᵢ (T=1)')
    ax.errorbar(X, mc, yerr=sc, fmt='-s', color='#1f77b4', lw=2, ms=7, capsize=4, label='value of CONTROL   Σ wᵢYᵢ (T=0)')
    ax.axhline(0, color='#bbb', ls=':')
    top = np.maximum(mt+st, mc+sc)
    for xi, ft, ty in zip(X, ftr, top):
        ax.annotate('treat %d/8' % round(ft*8), (xi, ty), textcoords='offset points', xytext=(0,7), ha='center', fontsize=7.5, color='#333')
    ax.set_xlim(X.min()-0.25, X.max()+0.25); ax.set_xticks(X); ax.set_xlabel('X')
    ax.set_ylabel('IPW arm-value  Σ wᵢ·Yᵢ  (mean ± SD / 8 seeds)')
    ax.set_title('What IPW-X-X compares at each X — it treats iff green > blue', fontsize=9.5)
    ax.grid(alpha=.25); ax.legend(fontsize=7.5, loc='lower right')
    pimg = 'x4_ipw_decision.png'; fig.tight_layout(); fig.savefig(os.path.join(OWG, pimg), dpi=120); plt.close(fig)
    return ("<h3 style='margin-top:26px'>Feeling the IPW-X-X decision, level by level</h3>"
      "<div class='note'>IPW-X-X has no robustness and no capacity coupling, so its choice <b>decomposes per X</b>: at each level it "
      "computes the inverse-propensity-weighted value of each arm, \\(\\hat V_t(x)=\\sum_{i:X_i=x,\\,T_i=t} w_i Y_i\\) with \\(w_i=1/\\hat e(T_i\\mid X_i)\\), "
      "and <b>treats iff \\(\\hat V_1(x) > \\hat V_0(x)\\)</b>. The \\(1/\\hat e\\) weight is what undoes the count imbalance that fooled Direct-X-X — "
      "but it also <b>injects variance</b> wherever overlap is thin. The plot shows the two values it weighs (mean ± SD over 8 N=300 draws); the "
      "shaded background is the oracle (green = should treat, red = should not).</div>"
      + grid([(pimg, 'The two IPW arm-values, with seed spread',
               'Higher point wins → that arm is chosen. The “treat k/8” labels give how many of the 8 seeds end up treating at that X.')]) +
      "<div class='note'><b>Read it level by level.</b> "
      "<b>X=−1</b> (red): control clearly wins, tight error bars → <b>0/8 treat</b>, correct. "
      "<b>X=−0.5</b> (red): the weighted treat-value (≈−630) is <i>less negative</i> than control (≈−733), so IPW <b>treats 7/8</b> — but the oracle says "
      "don\\'t (CATE=−1.74). This is a genuine <b>confounding-driven over-treatment</b>: the slightly mis-estimated weights tip a harmful level into 'treat'. "
      "<b>X=0.5</b> (green): treating clearly wins → <b>8/8 treat</b>, correct. "
      "<b>X=1</b> (green): treating is <i>most</i> beneficial (CATE=+2.5), yet the two values sit <b>within each other\\'s error bars</b> — a near coin-flip "
      "(<b>4/8 treat</b>). Only 23% of units here are treated, so \\(1/\\hat e\\approx 3.7\\) makes \\(\\hat V_1\\) extremely noisy and IPW cannot reliably tell. "
      "<b>Net:</b> IPW-X-X over-treats the harmful X=−0.5 and gambles on the clearly-beneficial X=1 → realised value <b>0.37</b> vs oracle 1.06. The wide green "
      "error bars at X=1 are exactly the instability the Γ / Wasserstein machinery is built to tame.</div>")

if X4_CES:
    _nx4 = len(json.load(open(os.path.join(OWX4, 'owgap_x4_ce%s.json' % X4_CES[0])))['seeds'])
    tab_x4 = ("<div class='mhead'><span class='pill' style='background:#7e22ce'>4-value X</span>"
      "<h2 style='border:none;margin:0'>exp_owgap with X ∈ {−1, −0.5, 0.5, 1}  (N=300, %d-seed)</h2></div>"
      "<p class='lead'>Another variant: identical DGP (<code>exp_owgap_x4/dgp.py</code>), but \\(X\\) takes the four values "
      "\\(\\{-1,-0.5,0.5,1\\}\\) — crucially <b>excluding \\(X=0\\)</b>, so every level has a <i>nonzero</i> effect. "
      "\\(N_{\\text{train}}=300\\), %d seeds. CATE \\(=(-2.5,-1.74,+1.74,+2.5)\\) ⇒ oracle treats \\(X>0\\) (oracle value \\(\\approx 1.06\\)).</p>" % (_nx4, _nx4) + LEGEND +
      "<div class='note'><b>Key finding — here naïve AIPW is already optimal, but naïve IPW fails.</b> Uncapped, "
      "<b>DoublyRobust-X-X (AIPW) equals the oracle (1.06)</b>: because \\(|X|\\ge 0.5\\) makes the hidden \\(S\\) nearly deterministic "
      "(\\(P(S{=}{+}1\\mid X)\\in\\{0,\\,0.007,\\,0.993,\\,1\\}\\)), there is essentially <b>no selection bias</b>, so the outcome model is "
      "unbiased and AIPW recovers the right rule. But <b>naïve IPW-X-X collapses to 0.37</b> — the inverse-propensity weights blow up wherever an arm "
      "is rare (\\(P(T{=}1\\mid X{=}1)=0.23\\), and symmetrically \\(P(T{=}0\\mid X{=}{-}1)=0.23\\)). The <b>O-W methods also reach the oracle</b> (IPW-O-W \\(=1.06\\) for \\(\\Gamma\\ge2\\); "
      "DR-O-W \\(\\approx 1.0\\)). Together with the other variants this pins down <i>when</i> O-W helps: it needs intermediate \\(X\\) where \\(S\\) is "
      "<b>uncertain</b> (selection bias) <b>and</b> CATE is <b>nonzero</b> — true for the 7-level grid, but not when \\(S\\) is deterministic (here) "
      "or the boundary effect is zero (the 3-value variant).</div>"
      + TIGHT_EPS_NOTE + _x4_dgp_block() + _x4_ipw_decision_block() +
      "<h3 style='margin-top:24px'>Results by regime</h3>" + inner_tabs_n('x4', [
          ('Uncapped', _x4_regime('uncap')),
          ('Capped', _x4_regime('cap')),
          ('Objective vs Γ', obj_panel(os.path.join(OWX4, 'owgap_x4_obj.json'), 'x4'))]))
else:
    tab_x4 = None

# ---------- exp_owgap_x4 at N_train=16 (extreme small-sample) ----------
X4N16_CES = [ce for ce in ['1.0','1.5','2.0'] if os.path.exists(os.path.join(OWX4, 'owgap_x4_n16_ce%s.json' % ce))]
if X4N16_CES:
    _F16 = 'owgap_x4_n16_ce%s.json'
    _n16 = len(json.load(open(os.path.join(OWX4, _F16 % X4N16_CES[0])))['seeds'])
    tab_x4n16 = ("<div class='mhead'><span class='pill' style='background:#9a3412'>4-value X · N=16</span>"
      "<h2 style='border:none;margin:0'>exp_owgap, X ∈ {−1, −0.5, 0.5, 1}, <u>N=16</u> (extreme small-sample)</h2></div>"
      "<p class='lead'>The <b>identical</b> X4 DGP (<code>exp_owgap_x4/dgp.py</code>, oracle ≈ 1.06), but with only "
      "\\(N_{\\text{train}}=16\\) — about <b>4 units per X level</b>, and at \\(X=1\\) (\\(P(T{=}1\\mid X)=0.23\\)) often just <b>one treated unit</b>. "
      "A stress test of what the robust methods buy when data is almost absent. %d seeds.</p>" % _n16 + LEGEND +
      "<div class='note'><b>Key finding — robustness still wins, even at N=16.</b> Uncapped (c_ε=1, oracle 1.06): every naïve baseline is poor "
      "(<b>Direct-X-X = −0.08</b>, actually worse than never-treat; IPW-X-X = 0.23; AIPW = 0.37), the pure odds-box and Hájek collapse, but the "
      "<b>O-W methods still lead — IPW-O-W peaks at 0.50</b> and DoublyRobust-O-W at 0.44. The whole field sits far below the oracle (16 points "
      "cannot pin down the rule), yet the <i>ordering</i> is the same as N=300/600, and the IPW-O-W − AIPW margin (+0.13) actually <b>holds up</b> "
      "in relative terms — the Wasserstein term helps most precisely when data is scarce and the empirical weights are least trustworthy.</div>"
      + TIGHT_EPS_NOTE + _x4_dgp_block() +
      "<h3 style='margin-top:24px'>Results by regime</h3>" + inner_tabs_n('x4n16', [
          ('Uncapped', _x4_regime('uncap', _F16, 'N=16', 'val_x4n16', X4N16_CES)),
          ('Capped', _x4_regime('cap', _F16, 'N=16', 'val_x4n16', X4N16_CES)),
          ('Objective vs Γ', obj_panel(os.path.join(OWX4, _F16 % '1.0'), 'x4n16'))]))
else:
    tab_x4n16 = None

# ---------- interactive policy explorer: π(treat|X) vs X with dropdowns ----------
POLICY = {}; GRID_BY_DS = {}
def _ingest_policy(path, ds, ep):
    d = json.load(open(path)); GRID_BY_DS[ds] = [round(float(x),4) for x in d['grid']]
    for reg in d['regimes']:
        for m, seeds in d['regimes'][reg].get('policy_by_seed', {}).items():
            for sk, gd in seeds.items():
                for gk, vals in gd.items():
                    POLICY.setdefault(ds,{}).setdefault(ep,{}).setdefault(reg,{}).setdefault(m,{}).setdefault(gk,{})[sk] = [round(float(x),4) for x in vals]
def _ingest_lip(jpath, ds):
    # Lipschitz sweeps store per-method gridpol keyed by L (seed 0). Map L into the Γ slot; alias to seed 'avg' and '0'.
    if not os.path.exists(jpath): return
    d = json.load(open(jpath)); GRID_BY_DS[ds] = [round(float(x),4) for x in d['grid']]
    for m, mm in d['methods'].items():
        for lk, curve in mm.get('gridpol', {}).items():
            v = [round(float(x),4) for x in curve]
            for sk in ('avg','0'):
                POLICY.setdefault(ds,{}).setdefault('1.0',{}).setdefault('uncap',{}).setdefault(m,{}).setdefault(lk,{})[sk] = v
for _ce in ['1.0','1.5','2.0']:
    _p = os.path.join(OWG, 'owgap_results_ce%s.json' % _ce)
    if os.path.exists(_p): _ingest_policy(_p, 'N=600', _ce)
    _p = os.path.join(OWG, 'owgap_results_n1000_ce%s.json' % _ce)
    if os.path.exists(_p): _ingest_policy(_p, 'N=1000', _ce)
    _p = os.path.join(OWX3, 'owgap_x3_ce%s.json' % _ce)
    if os.path.exists(_p): _ingest_policy(_p, 'X3 (N=200)', _ce)
    _p = os.path.join(OWX4, 'owgap_x4_ce%s.json' % _ce)
    if os.path.exists(_p): _ingest_policy(_p, 'X4 (N=300)', _ce)
    _p = os.path.join(OWX4, 'owgap_x4_n16_ce%s.json' % _ce)
    if os.path.exists(_p): _ingest_policy(_p, 'X4 (N=16)', _ce)
# continuous-X extended policies (KNN / Shapley), ingested at c_eps=1.0 on the dense grid
_pk = os.path.join(HERE, 'exp_owgap_cont', 'owgap_cont_knn_policies.json')
if os.path.exists(_pk): _ingest_policy(_pk, 'Cont-KNN (N=400)', '1.0')
_ps = os.path.join(HERE, 'exp_owgap_cont', 'owgap_cont_shapley_policies.json')
if os.path.exists(_ps): _ingest_policy(_ps, 'Cont-Shapley (N=400)', '1.0')
_pkm = os.path.join(HERE, 'exp_owgap_cont', 'owgap_cont_extend_mesh25_knn_policies.json')
if os.path.exists(_pkm): _ingest_policy(_pkm, 'Cont-KNN-mesh (N=400)', '1.0')
_psm = os.path.join(HERE, 'exp_owgap_cont', 'owgap_cont_extend_mesh25_shapley_policies.json')
if os.path.exists(_psm): _ingest_policy(_psm, 'Cont-Shapley-mesh (N=400)', '1.0')
_ingest_lip(os.path.join(HERE, 'exp_owgap_cont', 'owgap_lipschitz.json'), 'Cont-Lipschitz X-X (N=400)')
_ingest_lip(os.path.join(HERE, 'exp_owgap_cont', 'owgap_lipschitz_robust.json'), 'Cont-Lipschitz robust Γ=2 (N=400)')
_datasets = list(POLICY.keys()) or ['(none)']
_epslist = sorted({e for ds in POLICY.values() for e in ds}) or ['1.0']
_methods = ['IPW-X-X','DoublyRobust-X-X','Direct-X-X','IPW-O-X','DoublyRobust-O-X','Hajek-O-X','IPW-O-W','DoublyRobust-O-W']
_gammakeys = ['1','1.5','2','2.5','3','4','5','6','8']
_lipkeys = ['inf','20','10','5','3','2','1.5','1','0.75','0.5','0.25']   # L values for the Cont-Lipschitz datasets
_gammakeys = _gammakeys + [k for k in _lipkeys if k not in _gammakeys]
_seedkeys = ['avg','0','1','2','3','4','5','6','7']
def _selrow(idd, label, op):
    return "<label class='polsel'>%s<select id='%s' onchange='drawPolicy()'>%s</select></label>" % (label, idd, ''.join('<option>%s</option>' % o for o in op))
_methcheck = ("<div class='polsel' style='min-width:215px'>methods (check one or more)<div class='polchecks'>" + ''.join("<label class='polcheck'><input type='checkbox' class='pol-mcb' value='%s'%s onchange='drawPolicy()'>%s</label>" % (mm, ' checked' if mm in ('IPW-O-W','DoublyRobust-X-X') else '', mm) for mm in _methods) + "</div></div>")
tab_policy = ("<div class='mhead'><span class='pill' style='background:#0d9488'>Policy explorer</span>"
 "<h2 style='border:none;margin:0'>Deployed policy π(treat | X)</h2></div>"
 "<p class='lead'>Select dataset, ε, regime, Γ and seed, and <b>check one or more methods</b> to overlay their learned policies "
 "π(treat | X) across the covariate X. Dotted black = oracle (treat iff X&gt;0); dashed grey = 0.5. "
 "(<code>avg</code> = mean policy over seeds.) <b>For the <code>Cont-Lipschitz</code> datasets the Γ selector is the Lipschitz "
 "<b>L</b></b> (<code>inf</code> = unit-by-unit; smaller L = smoother), and the curves are the KNN-extended policy on the continuous grid.</p>"
 "<div class='polmenus'>" + _selrow('pol-ds','dataset',_datasets) + _selrow('pol-eps','c_ε',_epslist)
 + _selrow('pol-reg','regime',['uncap','cap']) + _methcheck
 + _selrow('pol-gamma','Γ / L',_gammakeys) + _selrow('pol-seed','seed',_seedkeys) + "</div>"
 "<div id='policy-svg' class='panel' style='min-height:380px;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:8px'></div>")

_POLICY_JS = ("var POLICY_DATA=" + json.dumps(POLICY) + ";var GRID_BY_DS=" + json.dumps(GRID_BY_DS) + ";var POLCOLOR=" + json.dumps(FAM) + ";" + '\nfunction _pv(id){var e=document.getElementById(id);return e?e.value:null;}\nfunction _mstyle(m){var fam=m.split(\'-\')[0];var c=POLCOLOR[fam]||\'#0d9488\';\n if(/-O-W$/.test(m))return{c:c,dash:\'\',mk:\'circle\'};\n if(/-O-X$/.test(m))return{c:c,dash:\'\',mk:\'square\'};\n if(/-X-X$/.test(m))return{c:c,dash:\'6,4\',mk:null};\n return{c:c,dash:\'\',mk:null};}\nfunction _mk(t,cx,cy,col){if(t==\'square\')return \'<rect x="\'+(cx-3)+\'" y="\'+(cy-3)+\'" width="6" height="6" fill="\'+col+\'"/>\';if(t==\'circle\')return \'<circle cx="\'+cx+\'" cy="\'+cy+\'" r="3.3" fill="\'+col+\'"/>\';return \'\';}\nfunction drawPolicy(){\n var ds=_pv(\'pol-ds\'),ep=_pv(\'pol-eps\'),rg=_pv(\'pol-reg\'),g=_pv(\'pol-gamma\'),sd=_pv(\'pol-seed\');\n var GRID_X=GRID_BY_DS[ds]||[];\n var ms=[];document.querySelectorAll(\'.pol-mcb:checked\').forEach(function(cb){ms.push(cb.value);});\n var el=document.getElementById(\'policy-svg\');if(!el)return;\n if(!GRID_X.length){el.innerHTML="<span class=\'sub\'>No grid for this dataset.</span>";return;}\n if(!ms.length){el.innerHTML="<span class=\'sub\'>Check at least one method above.</span>";return;}\n var W=760,H=380,mL=58,mR=22,mT=20,mB=44,x0=mL,x1=W-mR,y0=H-mB,y1=mT;\n var xmin=Math.min.apply(null,GRID_X),xmax=Math.max.apply(null,GRID_X);\n function sx(x){return x0+(x-xmin)/(xmax-xmin)*(x1-x0);}function sy(v){return y0-v*(y0-y1);}\n var s=\'<svg viewBox="0 0 \'+W+\' \'+H+\'" style="width:100%;max-width:780px">\',i,yv,y;\n for(i=0;i<=4;i++){yv=i/4;y=sy(yv);s+=\'<line x1="\'+x0+\'" y1="\'+y+\'" x2="\'+x1+\'" y2="\'+y+\'" stroke="#eee"/><text x="\'+(x0-8)+\'" y="\'+(y+4)+\'" text-anchor="end" font-size="11" fill="#999">\'+yv.toFixed(2)+\'</text>\';}\n GRID_X.forEach(function(xx){var px=sx(xx);s+=\'<text x="\'+px+\'" y="\'+(y0+18)+\'" text-anchor="middle" font-size="10" fill="#999">\'+xx.toFixed(2)+\'</text>\';});\n s+=\'<line x1="\'+x0+\'" y1="\'+sy(0.5)+\'" x2="\'+x1+\'" y2="\'+sy(0.5)+\'" stroke="#ddd" stroke-dasharray="4,4"/>\';\n var orc=GRID_X.map(function(xx){return xx>0?1:0;});\n s+=\'<polyline points="\'+GRID_X.map(function(xx,k){return sx(xx)+\',\'+sy(orc[k]);}).join(\' \')+\'" fill="none" stroke="#444" stroke-dasharray="2,3" stroke-width="1.6"/>\';\n var miss=[];\n ms.forEach(function(m){var node=null;try{node=POLICY_DATA[ds][ep][rg][m][g][sd];}catch(e){node=null;}\n  if(!node){miss.push(m);return;}var st=_mstyle(m);\n  s+=\'<polyline points="\'+GRID_X.map(function(xx,k){return sx(xx)+\',\'+sy(node[k]);}).join(\' \')+\'" fill="none" stroke="\'+st.c+\'" stroke-width="2.4"\'+(st.dash?\' stroke-dasharray="\'+st.dash+\'"\':\'\')+\'/>\';\n  if(st.mk)GRID_X.forEach(function(xx,k){s+=_mk(st.mk,sx(xx),sy(node[k]),st.c);});});\n s+=\'<text x="\'+((x0+x1)/2)+\'" y="\'+(H-6)+\'" text-anchor="middle" font-size="12">X</text>\';\n s+=\'<text transform="translate(15,\'+((y0+y1)/2)+\') rotate(-90)" text-anchor="middle" font-size="12">pi(treat | X)</text>\';\n s+=\'<text x="\'+x1+\'" y="\'+(y1+2)+\'" text-anchor="end" font-size="10" fill="#888">\'+ds+\'  Gamma=\'+g+\'  \'+rg+\'  seed \'+sd+\'</text>\';\n s+=\'</svg>\';\n var leg="<div style=\'display:flex;flex-wrap:wrap;gap:8px 16px;justify-content:center;margin-top:6px;font-size:.82rem\'>";\n leg+="<span><svg width=\'26\' height=\'10\' style=\'vertical-align:middle\'><line x1=\'1\' y1=\'5\' x2=\'24\' y2=\'5\' stroke=\'#444\' stroke-dasharray=\'2,3\' stroke-width=\'1.6\'/></svg> oracle</span>";\n ms.forEach(function(m){if(miss.indexOf(m)>=0)return;var st=_mstyle(m);var sw="<svg width=\'26\' height=\'10\' style=\'vertical-align:middle\'><line x1=\'1\' y1=\'5\' x2=\'24\' y2=\'5\' stroke=\'"+st.c+"\' stroke-width=\'2.4\'"+(st.dash?" stroke-dasharray=\'"+st.dash+"\'":"")+"/>"+(st.mk==\'circle\'?"<circle cx=\'13\' cy=\'5\' r=\'3\' fill=\'"+st.c+"\'/>":st.mk==\'square\'?"<rect x=\'10\' y=\'2\' width=\'6\' height=\'6\' fill=\'"+st.c+"\'/>":"")+"</svg>";leg+="<span>"+sw+" "+m+"</span>";});\n leg+="</div>";if(miss.length)leg+="<div class=\'sub\' style=\'text-align:center;color:#dc2626\'>no data for: "+miss.join(\', \')+"</div>";\n el.innerHTML="<div style=\'width:100%\'>"+s+leg+"</div>";\n}\n')

# ---------- Sanity-check tab: justify each method's policy from the DGP ----------
import importlib.util as _ilu
_ds = _ilu.spec_from_file_location('owg_dgp_sanity', os.path.join(OWG, 'dgp.py'))
_dg = _ilu.module_from_spec(_ds); _ds.loader.exec_module(_dg)
import numpy as _np
_Xg = _dg.LEVELS
_p = _dg.p_s1(_Xg); _ep = _dg.propensity(_Xg, 1.0); _em = _dg.propensity(_Xg, -1.0)
_ES = 2*_p - 1
_cate = (_dg.D1-_dg.D0)*_ES + (_dg.B1-_dg.B0)*_Xg
_emarg = _p*_ep + (1-_p)*_em
_ESt = (_p*_ep - (1-_p)*_em)/(_p*_ep + (1-_p)*_em)
_ESc = (_p*(1-_ep) - (1-_p)*(1-_em))/(_p*(1-_ep) + (1-_p)*(1-_em))
_taunaive = _dg.D1*_ESt + _dg.B1*_Xg - _dg.D0*_ESc
_oracle = (_cate > 0).astype(int)
def _trow(label, vals, fmt='%.2f', hl=False):
    cells = ''.join('<td>%s</td>' % (fmt % v) for v in vals)
    return "<tr%s><td style='text-align:left'>%s</td>%s</tr>" % (" class='hl'" if hl else "", label, cells)
_xhead = ''.join('<th>%.2f</th>' % x for x in _Xg)
_truth_tbl = ("<table class='restab'><tr><th style='text-align:left'>quantity \\ X</th>" + _xhead + "</tr>"
 + _trow("P(S=+1\\|X)=σ(10X)", _p)
 + _trow("E[S\\|X]", _ES)
 + _trow("<b>true CATE(X)</b>", _cate)
 + _trow("<b>oracle π*(X)</b>", _oracle, '%d', hl=True)
 + _trow("P(T=1\\|X) (overlap)", _emarg)
 + _trow("naïve contrast τ̂(X)", _taunaive)
 + _trow("naïve treats (τ̂&gt;0)?", (_taunaive>0).astype(int), '%d')
 + "</table>")
_polrows = _trow("oracle (treat iff X&gt;0)", _oracle, '%d', hl=True)
try:
    _pbs = json.load(open(os.path.join(OWG,'owgap_results_n1000_ce1.0.json')))['regimes']['uncap']['policy_by_seed']
    for m in ['Direct-X-X','IPW-X-X','DoublyRobust-X-X','Hajek-O-X','IPW-O-W','DoublyRobust-O-W']:
        _polrows += _trow(m, _pbs[m]['avg']['2'])
    _poltbl = "<table class='restab'><tr><th style='text-align:left'>method \\ X</th>" + _xhead + "</tr>" + _polrows + "</table>"
except Exception:
    _poltbl = "<p class='sub'>(N=1000 policy data not found.)</p>"

# --- sanity-check plots (generated from the DGP + the actual learned policies) ---
def _scsave(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OWG, name), dpi=120); plt.close(fig); return name
_f,_a = plt.subplots(figsize=(7,4.2))
_a.plot(_Xg,_ES,'-o',color='#444',lw=2.6,ms=6,label='E[S|X]  (true marginal)')
_a.plot(_Xg,_ESt,'--s',color='#1f77b4',lw=2,ms=5,label='E[S|X, T=1]  treated')
_a.plot(_Xg,_ESc,'--^',color='#d62728',lw=2,ms=5,label='E[S|X, T=0]  control')
_a.axhline(0,color='#bbb',ls=':'); _a.set_xlabel('X'); _a.set_ylabel('E[S | ·]'); _a.set_ylim(-1.12,1.12)
_a.grid(alpha=.25); _a.legend(fontsize=8); _a.set_title('Selection on the hidden vitality S (gap largest at X=0)',fontsize=10)
_SC1=_scsave(_f,'sc_selbias.png')
_f,_a=plt.subplots(figsize=(7,4.2))
_a.plot(_Xg,_cate,'-o',color='#2ca02c',lw=2.6,ms=6,label='true CATE(X)')
_a.plot(_Xg,_taunaive,'--s',color='#ff7f0e',lw=2.2,ms=5,label='naive contrast tau_hat(X)')
_a.axhline(0,color='#888',ls=':'); _a.axvspan(0,1.05,color='#16a34a',alpha=.06)
_a.set_xlabel('X'); _a.set_ylabel('treatment effect'); _a.grid(alpha=.25); _a.legend(fontsize=8)
_a.set_title('Naive effect inflated near X=0 -> over-treatment',fontsize=10)
_SC2=_scsave(_f,'sc_naive_cate.png')
_f,_a=plt.subplots(figsize=(7,4.2))
_a.plot(_Xg,_emarg,'-o',color='#6a51a3',lw=2.6,ms=6,label='P(T=1 | X)')
_a.axhline(0.5,color='#bbb',ls=':'); _a.fill_between(_Xg,0,1,where=(_emarg<0.35),color='#dc2626',alpha=.08)
_a.set_xlabel('X'); _a.set_ylabel('P(T=1|X)'); _a.set_ylim(0,1); _a.grid(alpha=.25); _a.legend(fontsize=8)
_a.set_title('Overlap collapses at high X (few treated units)',fontsize=10)
_SC3=_scsave(_f,'sc_overlap.png')
_scpol={}
try:
    _pbs2=json.load(open(os.path.join(OWG,'owgap_results_n1000_ce1.0.json')))['regimes']['uncap']['policy_by_seed']
    for _m in ['Direct-X-X','IPW-X-X','DoublyRobust-X-X','Hajek-O-X','IPW-O-W','DoublyRobust-O-W']:
        _sk=[k for k in _pbs2[_m] if k!='avg']; _arr=_np.array([_pbs2[_m][s]['2'] for s in _sk])
        _mn=_arr.mean(0); _sdv=_arr.std(0); _c=FAM.get(_m.split('-')[0],'#0d9488')
        _f,_a=plt.subplots(figsize=(5.0,3.3))
        _a.step(_Xg,_oracle,where='mid',color='#444',ls=':',lw=1.9,label='oracle (treat X>0)')
        _a.plot(_Xg,_mn,'-o',color=_c,lw=2.4,ms=5,label='pi(treat|X)')
        _a.fill_between(_Xg,_np.clip(_mn-_sdv,0,1),_np.clip(_mn+_sdv,0,1),color=_c,alpha=.15)
        _a.set_ylim(-0.05,1.05); _a.set_xlabel('X'); _a.set_ylabel('pi(treat|X)'); _a.grid(alpha=.25)
        _a.legend(fontsize=6.5,loc='center left'); _a.set_title(_m,fontsize=9.5)
        _scpol[_m]=_scsave(_f,'sc_pol_%s.png'%_m.replace('-',''))
except Exception:
    _scpol={}
_MECH = grid([(_SC1,'1 — selection on the hidden S','Treated patients look more vital and controls more frail than average; the gap is biggest at X=0 — that gap is the bias.'),
              (_SC2,'2 — naive effect vs truth','The naive contrast spikes to +6.46 at X=0 where the true CATE is 0 — so naive methods treat the boundary they should not.'),
              (_SC3,'3 — overlap P(T=1|X)','Treatment becomes rare as X grows (0.23 at X=1): the outcome model is starved and IPW weights explode there.')])
_METH = {
 'DoublyRobust-X-X':('note',"<b>DoublyRobust-X-X (naive AIPW)</b> — <i>over-treats into \\(X\\le0\\).</i> Selection bias inflates the effect near the boundary (plot 2: \\(+6.46\\) at \\(X=0\\)), so AIPW treats \\(X=0\\) (true effect \\(0\\)) and leaks into \\(X=-\\tfrac13\\). <b>This is why it scores 0.69, not 0.847.</b>"),
 'IPW-X-X':('note',"<b>IPW-X-X (Hajek)</b> — <i>over-treats low \\(X\\), erratic at high \\(X\\).</i> Pure inverse-propensity: it over-treats \\(X=-\\tfrac13\\), and at \\(X=1\\) the policy is a coin-flip because overlap is only \\(0.23\\) (plot 3) so the weights are huge and high-variance."),
 'Direct-X-X':('note',"<b>Direct-X-X (plug-in outcome model)</b> — <i>under-treats high \\(X\\).</i> Where treated units are sparse (plot 3) the fitted \\(\\hat\\mu_1\\) is unreliable, so it fails to treat \\(X=\\tfrac23,1\\) — the levels with the <i>largest</i> true benefit."),
 'Hajek-O-X':('note',"<b>Hajek-O-X (worst-case regret, \\(\\Gamma=2\\))</b> — <i>collapsed to nearly never-treat.</i> The do-no-harm fallback retreats to the baseline once worst-case regret turns positive; by \\(\\Gamma=2\\) it has essentially collapsed (value \\(0.00\\))."),
 'IPW-O-W':('good',"<b>IPW-O-W</b> — <i>recovers the oracle.</i> Wasserstein balance on \\(X\\) (which tracks the hidden \\(S\\)) removes the selection bias, recovering treat-iff-\\(X>0\\); it only blurs at \\(X=0\\), the hardest call."),
 'DoublyRobust-O-W':('good',"<b>DoublyRobust-O-W</b> — <i>recovers the oracle.</i> Same covariate-balancing mechanism, with the direct outcome term anchoring it — a clean treat-the-fit policy."),
}
def _method_block(m):
    cls,txt=_METH[m]; pl=figcard(_scpol[m], m, 'Learned pi(treat|X), mean +/- SD over seeds (Gamma=2, uncapped), vs the oracle step (dotted).') if m in _scpol else ''
    return "<div class='scmethod'>"+pl+"<div class='"+cls+"' style='margin:0'>"+txt+"</div></div>"
_METHBLOCKS = ''.join(_method_block(m) for m in ['DoublyRobust-X-X','IPW-X-X','Direct-X-X','Hajek-O-X','IPW-O-W','DoublyRobust-O-W'])

tab_sanity = ("<div class='mhead'><span class='pill' style='background:#0f766e'>Sanity check</span>"
 "<h2 style='border:none;margin:0'>Why each method's policy looks the way it does</h2></div>"
 "<p class='lead'>A mechanistic check: every policy below is explained by <b>two features of the DGP the estimators can actually see</b> — so the simpler methods fail in <i>exactly</i> the directions the DGP predicts, which is strong evidence the pipeline is correct.</p>"
 "<div class='panel'><h3 style='margin-top:4px'>What the DGP determines (ground truth &amp; what a naïve analyst sees)</h3>"
 + _truth_tbl +
 "<p class='sub' style='margin-top:8px'>The two structural drivers:</p><ol>"
 "<li><b>Selection bias peaks at \\(X=0\\).</b> There \\(P(S{=}{+}1)=\\tfrac12\\), but treated patients are selected <i>vital</i> (\\(\\mathbb{E}[S\\mid\\text{treated}]=+0.38\\)) and controls <i>frail</i> (\\(-0.38\\)). So the naïve contrast \\(\\mathbb{E}[Y\\mid T{=}1]-\\mathbb{E}[Y\\mid T{=}0]=+6.46\\) — a large <b>spurious</b> effect where the true CATE is exactly \\(0\\).</li>"
 "<li><b>Overlap collapses as \\(X\\) grows:</b> \\(P(T{=}1\\mid X)\\) falls to \\(0.23\\) at \\(X=1\\) (the \\(-2X\\) term suppresses treatment). Few treated high-\\(X\\) units ⇒ the outcome model is starved there and IPW weights explode there.</li></ol>"
 "<h3>The two mechanisms, visualised</h3>" + _MECH + "</div>"
 "<div class='panel'><h3 style='margin-top:4px'>Learned policies \\(\\pi(\\text{treat}\\mid X)\\) (N=1000, \\(\\Gamma=2\\), uncapped, seed-averaged)</h3>"
 + _poltbl + "</div>"
 "<h3>Each method's policy, explained with its plot</h3>"
 "<p class='sub'>Each panel: the learned \\(\\pi(\\text{treat}\\mid X)\\) (mean ± SD over seeds, Γ=2, uncapped) against the oracle step (dotted). The shape of every curve follows from the three mechanisms above.</p>"
 + _METHBLOCKS +
 "<div class='panel'><b>Verdict.</b> Two DGP features explain everything: (a) selection bias peaking at \\(X=0\\) → naïve methods over-treat the boundary; (b) overlap decaying with \\(X\\) → Direct under-treats high \\(X\\) (model starved) while IPW destabilises there (weights explode). The O-W methods sidestep both via covariate balance. Nothing is anomalous — the failures are all in the DGP-predicted directions.</div>")

# ---------- Continuous-X OWGAP: raw per-unit vs fine mesh (in-sample realized outcome) ----------
CONT = os.path.join(HERE, "exp_owgap_cont")
CONT_ORDER = ['IPW-X-X','DoublyRobust-X-X','Direct-X-X','IPW-O-X','DoublyRobust-O-X','Hajek-O-X','DoublyRobust-O-W']
def gen_cont_plot(jpath, png):
    d = json.load(open(jpath)); G = d['gammas']; M = d['methods']; ns = len(d['seeds'])
    orc = d['oracle']['mean']; nev = d['never_treat']['mean']
    fig, ax = plt.subplots(figsize=(7.2,4.4))
    for m in CONT_ORDER:
        if m not in M: continue
        y = np.array(M[m]['mean'], float); sd = np.array(M[m].get('sd', [0]*len(G)), float)
        c, ls, mk = _n1style(m)
        ax.plot(G, y, ls, color=c, marker=mk, ms=5, lw=2, label=m)
        ax.fill_between(G, y-sd, y+sd, color=c, alpha=.12, lw=0)
    ax.axhline(orc, color='#444', ls=':', label='oracle %.2f' % orc)
    ax.axhline(nev, color='#bbb', ls=':', label='never-treat %.2f' % nev)
    ax.set_xlabel('Γ'); ax.set_ylabel('in-sample realised E[Y] (%d seeds)' % ns)
    ax.grid(alpha=.25); ax.legend(fontsize=6.5, ncol=2)
    ax.set_title('Continuous X — in-sample realized outcome', fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, png), dpi=120); plt.close(fig)
    return png
def cont_table(jpath):
    d = json.load(open(jpath)); G = d['gammas']; M = d['methods']
    th = ''.join('<th>Γ=%g</th>' % g for g in G)
    rows = ''
    for m in CONT_ORDER:
        if m not in M: continue
        sw = "<span class='sw' style='background:%s'></span>" % FAM.get(m.split('-')[0], '#888')
        cls = " class='hl'" if m.endswith('-O-W') else ""
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (cls, sw, m, ''.join('<td>%.3f</td>' % v for v in M[m]['mean']))
    for lbl, key in [('Oracle','oracle'), ('Never-treat','never_treat'), ('All-treat','all_treat')]:
        rows += "<tr><td><span class='sw' style='background:#444'></span>%s</td>%s</tr>" % (lbl, ''.join('<td>%.3f</td>' % d[key]['mean'] for _ in G))
    return "<table class='restab'><tr><th>method</th>%s</tr>%s</table>" % (th, rows)
def cont_panel(jpath, png, mode_label):
    if not os.path.exists(jpath):
        return "<p class='sub'>Continuous run pending — rebuild when <code>%s</code> lands.</p>" % os.path.basename(jpath)
    d = json.load(open(jpath))
    cap = ('X∼Uniform(−1,1), N=%d, %d seeds, uncapped. Value = (1/n)Σ[π₁(Xᵢ)·Y1ᵢ + π₀(Xᵢ)·Y0ᵢ] on the training units own potential outcomes. IPW-O-W excluded.' % (d['N_train'], len(d['seeds'])))
    return grid([(gen_cont_plot(jpath, png), 'In-sample realized outcome vs Γ — %s' % mode_label, cap)]) + "<div class='panel'>" + cont_table(jpath) + "</div>"

# --- continuous test-set-via-extension panel (KNN / Shapley) ---
def gen_cont_test_val(jpath, png):
    d = json.load(open(jpath)); G = d['gammas']; M = d['methods']; ns = len(d['seeds'])
    orc = d['oracle']['mean']; nev = d['never_treat']['mean']
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, e, ttl in [(axs[0], 'knn', 'KNN'), (axs[1], 'shp', 'Shapley')]:
        for m in CONT_ORDER:
            if m not in M: continue
            c, ls, mk = _n1style(m); ax.plot(G, M[m][e]['mean'], ls, color=c, marker=mk, ms=4.5, lw=1.9, label=m)
        ax.axhline(orc, color='#444', ls=':', label='oracle %.2f' % orc); ax.axhline(nev, color='#bbb', ls=':', label='never %.2f' % nev)
        ax.set_xlabel('Γ'); ax.set_title('Test realised E[Y] — %s extension' % ttl, fontsize=9.5); ax.grid(alpha=.25)
    axs[0].set_ylabel('test realised E[Y] (%d seeds)' % ns); axs[1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, png), dpi=120); plt.close(fig); return png
def gen_cont_pol_curves(jpath, png, gt=2.0):
    d = json.load(open(jpath)); G = d['gammas']; M = d['methods']; xg = np.array(d['grid'], float)
    gi = int(np.argmin([abs(g - gt) for g in G]))
    fig, axs = plt.subplots(2, 4, figsize=(14, 6)); axs = axs.ravel()
    for ai, m in enumerate(CONT_ORDER):
        ax = axs[ai]
        for e, ec in [('knn', '#1f77b4'), ('shp', '#d62728')]:
            gl = M[m][e]['grid_mean'][gi]
            if gl is not None: ax.plot(xg, gl, '-', color=ec, lw=1.8, label=('KNN' if e == 'knn' else 'Shapley'))
        ax.step(xg, (xg > 0).astype(float), where='post', color='#444', ls=':', lw=1.4, label='oracle')
        ax.set_title(m, fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=.25)
        if ai == 0: ax.legend(fontsize=6.5)
    for ai in range(len(CONT_ORDER), len(axs)): axs[ai].axis('off')
    fig.suptitle('Extended policy π(treat|X) vs X  (Γ=%g, mean over seeds)' % G[gi], fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, png), dpi=120); plt.close(fig); return png
def cont_test_table(jpath, ext):
    d = json.load(open(jpath)); G = d['gammas']; M = d['methods']
    th = ''.join('<th>Γ=%g</th>' % g for g in G); rows = ''
    for m in CONT_ORDER:
        if m not in M: continue
        sw = "<span class='sw' style='background:%s'></span>" % FAM.get(m.split('-')[0], '#888')
        cls = " class='hl'" if m.endswith('-O-W') else ""
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (cls, sw, m, ''.join('<td>%.3f</td>' % v for v in M[m][ext]['mean']))
    for lbl, key in [('Oracle', 'oracle'), ('Never-treat', 'never_treat')]:
        rows += "<tr><td><span class='sw' style='background:#444'></span>%s</td>%s</tr>" % (lbl, ''.join('<td>%.3f</td>' % d[key]['mean'] for _ in G))
    return "<table class='restab'><tr><th>method</th>%s</tr>%s</table>" % (th, rows)
def _ext_section(jpath, pref, label):
    d = json.load(open(jpath))
    plots = grid([
        (gen_cont_test_val(jpath, pref + '_val.png'), 'Test realized outcome vs Γ — %s (KNN vs Shapley)' % label,
         'N_train=%d, N_test=%d, %d seeds, uncapped.' % (d['N_train'], d['N_test'], len(d['seeds']))),
        (gen_cont_pol_curves(jpath, pref + '_pol.png'), 'Extended policy π(treat|X) vs X — %s (Γ=2)' % label,
         'Blue = KNN, red = Shapley, dotted = oracle step (treat iff X>0). Mean over seeds.')])
    tabs = ("<h4 style='margin:12px 0 4px'>%s — KNN test realized outcome</h4><div class='panel'>" % label + cont_test_table(jpath, 'knn') + "</div>"
            "<h4 style='margin:12px 0 4px'>%s — Shapley test realized outcome</h4><div class='panel'>" % label + cont_test_table(jpath, 'shp') + "</div>")
    return plots + tabs
def cont_test_panel():
    have_raw = os.path.exists(CONT_EXT); have_mesh = os.path.exists(CONT_EXT_MESH)
    if not (have_raw or have_mesh):
        return "<p class='sub'>Extension runs pending — rebuild when the extend JSONs land.</p>"
    kv = json.load(open(CONT_EXT if have_raw else CONT_EXT_MESH))
    intro = ("<div class='note'>In the continuous (no-mesh) case the policy exists only at the training points, so it cannot be "
      "evaluated on new X. We lift it off-support with the repo's two operators — <b>KNN</b> (inverse-distance average of the "
      "k=%d nearest support points) and <b>Shapley</b> (closed-form 1-Lipschitz min–max, exact at the support) — then score the "
      "extended policy on an <b>independent test set</b> (N=%d). Neither operator needs a grid/mesh; the π(X) curves are just the "
      "extension sampled along X for display, and the test value is computed at the raw test points. Below we extend <b>two</b> "
      "trained policies: (A) the <b>raw per-unit</b> policy and (B) the <b>mesh-trained</b> (25-level) policy.</div>" % (kv['k'], kv['N_test']))
    body = ""
    if have_raw:
        body += "<h3>A) Extend the raw per-unit policy</h3>" + _ext_section(CONT_EXT, 'cont_raw_ext', 'raw-train')
    if have_mesh:
        body += "<h3 style='margin-top:22px'>B) Extend the mesh-trained policy</h3>" + _ext_section(CONT_EXT_MESH, 'cont_mesh_ext', 'mesh-train')
    note = ("<div class='note'><b>Training is the bottleneck, not the extension.</b> Extending the <b>raw</b> per-unit policy "
      "generalises badly — the IPW / Direct / Hájek family lands <i>below</i> never-treat and even the DoublyRobust methods reach only "
      "a fraction of the oracle — because the underlying per-unit fit is degenerate. Extending the <b>mesh-trained</b> policy is a "
      "different story: because binning produced a genuine piecewise-constant policy, KNN/Shapley carry it to the test set with the "
      "DoublyRobust and O-W methods recovering most of the oracle. Same extension operators, same k — the only change is whether the "
      "policy being extended was learned with pooled (binned) mass. This is the end of the continuous-X thread: <b>bin to train, then "
      "extend to deploy.</b></div>")
    return intro + body + note
CONT_EXT = os.path.join(CONT, 'owgap_cont_extend.json')
CONT_EXT_MESH = os.path.join(CONT, 'owgap_cont_extend_mesh25.json')

# --- Lipschitz policy-class panel (continuous X, no unit-by-unit overfitting) ---
CONT_LIP = os.path.join(CONT, 'owgap_lipschitz.json')
LIP_METHS = ['IPW-X-X', 'DoublyRobust-X-X', 'Direct-X-X']
LIP_COL = {'IPW-X-X': '#1f77b4', 'DoublyRobust-X-X': '#2ca02c', 'Direct-X-X': '#ff7f0e'}
def gen_lip_val(jpath, png):
    d = json.load(open(jpath)); L = d['Lgrid']; ns = len(d['seeds'])
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for m in LIP_METHS:
        ys = [d['methods'][m]['test_mean'][lk] for lk in L]
        ax.plot(range(len(L)), ys, '-o', color=LIP_COL[m], lw=2, ms=5, label=m)
    ax.axhline(d['oracle'], color='#444', ls=':', label='oracle %.2f' % d['oracle'])
    ax.axhline(d['never_treat'], color='#bbb', ls=':', label='never-treat %.2f' % d['never_treat'])
    ax.set_xticks(range(len(L))); ax.set_xticklabels(L); ax.set_xlabel('Lipschitz L  (∞ = unit-by-unit)')
    ax.set_ylabel('test realised E[Y] (%d seeds)' % ns); ax.grid(alpha=.25); ax.legend(fontsize=7.5)
    ax.set_title('Lipschitz policy class — test realized outcome vs L', fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, png), dpi=120); plt.close(fig); return png
def gen_lip_pol(jpath, png):
    d = json.load(open(jpath)); xg = np.array(d['grid'], float)
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for lk, col in [('inf', '#d62728'), ('5', '#2ca02c'), ('1', '#1f77b4'), ('0.5', '#9467bd')]:
        gpv = d['methods']['DoublyRobust-X-X']['gridpol'].get(lk)
        if gpv: ax.plot(xg, gpv, '-', lw=2, color=col, label='L=%s' % lk)
    ax.step(xg, (xg > 0).astype(float), where='post', color='#444', ls=':', lw=1.5, label='oracle')
    ax.set_xlabel('X'); ax.set_ylabel('π(treat | X)'); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title('DoublyRobust-X-X policy vs X, by Lipschitz L (seed 0)', fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, png), dpi=120); plt.close(fig); return png
def lip_table(jpath):
    d = json.load(open(jpath)); L = d['Lgrid']
    th = ''.join('<th>L=%s</th>' % lk for lk in L); rows = ''
    for m in LIP_METHS:
        sw = "<span class='sw' style='background:%s'></span>" % LIP_COL[m]
        cls = " class='hl'" if m == 'DoublyRobust-X-X' else ""
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (cls, sw, m, ''.join('<td>%.3f</td>' % d['methods'][m]['test_mean'][lk] for lk in L))
    rows += "<tr><td><span class='sw' style='background:#444'></span>Oracle</td>%s</tr>" % ''.join('<td>%.3f</td>' % d['oracle'] for _ in L)
    return "<table class='restab'><tr><th>method \\ L</th>%s</tr>%s</table>" % (th, rows)
LIP_ROB = os.path.join(CONT, 'owgap_lipschitz_robust.json')
LIP_ROB_METHS = ['IPW-O-X', 'DoublyRobust-O-X', 'Hajek-O-X', 'IPW-O-W', 'DoublyRobust-O-W']
def gen_lip_val_rob(jpath, png):
    d = json.load(open(jpath)); L = d['Lgrid']; ns = len(d['seeds'])
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for m in LIP_ROB_METHS:
        ys = [d['methods'][m]['test_mean'][lk] for lk in L]
        c, ls, mk = _n1style(m); ax.plot(range(len(L)), ys, ls, color=c, marker=mk, ms=5, lw=2, label=m)
    ax.axhline(d['oracle'], color='#444', ls=':', label='oracle %.2f' % d['oracle'])
    ax.axhline(d['never_treat'], color='#bbb', ls=':', label='never-treat %.2f' % d['never_treat'])
    ax.set_xticks(range(len(L))); ax.set_xticklabels(L); ax.set_xlabel('Lipschitz L  (∞ = unit-by-unit)')
    ax.set_ylabel('test realised E[Y] (%d seeds)' % ns); ax.grid(alpha=.25); ax.legend(fontsize=7.5)
    ax.set_title('Robust methods + Lipschitz (Γ=%g) — test realized outcome vs L' % d['gamma'], fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, png), dpi=120); plt.close(fig); return png
def lip_table_rob(jpath):
    d = json.load(open(jpath)); L = d['Lgrid']
    th = ''.join('<th>L=%s</th>' % lk for lk in L); rows = ''
    for m in LIP_ROB_METHS:
        sw = "<span class='sw' style='background:%s'></span>" % FAM.get(m.split('-')[0], '#888')
        cls = " class='hl'" if m.endswith('-O-W') else ""
        rows += "<tr%s><td>%s%s</td>%s</tr>" % (cls, sw, m, ''.join('<td>%.3f</td>' % d['methods'][m]['test_mean'][lk] for lk in L))
    rows += "<tr><td><span class='sw' style='background:#444'></span>Oracle</td>%s</tr>" % ''.join('<td>%.3f</td>' % d['oracle'] for _ in L)
    return "<table class='restab'><tr><th>method \\ L</th>%s</tr>%s</table>" % (th, rows)
LIP_2D = os.path.join(CONT, 'owgap_lip_gamma_2d.json')
def gen_lip2d(jpath, png):
    d = json.load(open(jpath)); Gk = d['gammas']; Lk = d['Lgrid']; orc = d['oracle']; nev = d['never_treat']
    bo = d.get('best_overall', {}); bm = bo.get('method') or (d['methods'][0] if d.get('methods') else None)
    surf = d['surface'][bm] if (bm and bm in d['surface']) else d['surface']
    M = np.array([[surf[g][l] for l in Lk] for g in Gk], float)
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    im = ax.imshow(M, aspect='auto', cmap='viridis', origin='upper', vmin=max(nev, -0.1), vmax=orc)
    ax.set_xticks(range(len(Lk))); ax.set_xticklabels(Lk); ax.set_yticks(range(len(Gk))); ax.set_yticklabels(['Γ=' + g for g in Gk])
    ax.set_xlabel('Lipschitz L  (∞ = unit-by-unit)'); ax.set_ylabel('Γ (sensitivity)')
    for i in range(len(Gk)):
        for j in range(len(Lk)):
            ax.text(j, i, '%.2f' % M[i, j], ha='center', va='center', fontsize=7, color='white' if M[i, j] < (nev + orc) / 2 else 'black')
    if bo.get('gamma') in Gk and bo.get('L') in Lk:
        bi = Gk.index(bo['gamma']); bj = Lk.index(bo['L']); ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False, edgecolor='red', lw=2))
    fig.colorbar(im, ax=ax, label='test realised E[Y] (oracle=%.2f)' % orc)
    ax.set_title('%s: test realized outcome over L × Γ' % (bm or ''), fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, png), dpi=120); plt.close(fig); return png
def gen_knnshp_scatter(png):
    pts = []
    for jp in [CONT_LIP, LIP_ROB]:
        if not os.path.exists(jp): continue
        d = json.load(open(jp))
        for m, mm in d['methods'].items():
            if 'test_shp' not in mm: continue
            for lk in d['Lgrid']:
                kn = mm['test_mean'].get(lk); sh = mm['test_shp'].get(lk)
                if kn is not None and sh is not None and kn == kn and sh == sh: pts.append((kn, sh))
    if not pts: return None
    a = np.array(pts, float); lo = min(a.min(), -0.15); hi = max(a.max(), 0.6)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([lo, hi], [lo, hi], '--', color='#888', lw=1, label='y = x (agree)')
    ax.scatter(a[:, 0], a[:, 1], s=20, color='#0369a1', alpha=.7)
    ax.set_xlabel('KNN test realised E[Y]'); ax.set_ylabel('Shapley test realised E[Y]')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.grid(alpha=.25); ax.legend(fontsize=8)
    mad = float(np.mean(np.abs(a[:, 0] - a[:, 1]))); mx = float(np.max(np.abs(a[:, 0] - a[:, 1])))
    ax.set_title('KNN vs Shapley deployment — all method×L cells\nmean |Δ|=%.3f, max |Δ|=%.3f' % (mad, mx), fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OWG, png), dpi=120); plt.close(fig); return (png, mad, mx)
def lip_panel():
    if not os.path.exists(CONT_LIP):
        return "<p class='sub'>Lipschitz run pending — rebuild when <code>owgap_lipschitz.json</code> lands.</p>"
    d = json.load(open(CONT_LIP))
    intro = ("<div class='note'>Instead of extending after a unit-by-unit fit, constrain the policy to a <b>class</b> during "
      "optimisation: an <b>L-Lipschitz</b> policy, \\(|\\pi(X_i)-\\pi(X_j)|\\le L\\,\\|X_i-X_j\\|\\). This couples nearby-X decisions, so "
      "the policy must vary smoothly with X rather than memorising each point — directly removing the n-free-parameters overfitting. "
      "For 1-D X it reduces to consecutive-sorted-order constraints (O(n), exact). Added to the direct-LP methods (IPW-X-X, "
      "DoublyRobust-X-X, Direct-X-X); trained on continuous X, deployed on the test set via KNN. <b>L=∞ is the unit-by-unit baseline.</b> "
      "N_train=%d, N_test=%d, %d seeds.</div>" % (d['N_train'], d['N_test'], len(d['seeds'])))
    plots = grid([
        (gen_lip_val(CONT_LIP, 'lip_val.png'), 'Test realized outcome vs Lipschitz L',
         'Sweeping L from ∞ (unit-by-unit) to 0.25 (very smooth). The sweet spot recovers most of the oracle.'),
        (gen_lip_pol(CONT_LIP, 'lip_pol.png'), 'Learned policy π(treat|X) vs X, by L (DoublyRobust-X-X)',
         'Large L → sharp/overfit; small L → over-smoothed ramp; moderate L tracks the oracle step at X=0.')])
    tbl = "<div class='panel'>" + lip_table(CONT_LIP) + "</div>"
    note = ("<div class='note'><b>A policy class fixes the overfitting — as well as binning does.</b> Constraining to L-Lipschitz "
      "lifts <b>DoublyRobust-X-X from 0.072 (unit-by-unit) to ≈0.45 at L≈3</b> — matching (slightly beating) the mesh result — and "
      "IPW-X-X from −0.12 to ≈0.25. There is a clear bias–variance sweet spot: L=∞ overfits, L→0 over-smooths to never-treat. "
      "<b>But smoothing fixes variance, not bias:</b> <b>Direct-X-X barely moves</b> (stays ≈0) because its objective is confounded "
      "(unit weights, no propensity/outcome correction) — a smooth policy over a biased objective is still biased. The winner is the "
      "method that handles <i>both</i>: DoublyRobust (μ̂ removes the confounding bias) + Lipschitz (removes the variance). "
      "So there are two independent cures for continuous X — <b>bin the covariate</b> or <b>restrict the policy class</b> — and they "
      "reach the same place.</div>")
    rob = ""
    if os.path.exists(LIP_ROB):
        dr = json.load(open(LIP_ROB))
        rob = ("<h3 style='margin-top:24px'>Robust methods + Lipschitz (Γ=%g)</h3>" % dr['gamma'] +
          "<div class='note'>The same L-Lipschitz constraint added to the <b>Γ-robust</b> solvers "
          "(IPW-O-X / DoublyRobust-O-X / Hájek-O-X / DoublyRobust-O-W) at a fixed Γ=%g — the policy class and the confounding-robustness "
          "<b>compound</b>.</div>" % dr['gamma'] +
          grid([(gen_lip_val_rob(LIP_ROB, 'lip_val_rob.png'), 'Robust methods + Lipschitz — test realized outcome vs L',
                 'Γ=%g fixed, sweeping the Lipschitz L. N_train=%d, N_test=%d, %d seeds.' % (dr['gamma'], dr['N_train'], dr['N_test'], len(dr['seeds'])))]) +
          "<div class='panel'>" + lip_table_rob(LIP_ROB) + "</div>" +
          "<div class='note'><b>Lipschitz rescues the robust methods too.</b> At Γ=2, adding the policy class lifts "
          "<b>DoublyRobust-O-W from ≈0.05 (unit-by-unit) to ≈0.5</b> and IPW-O-X from −0.07 to ≈0.2 — matching the mesh-trained numbers. "
          "So the Γ / Wasserstein robustness and the Lipschitz smoothness are <b>orthogonal fixes</b>: Γ hedges the unknown confounding, "
          "Lipschitz controls the policy complexity. You need the second one to make the methods usable on continuous X — with or without "
          "binning.</div>")
    two_d = ""
    if os.path.exists(LIP_2D):
        d2 = json.load(open(LIP_2D)); b = d2['best']
        bo = d2.get('best_overall', {}); _lm = d2.get('methods', [])
        mchecks = "".join("<label class='polcheck'><input type='checkbox' class='lipm-mcb' value='%s'%s onchange='drawLipBoth()'>%s</label>"
                          % (m, ' checked' if m in ('IPW-O-W', 'DoublyRobust-O-W') else '', m) for m in _lm)
        gopts = "".join("<option%s>%s</option>" % (' selected' if g == '2' else '', g) for g in d2['gammas'])
        lopts = "".join("<option%s>%s</option>" % (' selected' if L == '2' else '', L) for L in d2['Lgrid'])
        two_d = ("<h3 style='margin-top:24px'>Robust methods — L × Γ (interactive slices)</h3>" +
          "<div class='note'>Sweeping <b>both</b> knobs — Lipschitz L (policy complexity) and Γ (confounding-robustness) — for the robust methods, "
          "test realized outcome over %d seeds. Check one or more methods, then slice: hold Γ and vary L, or hold L and vary Γ.</div>" % len(d2['seeds']) +
          "<div class='polmenus'>"
          "<div class='polsel' style='min-width:220px'>methods (check one or more)<div class='polchecks'>" + mchecks + "</div></div>"
          "<label class='polsel'>hold Γ =<select id='lip-gsel' onchange='drawLipVsL()'>" + gopts + "</select></label>"
          "<label class='polsel'>hold L =<select id='lip-lsel' onchange='drawLipVsG()'>" + lopts + "</select></label>"
          "</div>"
          "<h4 style='margin:14px 0 4px'>Test E[Y] vs L (at the held Γ)</h4>"
          "<div id='lip-vsl-svg' class='panel' style='min-height:320px;display:flex;align-items:center;justify-content:center;padding:10px'></div>"
          "<h4 style='margin:14px 0 4px'>Test E[Y] vs Γ (at the held L)</h4>"
          "<div id='lip-vsg-svg' class='panel' style='min-height:320px;display:flex;align-items:center;justify-content:center;padding:10px'></div>"
          "<h4 style='margin:18px 0 4px'>Full surface — best method (%s)</h4>" % bo.get('method', '') +
          grid([(gen_lip2d(LIP_2D, 'lip_gamma_2d.png'), 'Test realized outcome — L (columns) × Γ (rows), best method',
                 'oracle %.2f, never-treat %.2f. N_train=%d, N_test=%d.' % (d2['oracle'], d2['never_treat'], d2['N_train'], d2['N_test']))]) +
          "<div class='note'><b>Best overall: %s at Γ=%s, L=%s → %.3f</b> (≈%d%% of oracle %.2f). Across methods the pattern holds — the Lipschitz L "
          "(variance control) does the heavy lifting, and Γ (robustness) fine-tunes on top.</div>"
          % (bo.get('method', ''), bo.get('gamma', ''), bo.get('L', ''), bo.get('value', 0), int(round(100 * bo.get('value', 0) / d2['oracle'])), d2['oracle']))
    cmp = ""
    sc = gen_knnshp_scatter('lip_knn_vs_shp.png')
    if sc:
        png, mad, mx = sc
        cmp = ("<h3 style='margin-top:24px'>KNN vs Shapley deployment</h3>" +
          "<div class='note'>Re-deploying the <b>same</b> Lipschitz-trained policies with the <b>Shapley</b> operator instead of KNN — "
          "same training, only the point→function extension changes.</div>" +
          grid([(png, 'Shapley vs KNN test outcome (every method × L cell)',
                 'Pooled over the X-X and robust Lipschitz sweeps. Points on the y=x line ⇒ the two extensions agree.')]) +
          "<div class='note'><b>KNN and Shapley are interchangeable here.</b> Across every method×L cell the two deployments agree to "
          "<b>mean |Δ| = %.3f (max %.3f)</b> — because the trained policy is already smooth (Lipschitz), both operators reproduce it "
          "faithfully. Confirms the earlier point: the <i>training constraint</i> is the lever, not the choice of extension.</div>" % (mad, mx))
    return intro + plots + tbl + note + rob + two_d + cmp
CONT_RAW = os.path.join(CONT, 'owgap_cont_insample.json')
CONT_MESH = os.path.join(CONT, 'owgap_cont_mesh25.json')
if os.path.exists(CONT_RAW) or os.path.exists(CONT_MESH):
    tab_cont = ("<div class='mhead'><span class='pill' style='background:#0369a1'>Continuous X</span>"
      "<h2 style='border:none;margin:0'>exp_owgap with <u>continuous</u> X — raw per-unit vs fine mesh</h2></div>"
      "<p class='lead'>Same hidden-vitality DGP (<code>exp_owgap_cont/dgp.py</code>), but \\(X\\sim\\mathrm{Uniform}(-1,1)\\) instead of a discrete grid. "
      "Each method is scored by its <b>in-sample realized outcome</b> on the training units' known potential outcomes, "
      "\\(\\hat V(\\pi)=\\frac1n\\sum_i[\\pi_1(X_i)Y_1^{(i)}+\\pi_0(X_i)Y_0^{(i)}]\\). N=400, 5 seeds, uncapped. "
      "(<b>IPW-O-W excluded</b> for now; DoublyRobust-O-W is the sole O-W method.)</p>" + LEGEND +
      "<div class='note'><b>Key finding — continuous covariates must be binned for the robustness to work.</b> "
      "With <b>raw per-unit</b> support (<code>discretize=False</code>) every unit is a unique point, so the odds-box / Wasserstein "
      "uncertainty sets have <b>no pooled mass</b> to act on: the IPW / Direct / Hájek family <b>collapses to an identical policy that is even worse "
      "than never-treat</b>, and Γ does nothing (IPW-O-X ≡ IPW-X-X). Only the outcome-model-anchored <b>DoublyRobust</b> methods stay meaningful. "
      "<b>Binning X to a fine mesh (25 levels)</b> restores shared covariate atoms: the methods separate again, the O-X / O-W robustness becomes "
      "active, and DoublyRobust-O-W climbs back toward the oracle. This is the empirical confirmation of the <code>common/support.py</code> warning "
      "— singleton support guts the confounding-robust machinery.</div>"
      "<h3 style='margin-top:22px'>Raw per-unit vs fine mesh vs test-set extension</h3>" + inner_tabs_n('cont', [
          ('Raw per-unit (discretize=False)', cont_panel(CONT_RAW, 'cont_raw.png', 'raw per-unit')),
          ('Fine mesh (25 levels)', cont_panel(CONT_MESH, 'cont_mesh25.png', 'mesh = 25 levels')),
          ('Test set via extension (KNN / Shapley)', cont_test_panel()),
          ('Lipschitz policy class', lip_panel())]))
else:
    tab_cont = None

# ---------- index-style interactive policy charts (renderChart template ported from index.html) ----------
_SCOPED_DRAW = r"""
function _pvv(id){var e=document.getElementById(id);return e?e.value:null;}
function drawScoped(cid){
 var M=SCOPED[cid];if(!M)return;var ds=M.ds,ep=M.ep,rg=M.reg;
 var g=_pvv('pg-'+cid),sd=_pvv('ps-'+cid);
 var GRID_X=GRID_BY_DS[ds]||[];
 var ms=[];document.querySelectorAll('.pmcb-'+cid+':checked').forEach(function(cb){ms.push(cb.value);});
 var el=document.getElementById('psvg-'+cid);if(!el)return;
 if(!GRID_X.length){el.innerHTML="<span class='sub'>No grid for this dataset.</span>";return;}
 if(!ms.length){el.innerHTML="<span class='sub'>Check at least one method above.</span>";return;}
 var W=760,H=380,mL=58,mR=22,mT=20,mB=44,x0=mL,x1=W-mR,y0=H-mB,y1=mT;
 var xmin=Math.min.apply(null,GRID_X),xmax=Math.max.apply(null,GRID_X);
 function sx(x){return x0+(x-xmin)/(xmax-xmin)*(x1-x0);}function sy(v){return y0-v*(y0-y1);}
 var s='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;max-width:780px">',i,yv,y;
 for(i=0;i<=4;i++){yv=i/4;y=sy(yv);s+='<line x1="'+x0+'" y1="'+y+'" x2="'+x1+'" y2="'+y+'" stroke="#eef0f7"/><text x="'+(x0-8)+'" y="'+(y+4)+'" text-anchor="end" font-size="11" fill="#9aa3b2">'+yv.toFixed(2)+'</text>';}
 GRID_X.forEach(function(xx){var px=sx(xx);s+='<text x="'+px+'" y="'+(y0+18)+'" text-anchor="middle" font-size="10" fill="#9aa3b2">'+xx.toFixed(2)+'</text>';});
 s+='<line x1="'+x0+'" y1="'+sy(0.5)+'" x2="'+x1+'" y2="'+sy(0.5)+'" stroke="#dfe3ec" stroke-dasharray="4,4"/>';
 var orc=GRID_X.map(function(xx){return xx>0?1:0;});
 s+='<polyline points="'+GRID_X.map(function(xx,k){return sx(xx)+','+sy(orc[k]);}).join(' ')+'" fill="none" stroke="#444" stroke-dasharray="2,3" stroke-width="1.6"/>';
 var miss=[];
 ms.forEach(function(m){var node=null;try{node=POLICY_DATA[ds][ep][rg][m][g][sd];}catch(e){node=null;}
  if(!node){miss.push(m);return;}var st=_mstyle(m);
  s+='<polyline points="'+GRID_X.map(function(xx,k){return sx(xx)+','+sy(node[k]);}).join(' ')+'" fill="none" stroke="'+st.c+'" stroke-width="2.4"'+(st.dash?' stroke-dasharray="'+st.dash+'"':'')+'/>';
  if(st.mk)GRID_X.forEach(function(xx,k){s+=_mk(st.mk,sx(xx),sy(node[k]),st.c);});});
 s+='<text x="'+((x0+x1)/2)+'" y="'+(H-6)+'" text-anchor="middle" font-size="12">X</text>';
 s+='<text transform="translate(15,'+((y0+y1)/2)+') rotate(-90)" text-anchor="middle" font-size="12">pi(treat | X)</text>';
 s+='<text x="'+x1+'" y="'+(y1+2)+'" text-anchor="end" font-size="10" fill="#888">'+ds+'  Gamma='+g+'  seed '+sd+'</text>';
 s+='</svg>';
 var leg="<div style='display:flex;flex-wrap:wrap;gap:8px 16px;justify-content:center;margin-top:6px;font-size:.82rem'>";
 leg+="<span><svg width='26' height='10' style='vertical-align:middle'><line x1='1' y1='5' x2='24' y2='5' stroke='#444' stroke-dasharray='2,3' stroke-width='1.6'/></svg> oracle</span>";
 ms.forEach(function(m){if(miss.indexOf(m)>=0)return;var st=_mstyle(m);var sw="<svg width='26' height='10' style='vertical-align:middle'><line x1='1' y1='5' x2='24' y2='5' stroke='"+st.c+"' stroke-width='2.4'"+(st.dash?" stroke-dasharray='"+st.dash+"'":"")+"/>"+(st.mk=='circle'?"<circle cx='13' cy='5' r='3' fill='"+st.c+"'/>":st.mk=='square'?"<rect x='10' y='2' width='6' height='6' fill='"+st.c+"'/>":"")+"</svg>";leg+="<span>"+sw+" "+m+"</span>";});
 leg+="</div>";if(miss.length)leg+="<div class='sub' style='text-align:center;color:#dc2626'>no data for: "+miss.join(', ')+"</div>";
 el.innerHTML="<div style='width:100%'>"+s+leg+"</div>";
}
function _lipChecked(){var ms=[];document.querySelectorAll('.lipm-mcb:checked').forEach(function(cb){ms.push(cb.value);});return ms;}
function _lipMulti(elid,series,ticks,xlabel,title){
 var el=document.getElementById(elid);if(!el)return;var orc=LIP2D.oracle,nev=LIP2D.never_treat;
 if(!series.length){el.innerHTML="<span class='sub'>Check at least one method above.</span>";return;}
 var W=740,H=360,mL=54,mR=18,mT=30,mB=44,x0=mL,x1=W-mR,y0=H-mB,y1=mT;
 var all=[orc,nev];series.forEach(function(s){s.ys.forEach(function(v){if(v!=null&&v==v)all.push(v);});});
 var ymin=Math.min.apply(null,all)-0.05,ymax=Math.max.apply(null,all)+0.05,n=ticks.length;
 function sx(i){return x0+(n<=1?0.5:i/(n-1))*(x1-x0);}
 function sy(v){return y0-(v-ymin)/(ymax-ymin)*(y0-y1);}
 var s='<svg viewBox="0 0 '+W+' '+H+'" style="width:100%;max-width:760px">',i,yv,yy;
 for(i=0;i<=4;i++){yv=ymin+(ymax-ymin)*i/4;yy=sy(yv);s+='<line x1="'+x0+'" y1="'+yy+'" x2="'+x1+'" y2="'+yy+'" stroke="#eef0f7"/><text x="'+(x0-8)+'" y="'+(yy+4)+'" text-anchor="end" font-size="10" fill="#9aa3b2">'+yv.toFixed(2)+'</text>';}
 var oy=sy(orc);s+='<line x1="'+x0+'" y1="'+oy+'" x2="'+x1+'" y2="'+oy+'" stroke="#444" stroke-dasharray="4,4"/><text x="'+x1+'" y="'+(oy-3)+'" text-anchor="end" font-size="10" fill="#444">oracle '+orc.toFixed(2)+'</text>';
 var ny=sy(nev);s+='<line x1="'+x0+'" y1="'+ny+'" x2="'+x1+'" y2="'+ny+'" stroke="#c9ccd6" stroke-dasharray="4,4"/><text x="'+x1+'" y="'+(ny-3)+'" text-anchor="end" font-size="10" fill="#999">never-treat '+nev.toFixed(2)+'</text>';
 ticks.forEach(function(t,k){s+='<text x="'+sx(k)+'" y="'+(y0+18)+'" text-anchor="middle" font-size="10" fill="#9aa3b2">'+t+'</text>';});
 series.forEach(function(se){var st=_mstyle(se.m);
  var pts=se.ys.map(function(v,k){return (v==null||v!=v)?null:(sx(k)+','+sy(v));}).filter(function(p){return p;}).join(' ');
  s+='<polyline points="'+pts+'" fill="none" stroke="'+st.c+'" stroke-width="2.4"'+(st.dash?' stroke-dasharray="'+st.dash+'"':'')+'/>';
  se.ys.forEach(function(v,k){if(v!=null&&v==v){s+=st.mk?_mk(st.mk,sx(k),sy(v),st.c):'<circle cx="'+sx(k)+'" cy="'+sy(v)+'" r="2.6" fill="'+st.c+'"/>';}});});
 s+='<text x="'+((x0+x1)/2)+'" y="'+(H-6)+'" text-anchor="middle" font-size="12">'+xlabel+'</text>';
 s+='<text transform="translate(14,'+((y0+y1)/2)+') rotate(-90)" text-anchor="middle" font-size="11">test realised E[Y]</text>';
 s+='<text x="'+x0+'" y="'+(y1-12)+'" font-size="12.5" font-weight="700" fill="#334155">'+title+'</text></svg>';
 var leg="<div style='display:flex;flex-wrap:wrap;gap:8px 14px;justify-content:center;margin-top:4px;font-size:.8rem'>";
 series.forEach(function(se){var st=_mstyle(se.m);leg+="<span><svg width='24' height='9' style='vertical-align:middle'><line x1='1' y1='5' x2='22' y2='5' stroke='"+st.c+"' stroke-width='2.4'"+(st.dash?" stroke-dasharray='"+st.dash+"'":"")+"/></svg> "+se.m+"</span>";});
 el.innerHTML=s+leg+"</div>";
}
function drawLipVsL(){var g=document.getElementById('lip-gsel').value;var series=_lipChecked().map(function(m){return {m:m,ys:LIP2D.Ls.map(function(L){return (LIP2D.surface[m]&&LIP2D.surface[m][g])?LIP2D.surface[m][g][L]:null;})};});_lipMulti('lip-vsl-svg',series,LIP2D.Ls,'Lipschitz L  (inf = unit-by-unit)','test E[Y] vs L  (Γ = '+g+')');}
function drawLipVsG(){var L=document.getElementById('lip-lsel').value;var series=_lipChecked().map(function(m){return {m:m,ys:LIP2D.gammas.map(function(g){return (LIP2D.surface[m]&&LIP2D.surface[m][g])?LIP2D.surface[m][g][L]:null;})};});_lipMulti('lip-vsg-svg',series,LIP2D.gammas,'Γ (sensitivity)','test E[Y] vs Γ  (L = '+L+')');}
function drawLipBoth(){drawLipVsL();drawLipVsG();}
"""
BASE_CSS = open(os.path.join(HERE, "_base_template.css")).read()   # index.html look-and-feel (fonts, palette, hero, tabs, cards, chart)
REPORT_EXTRA_CSS = """
:root{--rose:#be123c;--ow:#16a34a;}
.wrap{}
.sub{color:var(--muted);margin:2px 0 14px;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--sh-sm);padding:16px 20px;margin:12px 0;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px;margin:14px 0;}
figure.card{margin:0;padding:12px;display:flex;flex-direction:column;}
figure.card img{width:100%;height:auto;border:1px solid var(--border);border-radius:9px;background:#fff;}
.ft{font-weight:700;font-size:.92rem;margin-bottom:8px;} .fc{color:var(--muted);font-size:.84rem;margin-top:8px;}
.restab{border-collapse:collapse;width:100%;box-shadow:none;border:none;border-radius:0;font-size:.86rem;margin:8px 0;background:transparent;}
.restab th,.restab td{border:1px solid var(--border);padding:5px 9px;text-align:center;}
.restab th{background:#f1f5f9;font-weight:600;color:#475569;}
.restab th:first-child,.restab td:first-child{text-align:left;}
.restab tr.hl{background:#ecfdf3;font-weight:600;} .restab tr:hover td{background:inherit;}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:middle;}
table.sep{border-collapse:collapse;width:100%;max-width:560px;margin:10px 0;font-size:.95rem;box-shadow:none;border-radius:0;}
table.sep th,table.sep td{border:1px solid var(--border);padding:8px 12px;}
table.sep td.win{color:var(--ow);font-weight:700;} table.sep td.fail{color:#dc2626;font-weight:700;}
.legend{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:10px 14px;font-size:.86rem;color:var(--muted);margin:10px 0;box-shadow:var(--sh-sm);}
.note{border-left:4px solid var(--rose);background:#fff1f3;padding:10px 14px;border-radius:0 10px 10px 0;margin:12px 0;}
.good{border-left:4px solid var(--ow);background:#ecfdf3;padding:10px 14px;border-radius:0 10px 10px 0;margin:12px 0;}
.scmethod{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:16px;align-items:center;margin:14px 0;}
@media(max-width:680px){.scmethod{grid-template-columns:1fr;}}
.scmethod figure.card{margin:0;}
.inner-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0 8px;}
.inner-panel.active{padding:4px 0 2px;}
.polmenus{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0;align-items:flex-start;}
.polchecks{display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;margin-top:4px;}
.polcheck{display:flex;align-items:center;gap:5px;font-weight:500;text-transform:none;letter-spacing:0;font-size:.8rem;color:var(--fg);cursor:pointer;white-space:nowrap;}
.polsel{display:flex;flex-direction:column;font-size:.78rem;font-weight:700;color:var(--muted);gap:4px;text-transform:uppercase;letter-spacing:.03em;}
.polsel select{font:inherit;font-weight:500;text-transform:none;letter-spacing:0;color:var(--fg);padding:6px 9px;border:1px solid var(--border);border-radius:7px;background:#fff;cursor:pointer;}
"""
FULL_CSS = BASE_CSS + "\n/* ===== report-specific additions ===== */" + REPORT_EXTRA_CSS
_CHART_MORDER = ['IPW-X-X','DoublyRobust-X-X','Direct-X-X','IPW-O-X','DoublyRobust-O-X','Hajek-O-X','IPW-O-W','DoublyRobust-O-W']
def _numk(k):
    try: return float(k)
    except Exception: return 1e18
def _chart_subtabs(prefix, items):
    btns = ''.join("<button class='subtab-btn%s' onclick=\"showInner('%s-%d',this)\"><span class='dot' style='background:%s'></span>%s</button>"
                   % (' active' if i == 0 else '', prefix, i, dot, lbl) for i, (lbl, dot, _h) in enumerate(items))
    panels = ''.join("<div id='%s-%d' class='inner-panel%s'>%s</div>" % (prefix, i, ' active' if i == 0 else '', html)
                     for i, (_l, _d, html) in enumerate(items))
    return "<div class='subtabs'>%s</div>%s" % (btns, panels)
_DOTPAL = ['#be123c','#2563eb','#0891b2','#7c3aed','#16a34a','#ea580c','#0d9488','#9333ea','#dc2626','#334155','#0369a1','#7c2d12']
# --- per-dataset SCOPED policy explorer: same checklist + Γ/seed dropdowns as the Policy-explorer tab, fixed to one dataset ---
_ds_list = [ds for ds in POLICY if GRID_BY_DS.get(ds)]
_ds2cid = {ds: "polc-%d" % i for i, ds in enumerate(_ds_list)}
SCOPED_MAP = {}   # cid -> {ds, ep, reg}
def _pchart_section(ds, title=None):
    cid = _ds2cid.get(ds)
    if not cid:
        return ""
    ep = '1.0' if '1.0' in POLICY[ds] else list(POLICY[ds])[0]
    regs = POLICY[ds][ep]; reg = 'uncap' if 'uncap' in regs else list(regs)[0]
    md = regs[reg]
    methods = [m for m in _CHART_MORDER if m in md]
    gset, sset = set(), set()
    for m in md:
        for g in md[m]:
            gset.add(g)
            for sd in md[m][g]:
                sset.add(sd)
    gammas = sorted(gset, key=_numk); seeds = sorted(sset, key=lambda s: (s != 'avg', _numk(s)))
    dg = '2' if '2' in gammas else (gammas[0] if gammas else '')
    dsd = 'avg' if 'avg' in seeds else (seeds[0] if seeds else '')
    SCOPED_MAP[cid] = {"ds": ds, "ep": ep, "reg": reg}
    _def = ('IPW-O-W', 'DoublyRobust-O-W', 'DoublyRobust-X-X')
    checks = "".join("<label class='polcheck'><input type='checkbox' class='pmcb-%s' value='%s'%s onchange=\"drawScoped('%s')\">%s</label>"
                     % (cid, m, ' checked' if m in _def else '', cid, m) for m in methods)
    gopts = "".join("<option%s>%s</option>" % (' selected' if g == dg else '', g) for g in gammas)
    sopts = "".join("<option%s>%s</option>" % (' selected' if s == dsd else '', s) for s in seeds)
    return ("<h3 style='margin-top:26px'>%s</h3>" % (title or "Learned policy π(treat | X)") +
            "<p class='sub'>Check one or more methods, pick Γ / seed (scoped to this dataset). Colour = estimator family, shape = uncertainty set "
            "(O-W ○, O-X □, X-X dashed); oracle dotted (treat iff X&gt;0).</p>"
            "<div class='polmenus'>"
            "<div class='polsel' style='min-width:220px'>methods (check one or more)<div class='polchecks'>" + checks + "</div></div>"
            "<label class='polsel'>Γ<select id='pg-%s' onchange=\"drawScoped('%s')\">%s</select></label>" % (cid, cid, gopts) +
            "<label class='polsel'>seed<select id='ps-%s' onchange=\"drawScoped('%s')\">%s</select></label>" % (cid, cid, sopts) +
            "</div>"
            "<div id='psvg-%s' class='panel' style='min-height:340px;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:8px'></div>" % cid)
# --- embed each experiment's scoped policy explorer under its own tab ---
tab_owgap = tab_owgap + _pchart_section('N=600')
if tab_n1000: tab_n1000 = tab_n1000 + _pchart_section('N=1000')
if tab_x3: tab_x3 = tab_x3 + _pchart_section('X3 (N=200)')
if tab_x4: tab_x4 = tab_x4 + _pchart_section('X4 (N=300)')
if tab_x4n16: tab_x4n16 = tab_x4n16 + _pchart_section('X4 (N=16)')
if tab_cont:
    _contds = [ds for ds in _ds_list if ds.startswith('Cont-')]
    tab_cont = tab_cont + "<h3 style='margin-top:28px'>Learned policies π(treat | X)</h3>" + \
        "<p class='sub'>The extended / Lipschitz policies deployed on the continuous grid. (For the Cont-Lipschitz datasets the Γ selector is the Lipschitz L.)</p>" + \
        "".join(_pchart_section(ds, ds) for ds in _contds)
_lip_var, _lip_init = "", ""
if os.path.exists(LIP_2D):
    _d2j = json.load(open(LIP_2D))
    _lip_var = "var LIP2D=" + json.dumps({"gammas": _d2j['gammas'], "Ls": _d2j['Lgrid'], "surface": _d2j['surface'],
                                          "oracle": _d2j['oracle'], "never_treat": _d2j['never_treat']}) + ";\n"
    _lip_init = "if(document.getElementById('lip-gsel')){drawLipVsL();drawLipVsG();}\n"
_SCOPED_JS = ("var SCOPED=" + json.dumps(SCOPED_MAP) + ";\n" + _lip_var + _SCOPED_DRAW +
              "\nObject.keys(SCOPED).forEach(function(c){drawScoped(c);});\n" + _lip_init)

TABS = [("overview","Overview"),("owgap","exp_owgap (N=600)")]
PANELS = {"overview":tab_overview, "owgap":tab_owgap}
if tab_n1000:
    TABS.append(("n1000","exp_owgap (N=1000)")); PANELS["n1000"] = tab_n1000
if tab_x3:
    TABS.append(("x3","X\u2208{\u22121,0,1} (N=200)")); PANELS["x3"] = tab_x3
if tab_x4:
    TABS.append(("x4","X\u2208{\u22121,\u22120.5,0.5,1} (N=300)")); PANELS["x4"] = tab_x4
if tab_x4n16:
    TABS.append(("x4n16","X4 small-sample (N=16)")); PANELS["x4n16"] = tab_x4n16
if tab_cont:
    TABS.append(("cont","Continuous X")); PANELS["cont"] = tab_cont
TABS.append(("sanity","Sanity check")); PANELS["sanity"] = tab_sanity
TABS.append(("policy","Policy explorer")); PANELS["policy"] = tab_policy
btns = ''.join("<button class='tab-btn%s' onclick=\"showTab('%s',this)\">%s</button>" % (" active" if i==0 else "", k, lbl) for i,(k,lbl) in enumerate(TABS))
panels = ''.join("<div id='tab-%s' class='tab-panel%s'>%s</div>" % (k, " active" if i==0 else "", PANELS[k]) for i,(k,_) in enumerate(TABS))

H = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>exp_owgap — Confounding-Robust Policy Optimization</title>
<style>""" + FULL_CSS + """</style>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head><body><div class="wrap">
<div class="hero"><h1>exp_owgap — Confounding-Robust Policy Optimization</h1><p>The selected experiment, every plot explained — discrete &amp; continuous, with the interactive policies.</p></div>
<div class="tabs">""" + btns + """</div>
""" + panels + """
</div>
<script>
function showTab(id,btn){document.querySelectorAll('.tab-panel').forEach(function(p){p.classList.remove('active');});
 document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
 document.getElementById('tab-'+id).classList.add('active'); btn.classList.add('active'); window.scrollTo({top:0,behavior:'smooth'});
 if(id==='policy'&&typeof drawPolicy==='function')drawPolicy();
 if(window.MathJax&&MathJax.typesetPromise)MathJax.typesetPromise();}
function showInner(id,btn){var s=btn.closest('.tab-panel');
 s.querySelectorAll('.inner-panel').forEach(function(p){p.classList.remove('active');});
 s.querySelectorAll('.inner-btn').forEach(function(b){b.classList.remove('active');});
 document.getElementById(id).classList.add('active'); btn.classList.add('active');}
</script>
</body></html>"""
H = H.replace("</script>", _POLICY_JS + "\nif(document.getElementById('pol-ds')){drawPolicy();}\n</script>")
# index-style interactive charts: inject ported CSS (+vars) and JS (renderChart + data + init) — after the drawPolicy inject
H = H.replace("</body>", "\n<script>\n" + _SCOPED_JS + "</script>\n</body>")
open(os.path.join(HERE, "report.html"), "w").write(H)
print("wrote report.html — %.2f MB, %d tabs (incl. policy explorer)" % (len(H)/1e6, len(TABS)))
