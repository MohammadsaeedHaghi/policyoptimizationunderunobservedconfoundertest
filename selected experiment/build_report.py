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
 "\\[ T\\mid X,S \\;\\sim\\; \\mathrm{Bernoulli}\\big(e(X,S)\\big), \\qquad e(X,S)=\\mathrm{clip}\\!\\big(\\sigma(0.8\\,S-2\\,X),\\;0.02,\\;0.98\\big). \\]"
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

tab_owgap = ("<div class='mhead'><span class='pill' style='background:#be123c'>exp_owgap</span>"
 "<h2 style='border:none;margin:0'>Hidden-vitality clinical DGP</h2></div>"
 "<p class='lead'><b>Design:</b> X discrete (7 levels); S∈{−1,+1} unobserved, P(S=+1|X)=σ(10X); e(X,S)=σ(0.8S−2X); "
 "μ₀=8S, μ₁=9S+1.5X; CATE(X)=(2σ(10X)−1)+1.5X ⇒ treat iff X&gt;0. Oracle=0.847. (DGP: <code>exp_owgap/dgp.py</code>.)</p>"
 + DGP_EXPLAIN + LEGEND + DGP_TRUTH +
 "<h3 style='margin-top:28px'>Results by regime</h3>" +
 inner_tabs('owgap',
   "<h3>Realised value vs Γ (ε sweep)</h3>" + grid(owg_val_uncap) +
   "<h3>Result table (realised E[Y], c_ε=1 — O-W rows highlighted)</h3><div class='panel'>" + owg_u + "</div>"
   "<div class='good'><b>Finding (uncapped).</b> At Γ=2–3 both O-W beat the best naive competitor (AIPW=0.64) by +0.11–0.18; "
   "20-seed paired CI IPW-O-W − AIPW = +0.12 [0.09, 0.15] at Γ=3. Box-only AIPW (DR-O-X) fails → the Wasserstein term is the differentiator.</div>",
   "<h3>Realised value vs Γ (ε sweep)</h3>" + grid(owg_val_cap) +
   "<h3>Result table (realised E[Y], c_ε=1 — O-W rows highlighted)</h3><div class='panel'>" + owg_c + "</div>"
   "<div class='note'><b>Finding (capped).</b> The cap rescues AIPW to 0.71, so IPW-O-W carries the clean win while DoublyRobust-O-W ties at small Γ.</div>"))

tab_lim = ("<div class='mhead'><span class='pill' style='background:#7c3aed'>Limitations</span>"
 "<h2 style='border:none;margin:0'>Over-conservatism at extreme Γ×ε</h2></div>"
 "<p class='lead'>Inflating the uncertainty set far past anything realistic makes the worst-case-<i>value</i> methods hedge to never-treat — a property to disclose, not hide.</p>"
 + grid(lim) +
 "<div class='note'><b>Takeaway.</b> Pushing Γ and ε together drives IPW-O-W from 0.85 to ~0.08 (~never-treat); the doubly-robust "
 "variants are anchored (~0.6) by their direct μ̂ term; Hajek-O-X collapses earliest. ⇒ Anchor strongest claims on the uncapped, c_ε=1 operating point.</div>")

# ---------- N=1000 rerun (added as a NEW tab; the N=600 sections above are kept) ----------
N1000_CES = [ce for ce in ['1.0','1.5','2.0'] if os.path.exists(os.path.join(OWG, 'owgap_results_n1000_ce%s.json' % ce))]
N1000_ORDER = ['IPW-X-X','DoublyRobust-X-X','Direct-X-X','IPW-O-X','DoublyRobust-O-X','Hajek-O-X','IPW-O-W','DoublyRobust-O-W']
def _n1style(name):
    fam = name.split('-')[0]; c = FAM.get(fam, '#888')
    if name.endswith('-O-W'): return c,'-','o'
    if name.endswith('-O-X'): return c,'-','s'
    if name.endswith('-X-X'): return c,'--',None
    return c,'-',None
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
      "<h3 style='margin-top:28px'>Results by regime</h3>" + inner_tabs('n1000', _n1000_regime('uncap'), _n1000_regime('cap')) +
      "<div class='note'>N=1000 is a separate rerun at fewer seeds (%d) than the N=600 headline (5-seed, 20 for the CI); treat the exact margins as preliminary until all ε and more seeds are in.</div>" % _ns)
else:
    tab_n1000 = None

# ---------- interactive policy explorer: π(treat|X) vs X with dropdowns ----------
POLICY = {}; GRID_X = []
def _ingest_policy(path, ds, ep):
    global GRID_X
    d = json.load(open(path)); GRID_X = [round(float(x),4) for x in d['grid']]
    for reg in d['regimes']:
        for m, seeds in d['regimes'][reg].get('policy_by_seed', {}).items():
            for sk, gd in seeds.items():
                for gk, vals in gd.items():
                    POLICY.setdefault(ds,{}).setdefault(ep,{}).setdefault(reg,{}).setdefault(m,{}).setdefault(gk,{})[sk] = [round(float(x),4) for x in vals]
for _ce in ['1.0','1.5','2.0']:
    _p = os.path.join(OWG, 'owgap_results_ce%s.json' % _ce)
    if os.path.exists(_p): _ingest_policy(_p, 'N=600', _ce)
    _p = os.path.join(OWG, 'owgap_results_n1000_ce%s.json' % _ce)
    if os.path.exists(_p): _ingest_policy(_p, 'N=1000', _ce)
_datasets = list(POLICY.keys()) or ['(none)']
_epslist = sorted({e for ds in POLICY.values() for e in ds}) or ['1.0']
_methods = ['IPW-X-X','DoublyRobust-X-X','Direct-X-X','IPW-O-X','DoublyRobust-O-X','Hajek-O-X','IPW-O-W','DoublyRobust-O-W']
_gammakeys = ['1','1.5','2','2.5','3','4','5','6','8']
_seedkeys = ['avg','0','1','2','3','4']
def _selrow(idd, label, op):
    return "<label class='polsel'>%s<select id='%s' onchange='drawPolicy()'>%s</select></label>" % (label, idd, ''.join('<option>%s</option>' % o for o in op))
tab_policy = ("<div class='mhead'><span class='pill' style='background:#0d9488'>Policy explorer</span>"
 "<h2 style='border:none;margin:0'>Deployed policy π(treat | X)</h2></div>"
 "<p class='lead'>Select dataset, ε, regime, method, Γ and seed to see the learned treatment policy "
 "π(treat | X) across the covariate X. Dotted black = oracle (treat iff X&gt;0); dashed grey = 0.5. "
 "(<code>avg</code> = mean policy over seeds.)</p>"
 "<div class='polmenus'>" + _selrow('pol-ds','dataset',_datasets) + _selrow('pol-eps','c_ε',_epslist)
 + _selrow('pol-reg','regime',['uncap','cap']) + _selrow('pol-method','method',_methods)
 + _selrow('pol-gamma','Γ',_gammakeys) + _selrow('pol-seed','seed',_seedkeys) + "</div>"
 "<div id='policy-svg' class='panel' style='min-height:380px;display:flex;align-items:center;justify-content:center'></div>")

_POLICY_JS = ("var POLICY_DATA=" + json.dumps(POLICY) + ";var GRID_X=" + json.dumps(GRID_X) + ";"
 "function _pv(id){var e=document.getElementById(id);return e?e.value:null;}"
 "function drawPolicy(){var ds=_pv('pol-ds'),ep=_pv('pol-eps'),rg=_pv('pol-reg'),m=_pv('pol-method'),g=_pv('pol-gamma'),sd=_pv('pol-seed');"
 "var el=document.getElementById('policy-svg');if(!el)return;var node=null;try{node=POLICY_DATA[ds][ep][rg][m][g][sd];}catch(e){node=null;}"
 "if(!node){el.innerHTML=\"<span class='sub'>No policy data for this combination — try seed 'avg' or another ε/dataset.</span>\";return;}"
 "var W=720,H=380,mL=58,mR=22,mT=22,mB=46,x0=mL,x1=W-mR,y0=H-mB,y1=mT;"
 "var xmin=Math.min.apply(null,GRID_X),xmax=Math.max.apply(null,GRID_X);"
 "function sx(x){return x0+(x-xmin)/(xmax-xmin)*(x1-x0);}function sy(v){return y0-v*(y0-y1);}"
 "var s='<svg viewBox=\"0 0 '+W+' '+H+'\" style=\"width:100%;max-width:740px\">',i,yv,y;"
 "for(i=0;i<=4;i++){yv=i/4;y=sy(yv);s+='<line x1=\"'+x0+'\" y1=\"'+y+'\" x2=\"'+x1+'\" y2=\"'+y+'\" stroke=\"#eee\"/><text x=\"'+(x0-8)+'\" y=\"'+(y+4)+'\" text-anchor=\"end\" font-size=\"11\" fill=\"#999\">'+yv.toFixed(2)+'</text>';}"
 "GRID_X.forEach(function(xx){var px=sx(xx);s+='<text x=\"'+px+'\" y=\"'+(y0+18)+'\" text-anchor=\"middle\" font-size=\"10\" fill=\"#999\">'+xx.toFixed(2)+'</text>';});"
 "s+='<line x1=\"'+x0+'\" y1=\"'+sy(0.5)+'\" x2=\"'+x1+'\" y2=\"'+sy(0.5)+'\" stroke=\"#bbb\" stroke-dasharray=\"4,4\"/>';"
 "var orc=GRID_X.map(function(xx){return xx>0?1:0;});"
 "s+='<polyline points=\"'+GRID_X.map(function(xx,k){return sx(xx)+','+sy(orc[k]);}).join(' ')+'\" fill=\"none\" stroke=\"#444\" stroke-dasharray=\"2,3\" stroke-width=\"1.5\"/>';"
 "s+='<polyline points=\"'+GRID_X.map(function(xx,k){return sx(xx)+','+sy(node[k]);}).join(' ')+'\" fill=\"none\" stroke=\"#0d9488\" stroke-width=\"2.6\"/>';"
 "GRID_X.forEach(function(xx,k){s+='<circle cx=\"'+sx(xx)+'\" cy=\"'+sy(node[k])+'\" r=\"3.6\" fill=\"#0d9488\"/>';});"
 "s+='<text x=\"'+((x0+x1)/2)+'\" y=\"'+(H-6)+'\" text-anchor=\"middle\" font-size=\"12\">X</text>';"
 "s+='<text transform=\"translate(15,'+((y0+y1)/2)+') rotate(-90)\" text-anchor=\"middle\" font-size=\"12\">π(treat | X)</text>';"
 "s+='<text x=\"'+x1+'\" y=\"'+(y1+2)+'\" text-anchor=\"end\" font-size=\"10.5\" fill=\"#0d9488\">'+ds+' · '+m+' · Γ='+g+' · '+rg+' · seed '+sd+'</text>';"
 "s+='</svg>';el.innerHTML=s;}")

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

TABS = [("overview","Overview"),("owgap","exp_owgap (N=600)"),("lim","Limitations")]
PANELS = {"overview":tab_overview, "owgap":tab_owgap, "lim":tab_lim}
if tab_n1000:
    TABS.append(("n1000","exp_owgap (N=1000)")); PANELS["n1000"] = tab_n1000
TABS.append(("sanity","Sanity check")); PANELS["sanity"] = tab_sanity
TABS.append(("policy","Policy explorer")); PANELS["policy"] = tab_policy
btns = ''.join("<button class='tab-btn%s' onclick=\"showTab('%s',this)\">%s</button>" % (" active" if i==0 else "", k, lbl) for i,(k,lbl) in enumerate(TABS))
panels = ''.join("<div id='tab-%s' class='tab-panel%s'>%s</div>" % (k, " active" if i==0 else "", PANELS[k]) for i,(k,_) in enumerate(TABS))

H = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>exp_owgap — Confounding-Robust Policy Optimization</title>
<style>
:root{--fg:#1f2433;--muted:#6b7280;--border:#e4e7ef;--surface:#fff;--bg:#f6f7fb;--accent:#334155;--rose:#be123c;--ow:#16a34a;}
*{box-sizing:border-box} body{margin:0;font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 80px;}
h1{font-size:1.6rem;margin:0 0 4px;} h2{font-size:1.25rem;} h3{font-size:1.05rem;margin:24px 0 8px;}
.sub{color:var(--muted);margin:0 0 14px;} .lead{font-size:1.02rem;}
.tabs{position:sticky;top:0;z-index:10;display:flex;gap:8px;flex-wrap:wrap;background:var(--bg);padding:12px 0;border-bottom:1px solid var(--border);margin-bottom:18px;}
.tab-btn{border:1px solid var(--border);background:var(--surface);font:inherit;font-weight:650;color:var(--muted);padding:8px 16px;border-radius:9px;cursor:pointer;}
.tab-btn.active{color:#fff;background:linear-gradient(180deg,#475569,#334155);border-color:#334155;}
.tab-panel{display:none;} .tab-panel.active{display:block;}
.inner-tabs{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 4px;}
.inner-btn{border:1px solid var(--border);background:var(--surface);font:inherit;font-weight:600;font-size:.9rem;color:var(--muted);padding:6px 14px;border-radius:8px;cursor:pointer;}
.inner-btn.active{color:#fff;background:var(--accent);border-color:var(--accent);}
.inner-panel{display:none;border:1px solid var(--border);border-radius:0 10px 10px 10px;background:rgba(255,255,255,.5);padding:8px 16px 16px;}
.inner-panel.active{display:block;}
.mhead{display:flex;align-items:center;gap:12px;margin:6px 0 4px;}
.pill{display:inline-block;color:#fff;border-radius:20px;padding:3px 13px;font-weight:700;font-size:.82rem;}
.card,.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;box-shadow:0 2px 8px rgba(20,30,60,.05);}
.panel{padding:16px 20px;margin:12px 0;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px;margin:14px 0;}
figure.card{margin:0;padding:12px;display:flex;flex-direction:column;}
figure.card img{width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:#fff;}
.ft{font-weight:700;font-size:.92rem;margin-bottom:8px;} .fc{color:var(--muted);font-size:.84rem;margin-top:8px;}
table.restab{border-collapse:collapse;width:100%;font-size:.86rem;margin:8px 0;}
table.restab th,table.restab td{border:1px solid var(--border);padding:5px 9px;text-align:center;}
table.restab th:first-child,table.restab td:first-child{text-align:left;}
table.restab tr.hl{background:#ecfdf3;font-weight:600;}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:middle;}
table.sep{border-collapse:collapse;width:100%;max-width:560px;margin:10px 0;font-size:.95rem;}
table.sep th,table.sep td{border:1px solid var(--border);padding:8px 12px;}
table.sep td.win{color:var(--ow);font-weight:700;} table.sep td.fail{color:#dc2626;font-weight:700;}
.legend{background:#fff;border:1px solid var(--border);border-radius:10px;padding:10px 14px;font-size:.86rem;color:var(--muted);margin:10px 0;}
code{background:#eef1f7;padding:1px 5px;border-radius:5px;font-size:.85em;}
ol,ul{margin:6px 0 6px 18px;} li{margin:4px 0;}
.note{border-left:4px solid var(--rose);background:#fff1f3;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0;}
.good{border-left:4px solid var(--ow);background:#ecfdf3;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0;}
.scmethod{display:grid;grid-template-columns:minmax(280px,360px) 1fr;gap:16px;align-items:center;margin:14px 0;}
@media(max-width:680px){.scmethod{grid-template-columns:1fr;}}
.scmethod figure.card{margin:0;}
.polmenus{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0;}
.polsel{display:flex;flex-direction:column;font-size:.78rem;font-weight:700;color:var(--muted);gap:4px;text-transform:uppercase;letter-spacing:.03em;}
.polsel select{font:inherit;font-weight:500;text-transform:none;letter-spacing:0;color:var(--fg);padding:6px 9px;border:1px solid var(--border);border-radius:7px;background:#fff;cursor:pointer;}
</style>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head><body><div class="wrap">
<h1>exp_owgap — Confounding-Robust Policy Optimization</h1>
<p class="sub">The selected experiment, every plot explained. (owgap-only — exp_nmcap removed.)</p>
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
open(os.path.join(HERE, "report.html"), "w").write(H)
print("wrote report.html — %.2f MB, %d tabs (incl. policy explorer)" % (len(H)/1e6, len(TABS)))
