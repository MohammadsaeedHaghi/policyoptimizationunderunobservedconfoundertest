"""One-off: restructure the sub-exp_1d (bern) panel in index.html into the exp_first-style Capped/Uncapped inner-subtab
layout (interactive policy chart w/ seed+Γ dropdowns + interactive realised/objective charts + static perf PNGs per
regime + per-regime Finding), and repoint the bern render calls. Idempotent: skips if already applied."""
import re
from pathlib import Path
NEW=Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
idx=NEW/"index.html"; h=idx.read_text()
if "ichart-exp1d-cap-policy" in h:
    print("already applied; nothing to do"); raise SystemExit
H4='<h4 style="margin:18px 0 4px;font-size:1.04rem">'
def policy_panel(reg,rlabel):
    return ('%s%s' % (H4, 'Treatment policy \\(\\pi(\\text{treat}\\mid x)\\)</h4>')) + \
           ('\n    <div class="ichart" id="ichart-exp1d-%s-policy" style="max-width:940px"></div>' % reg)
NEWBLOCK = r'''    <p class="lead">A <strong>single discrete feature</strong> \(X\), one unobserved \(S\), two treatments — the
    simplest possible setting, shown in <strong>both capacity regimes</strong> (toggle below). <strong>Uncapped</strong>,
    nothing forces a ranking, so the robust covariate-balance term has no lever and <strong>R-OW \(\approx\) IPW</strong> —
    robustness does not win. <strong>Under the \(\le 50\%\) cap</strong> the Wasserstein term binds, so <strong>R-OW beats
    the box-only R-O</strong> and IPW, and the doubly-robust <strong>R-OW-DR is the single best method</strong>, reaching the
    best-deployable ceiling. The Wasserstein edge does <em>not</em> need multi-dimensional covariates — only that the
    capacity constraint binds.</p>

    <div class="card" style="border-left:4px solid var(--accent)">
    <strong>Configuration</strong>
    <table style="margin-top:8px">
    <tr><th>field</th><th>value</th></tr>
    <tr><td>DGP</td><td>discrete <strong>1-D</strong> 2-arm; \(X\) on a 21-point grid in \([-1,1]\); unobserved \(S\sim\)Bernoulli(0.5)</td></tr>
    <tr><td>outcome model</td><td>control \(\sigma(0.5)\); treatment \(\sigma(-1.2+4X+5S)\)</td></tr>
    <tr><td>treatment model</td><td>\(\operatorname{logit}P(T{=}1\mid X,S)=-\,X+\gamma(S-\tfrac12)\) (mis-targeted + S-channel)</td></tr>
    <tr><td>methods</td><td>R-OW · R-O · IPW · <strong>AIPW</strong> · Regret-O · Hajek-OW · <strong>R-OW-DR · R-O-DR</strong> · Kallus</td></tr>
    <tr><td>\(\gamma_{\text{true}}\) · Γ sweep</td><td>5 (matched \(\Gamma=e^{2.5}=12.18\)) · \(\{1,3,6,9,12.18,16\}\)</td></tr>
    <tr><td>regimes · z-score · \(n_{\text{train}}/n_{\text{test}}\) · seeds</td><td>uncapped (1,&nbsp;1) &amp; capped (1,&nbsp;0.5) · OFF (1-D) · 500 / 2000 · 3</td></tr>
    </table>
    </div>

    <div class="innertabs">
      <span class="innertabs-label">Performance:</span>
      <button class="inner-btn active" onclick="showInner('exp1d-cap',this)">Capped&nbsp; (treat ≤ 50%)</button>
      <button class="inner-btn" onclick="showInner('exp1d-uncap',this)">Uncapped&nbsp; (treat ≤ 100%)</button>
    </div>

    <div id="exp1d-cap" class="inner-panel active">
    <h3>Performance — capped (treat \(\le 50\%\))</h3>
    <p class="note">At most 50% of units may be treated, so each method must <strong>rank</strong> who benefits most. The
    optimal fraction (\(X\ge0\)) is \(\approx0.52\), so the cap <em>just</em> binds — enough to activate the robust
    covariate-balance term. Curves are mean \(\pm\) SD over 3 seeds.</p>
    HH4Treatment policy \(\pi(\text{treat}\mid x)\)</h4>
    <div class="ichart" id="ichart-exp1d-cap-policy" style="max-width:940px"></div>
    <p class="figcap"><strong>Interactive</strong> — deployed \(\pi(\text{treat}\mid X)\) over the 21-X grid; pick a
    <strong>seed</strong> or <strong>Average</strong> (under Average a value in \((0,1)\) is the <em>fraction of seeds</em>
    that treat at that \(X\)), the <strong>Γ&nbsp;▾</strong>, and toggle methods. At the matched Γ, R-OW-DR / R-OW
    concentrate the capped budget on the genuinely-helped high-\(X\) units; R-O / IPW spend budget on low-\(X\) cells the
    unobserved \(S\) inflated; Kallus hugs the never-treat baseline.</p>
    HH4Realised outcome vs Γ — train (in-sample) vs test (deployed)</h4>
    <div class="ichart" id="ichart-exp1d-cap-rz-train" style="max-width:880px"></div>
    <div class="ichart" id="ichart-exp1d-cap-rz-test" style="max-width:880px"></div>
    <p class="figcap"><strong>Top = train, bottom = test (deployed)</strong>; mean \(\pm\) SD over 3 seeds (band = ±1 SD;
    toggle methods to de-clutter). At the matched Γ the test ranking is <strong>R-OW-DR 0.718 &gt; AIPW 0.715 &gt; R-OW 0.705
    &gt; R-O-DR 0.703 &gt; Hajek-OW 0.698 &gt; IPW 0.670 &gt; R-O 0.646 &gt; Kallus 0.618 &gt; Regret-O 0.617</strong>.
    R-OW-DR is both best and the most seed-stable (SD ±0.002). Green dashed = best-means ceiling 0.726; Full-info 0.729 is
    in the note (off scale).</p>
    HH4Worst-case objective vs Γ</h4>
    <div class="ichart" id="ichart-exp1d-cap-obj" style="max-width:820px"></div>
    <p class="figcap">Worst-case in-sample objective (mean \(\pm\) SD over 3 seeds): the value methods (R-OW/R-O/IPW/AIPW)
    report a worst-case value, the regret methods a worst-case regret \(\le 0\).</p>
    <div class="figwrap"><img onerror="this.style.display='none'" src="assets/exp_1d/realized_bar_cap.png" alt="Realised value at matched Gamma, capped" style="width:100%;max-width:900px"></div>
    <p class="figcap">Realised value at the matched Γ, capped — <strong>R-OW-DR on top</strong>; the box-only <strong>R-O dips below IPW</strong> (odds-box robustness alone hurts), and the Wasserstein methods recover it.</p>
    <div class="figwrap"><img onerror="this.style.display='none'" src="assets/exp_1d/treat_fraction_cap.png" alt="Treat fraction vs Gamma, capped" style="width:100%;max-width:900px"></div>
    <p class="figcap">Fraction treated vs Γ, capped — methods press against the 0.5 cap (red dashed); the difference is <em>which</em> half they treat.</p>
    </div>

    <div id="exp1d-uncap" class="inner-panel">
    <h3>Performance — uncapped (treat \(\le 100\%\))</h3>
    <p class="note">No capacity limit: every unit may be treated independently, so there is no scarce budget to allocate and
    nothing forces a ranking. The robust covariate-balance term does not bind, so robustness has no lever here. Curves are
    mean \(\pm\) SD over 3 seeds.</p>
    HH4Treatment policy \(\pi(\text{treat}\mid x)\)</h4>
    <div class="ichart" id="ichart-exp1d-uncap-policy" style="max-width:940px"></div>
    <p class="figcap"><strong>Interactive</strong> — uncapped \(\pi(\text{treat}\mid X)\) (seed / Average, Γ, method toggles
    as above). Without the cap R-OW behaves \(\approx\) IPW — a per-\(X\) bang-bang with no pooling — because there is no
    scarce budget for the Wasserstein balance to reallocate.</p>
    HH4Realised outcome vs Γ — train (in-sample) vs test (deployed)</h4>
    <div class="ichart" id="ichart-exp1d-uncap-rz-train" style="max-width:880px"></div>
    <div class="ichart" id="ichart-exp1d-uncap-rz-test" style="max-width:880px"></div>
    <p class="figcap">mean \(\pm\) SD over 3 seeds. At the matched Γ: <strong>Hajek-OW 0.698 &gt; IPW 0.686 ≈ AIPW 0.685
    &gt; R-OW 0.674 &gt; R-O-DR 0.672 ≈ R-O 0.671 &gt; R-OW-DR 0.663 &gt; Kallus 0.618 &gt; Regret-O 0.617</strong>.
    <strong>R-OW (0.674) sits below IPW (0.686)</strong>, and R-OW-DR (0.663) below both — with no binding cap, robustness
    does not win (it can even cost a little). Best-means ceiling 0.726.</p>
    HH4Worst-case objective vs Γ</h4>
    <div class="ichart" id="ichart-exp1d-uncap-obj" style="max-width:820px"></div>
    <p class="figcap">Worst-case in-sample objective, uncapped (mean \(\pm\) SD over 3 seeds).</p>
    <div class="figwrap"><img onerror="this.style.display='none'" src="assets/exp_1d/realized_bar_uncap.png" alt="Realised value at matched Gamma, uncapped" style="width:100%;max-width:900px"></div>
    <p class="figcap">Realised value at the matched Γ, uncapped — the robust/DR methods do <em>not</em> lead; IPW, AIPW and Hajek-OW sit at the top.</p>
    <div class="figwrap"><img onerror="this.style.display='none'" src="assets/exp_1d/treat_fraction_uncap.png" alt="Treat fraction vs Gamma, uncapped" style="width:100%;max-width:900px"></div>
    <p class="figcap">Fraction treated vs Γ, uncapped — confounded methods drift toward treat-all.</p>
    </div>

    <div class="card" style="border-left:4px solid var(--c-row)">
    <strong>Finding.</strong> Read the capped regime as an <strong>ablation</strong> of two ingredients. Plain <strong>IPW</strong>
    (inverse-propensity only — no outcome model, no robustness) realises 0.670. The non-robust doubly-robust baseline
    <strong>AIPW</strong> (\(=\) IPW \(+\) an estimated outcome model \(\hat\mu_k\), still Γ=1) jumps to <strong>0.715</strong>:
    the <em>outcome model alone</em> is the big gain. Adding <em>confounding robustness</em> on top, <strong>R-OW-DR</strong>
    reaches <strong>0.718</strong> — essentially the best-deployable ceiling (0.726). Among the IPW-only methods the two
    Wasserstein methods (R-OW 0.705, Hajek-OW 0.698) beat IPW, while box-only <strong>R-O dips below IPW</strong>
    (0.646 &lt; 0.670) — odds-box robustness <em>alone</em> hurts; the Wasserstein balance recovers it. <strong>Uncapped</strong>,
    the whole effect disappears: with no binding capacity there is no scarce budget to reallocate, so R-OW (0.674) \(\approx\)
    falls just below IPW (0.686) and R-OW-DR (0.663) below both — the cap is exactly what activates the robust edge (consistent
    with the 2-D and Case-4 experiments). Code: <code>assets/exp_1d/{exp_1d_multiseed.py, exp_1d_multiseed_agg.py, perf_plots.py, build_policy_chart.py}</code>;
    raw numbers in <code>_multiseed/bern_seed*.json</code> + <code>exp_1d_results.json</code>.
    </div>

    '''
NEWBLOCK = NEWBLOCK.replace("HH4", H4)
start = h.index('<p class="lead">A <strong>single discrete feature</strong>')
end   = h.index('<h3 style="margin-top:30px">Why these results happen')
h = h[:start] + NEWBLOCK + h[end:]
# repoint bern render calls (uni calls untouched)
h = h.replace("renderChart('ichart-bern',POLICY_DATA.bern);",
              "renderChart('ichart-exp1d-cap-policy',POLICY_EXP1D_CAP.bern);\n"
              "renderChart('ichart-exp1d-uncap-policy',POLICY_EXP1D_UNCAP.bern);")
h = h.replace("renderChart('ichart-rz-bern-train',REALIZED_DATA.bern.train);\n"
              "renderChart('ichart-rz-bern-test',REALIZED_DATA.bern.test);",
              "renderChart('ichart-exp1d-cap-rz-train',REALIZED_DATA.bern.cap.train);\n"
              "renderChart('ichart-exp1d-cap-rz-test',REALIZED_DATA.bern.cap.test);\n"
              "renderChart('ichart-exp1d-uncap-rz-train',REALIZED_DATA.bern.uncap.train);\n"
              "renderChart('ichart-exp1d-uncap-rz-test',REALIZED_DATA.bern.uncap.test);")
h = h.replace("renderChart('ichart-obj-bern',OBJECTIVE_DATA.bern);",
              "renderChart('ichart-exp1d-cap-obj',OBJECTIVE_DATA.bern.cap);\n"
              "renderChart('ichart-exp1d-uncap-obj',OBJECTIVE_DATA.bern.uncap);")
idx.write_text(h)
print("restructured sub-exp_1d panel + repointed bern render calls")
