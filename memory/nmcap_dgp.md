---
name: nmcap-dgp
description: "exp_nmcap NON-MONOTONE discrete-X DGP (companion to owgap) where naive AIPW fails (treat-the-edges) and O-W recovers the middle band; + the 20-seed paired-CI lock of owgap-uncapped and the cap-rescue structural finding."
metadata:
  node_type: project
  type: project
  originSessionId: 16b87e39-c6c8-4604-955a-f4b218ddc989
---

**Goal (2026-06-24):** after [[owgap-dgp]], the user asked to (1) LOCK the strong uncapped owgap claim at 20 seeds
with PAIRED CIs, and (2) build a NON-MONOTONE DGP that fixes the capped regime (where owgap's DR-O-W only TIED
naive AIPW). Then a combined HTML explaining both: `assets/exp_nmcap/combined_report.html`.

**(1) owgap uncapped, 20 seeds, paired bootstrap CIs (`assets/exp_owgap/owgap_uncap_20seed_ce1.0.json`,
`paired_ci.py`):** vs naive DR-X-X (AIPW): IPW-O-W **Δ=+0.12 [0.09,0.15] SIG, 18/20 seeds**; DR-O-W **Δ=+0.09
[0.04,0.13] SIG, 17/20**. Box-only DR-O-X does NOT beat naive (Δ<0) -> the Wasserstein/O-W combo is the
differentiator. STRONG, paper-defensible. (Paired = same-seed diff = removes across-seed variance; the right test.)

**(2) exp_nmcap NON-MONOTONE DGP** (`assets/exp_nmcap/dgp.py`, search `search_nmcap.py`): X 7-level, S in{-1,+1}
**S _|_ X** (P=1/2); e=clip(σ(**1.2**S + **3**X^2),.02,.98) (S-confounding + EDGE-heavy selection); μ0=**3**S,
μ1=**4**S + **3**(0.3 - X^2). CATE=3(0.3-X^2) (S terms cancel, E[S|X]=0) -> NON-MONOTONE: **treat the MIDDLE band**
X^2<0.3 (X in {-1/3,0,1/3}), HARM the edges; always-treat is harmful (-0.43). oracle 0.29.
- MECHANISM: at the severe edges the treated are a high-S elite -> outcome model μ1_hat biased UP there -> naive
  AIPW over-credits the edges and (under cap<=50%) spends its budget treating the HARMFUL edges. O/O-W hedge the
  edge-imbalanced treated mass -> recover the middle band.
- N=500 15-seed paired CIs: **UNCAPPED naive AIPW goes NEGATIVE (-0.01); both O-W beat it +0.15..0.18 SIG.**
  CAPPED: cap limits AIPW damage (0.186) but **DR-O-W still SIG beats it (+0.04 [0.01,0.07])** = the gap owgap
  couldn't close; IPW-O-W ties AIPW capped. **Naive IPW self-corrects here** (edge over-selection shows in the
  X-propensity) -> ties O-W.

**THE COMBINED FINDING (the paper message):** each naive baseline fails in EXACTLY ONE setting and O-W survives
both: **naive IPW fails under unobserved-S confounding (owgap: 0.33 vs O-W 0.78)**; **naive AIPW fails under a
non-monotone effect (nmcap: -0.01/0.19 vs O-W 0.17/0.24)**. So no single naive method is safe across confounding
structures; the O-W (Wasserstein-balanced) methods are.

**STRUCTURAL CAVEAT (honest):** under a CAPACITY CAP, the cap RESCUES whichever naive correction is well-specified
(the outcome model in owgap-capped, the propensity in nmcap-capped) -> beating EVERY naive baseline capped is
structurally hard; O-W matches the best and beats the failing one. The cleanest uniform O-W wins are UNCAPPED. A
2nd search adding S-X correlation (`search_nmcap2.py`) just RE-rescued naive AIPW capped -> confirms the cap-rescue
is general, not DGP-specific.

Pipeline: `paired_ci.py` (per-seed values from saved policies + bootstrap CIs), `build_combined.py` (standalone
combined_report.html, reuses index.html renderChart/colours/CSS; SD bands + paired-CI tables + 2x2 robustness
summary). See [[owgap-dgp]], [[nonmono-experiment]], [[discrete-ow-wins-dgp]], [[no_true_propensities]],
[[save-all-artifacts]], [[html-raw-lessthan-bug]].
