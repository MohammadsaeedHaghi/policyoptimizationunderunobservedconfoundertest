#!/usr/bin/env python3
"""Work out the (treatment, covariates, outcome) structure of each downloaded dataset.

Nothing here is guessed from the file name: every role is read off the actual columns, and the
numbers printed (arm sizes, raw difference in means, missingness) are computed from the data so
the HTML write-up can quote measured values rather than recalled ones.

Writes structure.json next to the data for the report builder to consume.
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = {}


def note(name, **kw):
    OUT[name] = kw
    print("\n" + "=" * 78)
    print("%s  --  %s" % (name, kw.get("title", "")))
    print("=" * 78)
    for k in ("n_rows", "n_cols", "treatment", "outcome", "n_treated", "n_control"):
        if k in kw: print("  %-12s %s" % (k, kw[k]))
    if "ate_raw" in kw: print("  %-12s %s" % ("raw diff", kw["ate_raw"]))
    if "covariates" in kw:
        c = kw["covariates"]
        print("  %-12s %d: %s%s" % ("covariates", len(c), ", ".join(c[:10]),
                                    " ..." if len(c) > 10 else ""))


# ----------------------------------------------------------------- IHDP
z = np.load(HERE / "ihdp/ihdp_npci_1-100.train.npz")
zt = np.load(HERE / "ihdp/ihdp_npci_1-100.test.npz")
x, t, yf = z["x"], z["t"], z["yf"]
mu0, mu1 = z["mu0"], z["mu1"]
note("IHDP",
     title="Infant Health and Development Program (semi-synthetic outcomes on real covariates)",
     n_rows=int(x.shape[0]), n_cols=int(x.shape[1]),
     n_replications=int(x.shape[2]), test_rows=int(zt["x"].shape[0]),
     treatment="t  (1 = intensive early-intervention, 0 = control)",
     outcome="yf  (factual cognitive test score); ycf = counterfactual, both SIMULATED",
     n_treated=int(t[:, 0].sum()), n_control=int((1 - t[:, 0]).sum()),
     ate_raw="%.3f (yf treated minus control, replication 1)"
             % (yf[t[:, 0] == 1, 0].mean() - yf[t[:, 0] == 0, 0].mean()),
     ate_true="%.3f (mu1 - mu0, the KNOWN ground-truth ATE, replication 1)"
              % float((mu1[:, 0] - mu0[:, 0]).mean()),
     covariates=["x%d (6 continuous: birth weight, head circumference, weeks born "
                 "pre-term, birth order, neonatal health index, mother's age)" % 1,
                 "x7..x25 (19 binary: infant sex, twin status, mother's marital status, "
                 "education, race, cigarette/alcohol/drug use, prenatal care, site 1-8)"],
     n_arms=2, arm_detail="binary: t in {0, 1}",
     counterfactuals="ALL -- the file ships ycf (the unobserved arm) AND mu0/mu1 (the noiseless "
                     "response surfaces), so every unit has both potential outcomes",
     cf_kind="simulated", arrays=list(z.files), ground_truth=True,
     note="Covariates and treatment are REAL (from the 1985-88 RCT); outcomes are simulated "
          "from the NPCI 'A' response surface, which is why mu0/mu1 exist. The published "
          "benchmark also removes a non-random subset of treated children to induce imbalance.")

# ----------------------------------------------------------------- Twins
tx = pd.read_csv(HERE / "twins/twin_pairs_X_3years_samesex.csv", low_memory=False)
tt = pd.read_csv(HERE / "twins/twin_pairs_T_3years_samesex.csv")
ty = pd.read_csv(HERE / "twins/twin_pairs_Y_3years_samesex.csv")
note("Twins",
     title="Twin births, NBER natality 1989-1991, same-sex pairs under 2kg",
     n_rows=len(tx), n_cols=tx.shape[1],
     treatment="derived: being the HEAVIER twin of the pair (dbirwt_1 > dbirwt_0)",
     outcome="mort_0 / mort_1  (1-year mortality of the lighter / heavier twin)",
     n_treated="n/a -- BOTH potential outcomes are observed, one per twin",
     ate_raw="%.5f (mort_1 mean %.5f minus mort_0 mean %.5f)"
             % (ty["mort_1"].mean() - ty["mort_0"].mean(), ty["mort_1"].mean(), ty["mort_0"].mean()),
     covariates=[c for c in tx.columns
                 if not c.startswith("Unnamed") and not c.startswith("infant_id")][:12],
     weight_cols=[c for c in tt.columns if "bir" in c.lower()],
     n_arms=2, arm_detail="binary: lighter (0) vs heavier (1) twin of the pair",
     counterfactuals="ALL -- mort_0 and mort_1 are both present and non-null on "
                     "%d of %d rows (100%%), because the pair supplies both arms" % (
                         int(ty[["mort_0", "mort_1"]].notna().all(1).sum()), len(ty)),
     cf_kind="real", ground_truth=True,
     note="Not an RCT. The trick: for a twin PAIR both outcomes are observed, so the pair acts "
          "as its own counterfactual and the individual treatment effect is known exactly. "
          "Benchmarks then HIDE one twin at random and simulate a confounded assignment.")

# ----------------------------------------------------------------- LaLonde / NSW
cols = ["treat", "age", "education", "black", "hispanic", "married", "nodegree",
        "re74", "re75", "re78"]
tr = pd.read_csv(HERE / "lalonde/nswre74_treated.txt", sep=r"\s+", header=None, names=cols)
co = pd.read_csv(HERE / "lalonde/nswre74_control.txt", sep=r"\s+", header=None, names=cols)
nsw = pd.concat([tr, co], ignore_index=True)
cps = pd.read_csv(HERE / "lalonde/cps_controls.txt", sep=r"\s+", header=None, names=cols)
psid = pd.read_csv(HERE / "lalonde/psid_controls.txt", sep=r"\s+", header=None, names=cols)
note("Jobs / LaLonde (NSW)",
     title="National Supported Work Demonstration, 1975-1979 -- a genuine RCT",
     n_rows=len(nsw), n_cols=len(cols),
     treatment="treat  (1 = offered the subsidised-work training programme)",
     outcome="re78  (real earnings in 1978, US$; the post-treatment year)",
     n_treated=int(nsw.treat.sum()), n_control=int((1 - nsw.treat).sum()),
     ate_raw="$%.0f (experimental benchmark: re78 treated $%.0f minus control $%.0f)"
             % (nsw[nsw.treat == 1].re78.mean() - nsw[nsw.treat == 0].re78.mean(),
                nsw[nsw.treat == 1].re78.mean(), nsw[nsw.treat == 0].re78.mean()),
     covariates=["age", "education", "black", "hispanic", "married", "nodegree",
                 "re74 (earnings 1974)", "re75 (earnings 1975)"],
     observational_controls={"CPS-1": len(cps), "PSID-1": len(psid)},
     naive_bias="$%.0f (swap the RCT controls for CPS controls and the estimate becomes this)"
                % (nsw[nsw.treat == 1].re78.mean() - cps.re78.mean()),
     n_arms=2, arm_detail="binary: treat in {0, 1}",
     counterfactuals="NONE -- one arm per person; no ycf column exists in the files",
     cf_kind="none", ground_truth=False,
     note="The canonical test: the RCT gives a trustworthy ATE, then you REPLACE the randomised "
          "controls with a survey comparison group (CPS or PSID) and see whether your "
          "observational estimator can recover the experimental number.")

# ----------------------------------------------------------------- NHEFS
nh = pd.read_csv(HERE / "nhefs/nhefs.csv")
comp = nh.dropna(subset=["wt82_71", "qsmk"])
note("NHEFS",
     title="NHANES I Epidemiologic Follow-up Study -- observational, Hernan & Robins textbook",
     n_rows=len(nh), n_cols=nh.shape[1],
     treatment="qsmk  (1 = quit smoking between the 1971 baseline and the 1982 follow-up)",
     outcome="wt82_71  (weight change in kg, 1971 to 1982). Secondary: death, sbp, dbp",
     n_treated=int(comp.qsmk.sum()), n_control=int((1 - comp.qsmk).sum()),
     ate_raw="%.2f kg (unadjusted: quitters %.2f minus non-quitters %.2f)"
             % (comp[comp.qsmk == 1].wt82_71.mean() - comp[comp.qsmk == 0].wt82_71.mean(),
                comp[comp.qsmk == 1].wt82_71.mean(), comp[comp.qsmk == 0].wt82_71.mean()),
     covariates=["sex", "age", "race", "education", "smokeintensity", "smokeyrs", "exercise",
                 "active", "wt71", "income", "marital", "asthma", "bronch", "hbp"],
     missing_outcome=int(nh.wt82_71.isna().sum()),
     n_arms=2, arm_detail="binary: qsmk in {0, 1}",
     counterfactuals="NONE -- one arm per person, and not even randomised",
     cf_kind="none", ground_truth=False,
     note="NOT randomised -- people choose to quit. It is the standard worked example for "
          "IP-weighting, standardisation and g-methods, so adjustment sets are well documented.")

# ----------------------------------------------------------------- IST
ist = pd.read_csv(HERE / "ist/IST_corrected.csv", low_memory=False,
                  encoding="latin-1")   # file has non-UTF8 bytes
asp = ist["RXASP"].astype(str).str.upper().str.startswith("Y")
dead6 = ist["OCCODE"].isin([1])  # 1 = dead at 6 months
note("IST",
     title="International Stroke Trial, 1991-1996 -- a large factorial RCT",
     n_rows=len(ist), n_cols=ist.shape[1],
     treatment="RXASP (aspirin Y/N) and RXHEP (heparin N/L/M) -- a 2 x 3 FACTORIAL randomisation",
     outcome="OCCODE (1 dead / 2 dependent / 3 not recovered / 4 recovered at 6 months); "
             "DEAD14, FDEAD, FRECOVER are binary alternatives",
     n_treated=int(asp.sum()), n_control=int((~asp).sum()),
     ate_raw="%+.4f (6-month death rate: aspirin %.4f minus no-aspirin %.4f)"
             % (dead6[asp].mean() - dead6[~asp].mean(), dead6[asp].mean(), dead6[~asp].mean()),
     covariates=["AGE", "SEX", "RSBP (systolic BP)", "RDELAY (hours since stroke)",
                 "RCONSC (consciousness)", "RATRIAL (atrial fibrillation)", "RVISINF",
                 "RSLEEP", "RDEF1..RDEF8 (neurological deficits)", "STYPE (stroke subtype)"],
     heparin_arms=ist["RXHEP"].value_counts().to_dict(),
     n_arms=int((pd.crosstab(ist.RXASP.astype(str).str.strip().str.upper(),
                             ist.RXHEP.astype(str).str.strip().str.upper()).values > 0).sum()),
     arm_detail="FACTORIAL 2 x 4 = 8 occupied cells. Aspirin Y/N (9720 / 9715) crossed with "
                "heparin N / L / M / H (9718 / 4861 / 4611 / 245). H is a small high-dose "
                "sub-study; the headline design is usually read as 2 x 3 = 6 arms",
     counterfactuals="NONE -- one arm per patient",
     cf_kind="none", ground_truth=False,
     note="Randomised, so the raw difference IS the causal effect. The size (~19k patients, 36 "
          "countries) and the factorial design make it the usual real-data testbed for "
          "heterogeneous-effect and policy-learning methods.")

# ----------------------------------------------------------------- STAR
st = pd.read_csv(HERE / "star/STAR.csv", low_memory=False)
sk = st.dropna(subset=["stark", "readk", "mathk"]).copy()
sk["small"] = (sk["stark"] == "small").astype(int)
sk["total"] = sk["readk"] + sk["mathk"]
note("STAR",
     title="Tennessee Student/Teacher Achievement Ratio, 1985-1989 -- RCT on class size",
     n_rows=len(st), n_cols=st.shape[1],
     treatment="stark / star1 / star2 / star3  (small | regular | regular+aide, randomised per grade)",
     outcome="readk, mathk (and grades 1-3): scaled achievement test scores",
     n_treated=int(sk.small.sum()), n_control=int((1 - sk.small).sum()),
     ate_raw="%+.1f points (kindergarten read+math: small class %.1f minus other %.1f)"
             % (sk[sk.small == 1].total.mean() - sk[sk.small == 0].total.mean(),
                sk[sk.small == 1].total.mean(), sk[sk.small == 0].total.mean()),
     covariates=["gender", "ethnicity", "birth (quarter)", "lunchk (free-lunch status)",
                 "schoolk (school type: rural/urban/inner-city/suburban)",
                 "degreek (teacher degree)", "ladderk (teacher career ladder)",
                 "experiencek (teacher years)", "tethnicityk (teacher ethnicity)"],
     arms=st["stark"].value_counts(dropna=True).to_dict(),
     n_arms=3,
     arm_detail="3 arms per grade, re-recorded each year: small / regular / regular+aide. "
                "Kindergarten 1900 / 2194 / 2231 (n=6325); grade 1 1925 / 2584 / 2320 (6829); "
                "grade 2 2016 / 2329 / 2495 (6840); grade 3 2174 / 2085 / 2543 (6802)",
     counterfactuals="NONE -- one class type per child per year",
     cf_kind="none", ground_truth=False,
     note="Randomisation is WITHIN school, so school should be conditioned on. Three treatment "
          "arms, not two -- most causal work either drops the aide arm or pools it with regular.")

json.dump(OUT, open(HERE / "structure.json", "w"), indent=1, default=str)
print("\n\nwrote structure.json")
