#!/usr/bin/env python3
"""Smoke tests for semi-synthetic dataset candidates (UCI) for the OWGAP paper.

For each candidate we lock a tentative design -- observed covariates X_obs, a hidden
confounder S (a real column/composite we HIDE), a treatment T (real column when one exists),
and an outcome Y -- then measure the three quantities that decide viability under our
protocol (see the alpha-sweep boundary):

  coupling   how well X_obs proxies S: point-biserial corr(score, S) and AUC of a
             cross-validated logistic S ~ X_obs  (needs corr >~ 0.75-0.8 to be in-regime
             for a REAL-coupling design; weaker coupling => ACIC-style synthetic S instead)
  lambda     realized selection strength: T ~ X_obs + S logistic, S coded +-1,
             Lambda = exp(2*beta_S) = the matched Gamma with everything else observed
  s_to_y     does S actually move the outcome (confounding needs S -> T AND S -> Y)

Raw downloads are cached OUTSIDE the repo (UCI_CACHE env or scratch); only the summary
JSON lives in the repo. Usage: python3 "semi experiments/smoke_tests.py"
"""
import os, json, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("UCI_CACHE", "/scratch1/haghim/uci_cache")); CACHE.mkdir(parents=True, exist_ok=True)

def fetch(uci_id, name):
    """ucimlrepo fetch with a local parquet cache."""
    fx, fy = CACHE / f"{name}_X.pkl", CACHE / f"{name}_y.pkl"
    if fx.exists():
        return pd.read_pickle(fx), pd.read_pickle(fy)
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=uci_id)
    X, y = ds.data.features, ds.data.targets
    X.to_pickle(fx), y.to_pickle(fy)
    return X, y

def onehot(df):
    num = df.select_dtypes(include=[np.number]).copy()
    cat = df.select_dtypes(exclude=[np.number])
    if len(cat.columns):
        num = pd.concat([num, pd.get_dummies(cat.astype(str), drop_first=True, dtype=float)], axis=1)
    num = num.replace([np.inf, -np.inf], np.nan)
    return num.fillna(num.median(numeric_only=True))

def coupling(Xobs, S):
    """CV logistic S ~ X_obs: AUC + point-biserial corr of the score with S."""
    Z = onehot(Xobs); Zs = (Z - Z.mean()) / (Z.std() + 1e-9)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    p = cross_val_predict(lr, Zs.values, S, cv=3, method="predict_proba")[:, 1]
    return float(np.corrcoef(p, S)[0, 1]), float(roc_auc_score(S, p))

def sel_lambda(Xobs, S, T):
    """T ~ X_obs + S (S coded +-1) -> Lambda = exp(2 beta_S)."""
    Z = onehot(Xobs); Zs = (Z - Z.mean()) / (Z.std() + 1e-9)
    Zs["_S"] = 2 * np.asarray(S, float) - 1
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Zs.values, T)
    b = float(lr.coef_[0][list(Zs.columns).index("_S")])
    return float(np.exp(2 * abs(b))), b

R = {}

def run(name, fn):
    try:
        R[name] = fn(); R[name]["status"] = "ok"
        print(name, "->", json.dumps(R[name], default=float)[:300], flush=True)
    except Exception as e:
        R[name] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
        traceback.print_exc(); print(name, "FAILED", flush=True)

# ---------------- candidates ----------------

def adult():
    X, y = fetch(2, "adult")
    df = X.copy(); df["income"] = y.iloc[:, 0].astype(str).str.contains(">50K").astype(int)
    df = df.dropna(subset=["workclass", "occupation"])
    S = ((df["capital-gain"] > 0) | (df["hours-per-week"] > 50)).astype(int)   # hidden "drive"
    Xobs = df[["age", "education-num", "marital-status", "occupation", "sex", "race"]]
    c, auc = coupling(Xobs, S)
    return {"n": len(df), "design": "A (real X, synthetic S/T/Y)", "T_real": None,
            "S_def": "capital-gain>0 or hours>50 (hidden drive)", "P_S": float(S.mean()),
            "corr_XS": c, "auc_XS": auc,
            "s_to_y": float(df.loc[S == 1, "income"].mean() - df.loc[S == 0, "income"].mean())}

def bank():
    X, y = fetch(222, "bank")
    df = X.copy(); df["sub"] = (y.iloc[:, 0].astype(str) == "yes").astype(int)
    T = (df["campaign"] > 1).astype(int)                       # re-contacted = "treatment"
    S = (df["balance"] > df["balance"].median()).astype(int)   # hidden financial health
    Xobs = df[["age", "job", "education", "marital", "housing", "loan"]]
    c, auc = coupling(Xobs, S)
    lam, b = sel_lambda(Xobs, S, T)
    return {"n": len(df), "design": "B (real X,S,T; synthetic Y)", "T_real": "campaign>1", "P_T": float(T.mean()),
            "S_def": "balance > median (hidden wealth)", "P_S": float(S.mean()),
            "corr_XS": c, "auc_XS": auc, "lambda": lam, "beta_S": b,
            "s_to_y": float(df.loc[S == 1, "sub"].mean() - df.loc[S == 0, "sub"].mean())}

def mushroom():
    X, y = fetch(73, "mushroom")
    df = X.copy(); df["pois"] = (y.iloc[:, 0].astype(str) == "p").astype(int)
    S = (~df["odor"].isin(["n", "a", "l"])).astype(int)        # bad odor (hidden from learner)
    Xobs = df.drop(columns=["odor", "pois"])
    c, auc = coupling(Xobs, S)
    # logging forager: eats iff odor fine -> P(T=1|S) = 0.9/0.1 -> huge real-mechanism Lambda
    return {"n": len(df), "design": "C (classification->policy; hidden feature = S)", "T_real": "synthetic forager",
            "S_def": "odor not in {none,almond,anise} (hidden)", "P_S": float(S.mean()),
            "corr_XS": c, "auc_XS": auc,
            "s_to_y": float(df.loc[S == 1, "pois"].mean() - df.loc[S == 0, "pois"].mean())}

def student():
    X, y = fetch(320, "student")
    df = X.copy(); df["G3"] = pd.to_numeric(y.iloc[:, -1] if y.shape[1] else df.get("G3"))
    T = (df["paid"].astype(str) == "yes").astype(int)
    S = ((df["Medu"] >= 3) | (df["Fedu"] >= 3)).astype(int)    # hidden parental education
    Xobs = df[["age", "sex", "address", "studytime", "failures", "absences", "internet"]]
    c, auc = coupling(Xobs, S)
    lam, b = sel_lambda(Xobs, S, T)
    return {"n": len(df), "design": "B (real X,S,T; synthetic Y)", "T_real": "paid classes", "P_T": float(T.mean()),
            "S_def": "parent edu >= secondary (hidden)", "P_S": float(S.mean()),
            "corr_XS": c, "auc_XS": auc, "lambda": lam, "beta_S": b,
            "s_to_y": float(df.loc[S == 1, "G3"].mean() - df.loc[S == 0, "G3"].mean())}

def credit_default():
    X, y = fetch(350, "credit_default")
    df = X.copy(); df["def"] = y.iloc[:, 0].astype(int)
    delin = df[[c for c in df.columns if str(c).startswith("X6") or str(c).startswith("PAY_0") or str(c) == "X6"]]
    paycols = [c for c in df.columns if str(c) in ("X6", "X7", "X8", "X9", "X10", "X11", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")]
    S = (df[paycols].max(axis=1) >= 1).astype(int) if paycols else None   # hidden: any delinquency
    lim = [c for c in df.columns if str(c) in ("X1", "LIMIT_BAL")][0]
    T = (df[lim] > df[lim].median()).astype(int)               # bank grants high limit
    Xobs = df[[c for c in df.columns if str(c) in ("X2", "X3", "X4", "X5", "SEX", "EDUCATION", "MARRIAGE", "AGE")]]
    c, auc = coupling(Xobs, S)
    lam, b = sel_lambda(Xobs, S, T)
    return {"n": len(df), "design": "B (real X,S,T; synthetic Y)", "T_real": "high credit limit", "P_T": float(T.mean()),
            "S_def": "any past delinquency (hidden risk file)", "P_S": float(S.mean()),
            "corr_XS": c, "auc_XS": auc, "lambda": lam, "beta_S": b,
            "s_to_y": float(df.loc[S == 1, "def"].mean() - df.loc[S == 0, "def"].mean())}

def support2():
    X, y = fetch(880, "support2")
    df = pd.concat([X, y], axis=1)
    df["Y"] = 1 - pd.to_numeric(df["death"], errors="coerce").fillna(0)      # survive = good
    sev = pd.to_numeric(df["aps"], errors="coerce")
    ok = sev.notna() & df["dnr"].notna()
    df = df[ok]; sev = sev[ok]
    S = (sev > sev.median()).astype(int)                        # hidden physiologic severity
    T = (df["dnr"].astype(str).str.strip().str.lower() != "no dnr").astype(int)   # DNR order written
    Xobs = df[[c for c in ("age", "sex", "dzgroup", "num.co", "edu", "income", "race") if c in df.columns]]
    c, auc = coupling(Xobs, S)
    lam, b = sel_lambda(Xobs, S, T)
    return {"n": int(len(df)), "design": "B (real X,S,T; synthetic Y) or A", "T_real": "DNR order",
            "P_T": float(T.mean()), "S_def": "APS physiology score > median (hidden)", "P_S": float(S.mean()),
            "corr_XS": c, "auc_XS": auc, "lambda": lam, "beta_S": b,
            "s_to_y": float(df.loc[S == 1, "Y"].mean() - df.loc[S == 0, "Y"].mean())}

def heart():
    X, y = fetch(45, "heart")
    df = X.copy(); df["dis"] = (y.iloc[:, 0].astype(int) > 0).astype(int)
    S = (pd.to_numeric(df["thal"], errors="coerce").fillna(3) > 3).astype(int)   # hidden perfusion defect
    Xobs = df[["age", "sex", "cp", "trestbps", "chol"]]
    c, auc = coupling(Xobs, S)
    return {"n": len(df), "design": "A (tiny; likely too small)", "T_real": None,
            "S_def": "thallium defect (hidden)", "P_S": float(S.mean()),
            "corr_XS": c, "auc_XS": auc,
            "s_to_y": float(df.loc[S == 1, "dis"].mean() - df.loc[S == 0, "dis"].mean())}

def ist():
    """International Stroke Trial (NOT UCI; Edinburgh DataShare, open licence).
    Download: https://datashare.ed.ac.uk/bitstream/handle/10283/124/IST_corrected.csv"""
    df = pd.read_csv(CACHE / "IST_corrected.csv", low_memory=False, encoding="latin-1")
    occ = pd.to_numeric(df["OCCODE"], errors="coerce")
    ok = df["RCONSC"].notna() & occ.isin([1, 2, 3, 4])
    d = df[ok]; occ = occ[ok]
    Y = (occ >= 3).astype(int)                              # favorable 6-month outcome (real!)
    S = (d["RCONSC"].astype(str).str.upper().str.startswith("F")).astype(int)   # fully alert (hidden)
    T = (d["RXASP"].astype(str).str.upper() == "Y").astype(int)                 # randomized aspirin
    Xobs = d[["AGE", "SEX", "RSBP", "STYPE", "RDEF1", "RDEF2", "RDEF3", "RDEF4",
              "RDEF5", "RDEF6", "RDEF7", "RDEF8"]]
    c, auc = coupling(Xobs, S.values)
    # NOTE: S=TACS alternative measured corr 0.987 (near-deterministic in X) -> rejected:
    # ~zero Var(S|X) leaves nothing hidden to be robust to. RCONSC is the principled choice.
    return {"n": int(len(d)), "design": "D (RCT + injected confounding; fully real Y)",
            "T_real": "randomized aspirin (P=%.2f); logging bias to be injected" % T.mean(),
            "P_T": float(T.mean()), "S_def": "fully alert at baseline (RCONSC, hidden)",
            "P_S": float(S.mean()), "corr_XS": c, "auc_XS": auc,
            "s_to_y": float(Y[S.values == 1].mean() - Y[S.values == 0].mean())}

for nm, fn in [("ist", ist), ("adult", adult), ("bank_marketing", bank), ("mushroom", mushroom),
               ("student", student), ("credit_default", credit_default),
               ("support2", support2), ("heart", heart)]:
    run(nm, fn)

# incumbent reference (already measured in assets/exp_diabetes)
R["diabetes130_reference"] = {"status": "ok", "n": 69990, "design": "A done + B done",
                              "T_real": "insulin", "corr_XS": 0.196, "lambda": 1.26,
                              "note": "incumbent: A (in-regime synthetic S/T) + B (fully real, diagnostic)"}

(HERE / "smoke_results.json").write_text(json.dumps(R, indent=1, default=float))
print("\nsaved smoke_results.json")
