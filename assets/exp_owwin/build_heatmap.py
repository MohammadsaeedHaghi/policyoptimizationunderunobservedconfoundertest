"""Outcome heatmaps for exp_owwin: E[Y(t)|X,S] over the X-grid (cols) x S in {-1,+1} (rows), for control & treated,
plus the CATE = mu1-mu0. Saves outcome_heatmap.png (embedded as <img> in the tab)."""
import sys, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d)
X = d.LEVELS; S = np.array([1.0, -1.0])                         # rows: S=+1 (top), S=-1 (bottom)
M0 = np.array([[d.mu0(x, s) for x in X] for s in S])           # (2,7)
M1 = np.array([[d.mu1(x, s) for x in X] for s in S])
CT = M1 - M0                                                   # CATE = X (same both rows)
xt = [("%.2f" % x).rstrip("0").rstrip(".") for x in X]; yt = ["S=+1", "S=-1"]
vmax = float(np.max(np.abs([M0, M1])))
fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.2))
for ax, M, title, cmap, vmx in [(axes[0], M0, "Control  E[Y(0) | X, S] = 2.5·S", "RdBu_r", vmax),
                                (axes[1], M1, "Treated  E[Y(1) | X, S] = 2.5·S + X", "RdBu_r", vmax),
                                (axes[2], CT, "CATE = E[Y(1)−Y(0) | X] = X", "PuOr_r", 1.0)]:
    im = ax.imshow(M, cmap=cmap, vmin=-vmx, vmax=vmx, aspect="auto")
    ax.set_xticks(range(len(X))); ax.set_xticklabels(xt, fontsize=8); ax.set_yticks([0, 1]); ax.set_yticklabels(yt, fontsize=9)
    ax.set_xlabel("X", fontsize=9); ax.set_title(title, fontsize=9.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7.5,
                    color="white" if abs(M[i, j]) > 0.55 * vmx else "#222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("exp_owwin outcome structure:  S shifts both arms by 2.5 (confounding);  treatment adds X (CATE=X, independent of S)", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(HERE / "outcome_heatmap.png", dpi=130); plt.close(fig)
print("saved outcome_heatmap.png  | Y0 rows:", M0.tolist(), " Y1 rows:", M1.round(2).tolist())
