"""Per-generator bars (seen vs. held-out highlighted), ROC curves, and confusion matrix.
Usage: python scripts/plot_generators.py results/holdout_5 [--metric roc_auc]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--metric", default="roc_auc")
a = ap.parse_args()
run = Path(a.run)
m = json.loads((run / "metrics.json").read_text())
pg = m["per_generator"]

# per-generator metric bars
fig, ax = plt.subplots(figsize=(7, 4))
names = list(pg)
vals = [pg[n][a.metric] for n in names]
ax.bar(names, vals, color=["tab:red" if pg[n]["held_out"] else "tab:blue" for n in names])
ax.set(ylim=(0, 1.05), ylabel=a.metric, title=f"{a.metric} per generator (red = held out)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
fig.tight_layout(); fig.savefig(run / f"per_generator_{a.metric}.png", dpi=150)

# ROC curves: seen vs unseen vs each generator
d = np.load(run / "predictions.npz")
p, y, g = d["probs"], d["labels"], d["gens"]
holdout = [int(n[3:]) for n in names if pg[n]["held_out"]]
fig, ax = plt.subplots(figsize=(5.5, 5))
for gid in range(1, 6):
    s = (g == 0) | (g == gid)
    fpr, tpr, _ = roc_curve(y[s], p[s])
    ax.plot(fpr, tpr, ls="--" if gid in holdout else "-", label=f"gen{gid}" + (" (held out)" if gid in holdout else ""))
ax.plot([0, 1], [0, 1], "k:", lw=0.8)
ax.set(xlabel="FPR", ylabel="TPR", title="ROC per generator"); ax.legend()
fig.tight_layout(); fig.savefig(run / "roc_curves.png", dpi=150)

# confusion matrices: seen vs unseen
fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
seen_ids = [i for i in range(1, 6) if i not in holdout]
for axx, (t, ids) in zip(axs, (("seen", seen_ids), ("unseen", holdout))):
    s = (g == 0) | np.isin(g, ids)
    ConfusionMatrixDisplay.from_predictions(y[s], (p[s] >= 0.5).astype(int),
        display_labels=["real", "AI"], ax=axx, colorbar=False)
    axx.set_title(t)
fig.tight_layout(); fig.savefig(run / "confusion_matrices.png", dpi=150)
print(f"wrote plots to {run}")
