"""Compare seen vs unseen performance and generalization gap across several runs.
Usage: python scripts/plot_compare.py results/holdout_1 results/holdout_3 results/holdout_5 [--metric roc_auc]
Writes results/comparison_<metric>.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--metric", default="roc_auc")
ap.add_argument("--out", default="results")
a = ap.parse_args()

labels, seen, unseen = [], [], []
for r in a.runs:
    m = json.loads((Path(r) / "metrics.json").read_text())
    labels.append("hold " + ",".join(map(str, m["config"]["holdout"])))
    seen.append(m["seen"][a.metric]); unseen.append(m["unseen"][a.metric])

x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(x - w / 2, seen, w, label="seen"); ax[0].bar(x + w / 2, unseen, w, label="unseen")
ax[0].set_xticks(x, labels); ax[0].set(ylim=(0, 1.05), ylabel=a.metric, title="Seen vs unseen"); ax[0].legend()
gap = np.array(seen) - np.array(unseen)
ax[1].bar(x, gap, color="tab:red"); ax[1].set_xticks(x, labels)
ax[1].set(ylabel=f"{a.metric} gap (seen - unseen)", title="Generalization gap")
for i, v in enumerate(gap):
    ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
fig.tight_layout()
out = Path(a.out); out.mkdir(exist_ok=True)
fig.savefig(out / f"comparison_{a.metric}.png", dpi=150)
print(f"wrote {out / f'comparison_{a.metric}.png'}")
