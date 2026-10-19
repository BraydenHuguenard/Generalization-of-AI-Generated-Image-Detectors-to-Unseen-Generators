"""Training/validation loss and validation metric curves for one run.
Usage: python scripts/plot_training.py results/holdout_5
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

run = Path(sys.argv[1])
h = json.loads((run / "history.json").read_text())
ep = [r["epoch"] for r in h]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ep, [r["train_loss"] for r in h], marker="o", label="train")
ax[0].plot(ep, [r["val_loss"] for r in h], marker="o", label="val")
ax[0].set(title="BCE loss", xlabel="epoch"); ax[0].legend()
for k in ("accuracy", "f1", "roc_auc"):
    ax[1].plot(ep, [r[f"val_{k}"] for r in h], marker="o", label=k)
ax[1].set(title="Validation metrics (seen generators)", xlabel="epoch"); ax[1].legend()
fig.tight_layout()
fig.savefig(run / "training_curves.png", dpi=150)
print(f"wrote {run / 'training_curves.png'}")
