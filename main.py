"""fine-tune ResNet-50 as a real-vs-AI detector on the Defactify dataset
with one or more generators held out of training.

Defactify labels: Label_A = 0 real / 1 AI, Label_B = generator id
(0 = real, 1..5 = the five generators).

Example:
    python main.py --holdout 5 --epochs 5
    python main.py --holdout 3 5 --max-train 4000 --epochs 1   # quick smoke test
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from torchvision import models, transforms

GENERATORS = {0: "real", 1: "gen1", 2: "gen2", 3: "gen3", 4: "gen4", 5: "gen5"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", type=int, nargs="+", default=[5],
                   help="generator ids (1-5) excluded from training")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-train", type=int, default=None, help="subsample train set (debug)")
    p.add_argument("--max-eval", type=int, default=None, help="subsample val/test sets (debug)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="output dir (default results/holdout_<ids>)")
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(ds, tfm, args, shuffle):
    def apply(batch):
        batch["x"] = [tfm(im.convert("RGB")) for im in batch["Image"]]
        return batch

    ds = ds.with_transform(apply)

    def collate(items):
        return (torch.stack([i["x"] for i in items]),
                torch.tensor([i["Label_A"] for i in items], dtype=torch.float32),
                torch.tensor([i["Label_B"] for i in items]))

    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, collate_fn=collate)


def subsample(ds, n, seed):
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, gens = [], [], []
    for x, y, g in loader:
        probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu())
        labels.append(y)
        gens.append(g)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(gens).numpy()


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    out = dict(accuracy=accuracy_score(y, pred),
               precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0),
               f1=f1_score(y, pred, zero_division=0))
    out["roc_auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {k: float(v) for k, v in out.items()}


def evaluate_groups(probs, labels, gens, holdout):
    """Overall / seen / unseen / per-generator metrics.
    Each AI generator is scored against ALL real test images so AUC/precision are defined.
    """
    real = gens == 0
    res = {"overall": metrics(labels, probs)}
    for name, gen_ids in (("seen", [g for g in GENERATORS if g and g not in holdout]),
                          ("unseen", sorted(holdout))):
        m = real | np.isin(gens, gen_ids)
        res[name] = metrics(labels[m], probs[m])
    res["per_generator"] = {}
    for g in range(1, 6):
        m = real | (gens == g)
        res["per_generator"][GENERATORS[g]] = {
            "held_out": g in holdout, **metrics(labels[m], probs[m])}
    res["generalization_gap"] = {k: res["seen"][k] - res["unseen"][k] for k in res["seen"]}
    return res


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = get_device()
    out = Path(args.out or f"results/holdout_{'-'.join(map(str, sorted(args.holdout)))}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  holdout={args.holdout}  out={out}")

    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")

    train = ds["train"].filter(lambda b: [g not in args.holdout for g in b["Label_B"]],
                               batched=True, input_columns=["Label_B"])
    train = subsample(train, args.max_train, args.seed)
    val = subsample(ds["validation"].filter(lambda b: [g not in args.holdout for g in b["Label_B"]],
                                            batched=True, input_columns=["Label_B"]),
                    args.max_eval, args.seed)  # val = seen generators only (no peeking at unseen)
    test = subsample(ds["test"], args.max_eval, args.seed)  # all generators
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    train_tfm = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tfm = transforms.Compose([transforms.Resize((args.img_size, args.img_size)),
                                   transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_dl = make_loader(train, train_tfm, args, True)
    val_dl = make_loader(val, eval_tfm, args, False)
    test_dl = make_loader(test, eval_tfm, args, False)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    # real images are a minority (~1/6 of train), so up-weight the positive-class complement
    labels_a = np.array(train["Label_A"])
    n_real, n_fake = int((labels_a == 0).sum()), int((labels_a == 1).sum())
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_auc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for i, (x, y, _) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y); n += len(y)
            if i % 50 == 0:
                print(f"  epoch {epoch} step {i}/{len(train_dl)} loss {loss.item():.4f}", flush=True)
        scheduler.step()

        vp, vy, _ = predict(model, val_dl, device)
        vm = metrics(vy, vp)
        vloss = float(nn.functional.binary_cross_entropy(torch.tensor(vp), torch.tensor(vy)))
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": vloss,
                        **{f"val_{k}": v for k, v in vm.items()}})
        print(f"epoch {epoch}: train_loss {total / n:.4f} val_loss {vloss:.4f} "
              f"val_acc {vm['accuracy']:.4f} val_auc {vm['roc_auc']:.4f} ({time.time() - t0:.0f}s)")
        if vm["roc_auc"] > best_auc:  # select checkpoint on seen-generator val only
            best_auc = vm["roc_auc"]
            torch.save(model.state_dict(), out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    probs, labels, gens = predict(model, test_dl, device)
    results = evaluate_groups(probs, labels, gens, args.holdout)
    results["config"] = vars(args)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "predictions.npz", probs=probs, labels=labels, gens=gens)
    print(f"seen  acc {results['seen']['accuracy']:.4f} auc {results['seen']['roc_auc']:.4f}")
    print(f"unseen acc {results['unseen']['accuracy']:.4f} auc {results['unseen']['roc_auc']:.4f}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
