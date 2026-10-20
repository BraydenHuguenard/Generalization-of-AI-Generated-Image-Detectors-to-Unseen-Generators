# Setup & Running (Phase 1: ResNet-50)

## Environment
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train
The Defactify dataset (~7.5 GB) downloads automatically from Hugging Face on first run.
Labels: `Label_A` = real/AI, `Label_B` = generator id (0 = real, 1-5 = generators).

```bash
python main.py --holdout 5 --epochs 5          # train on gens 1-4 + real, hold out gen 5
python main.py --holdout 3 5 --epochs 5        # hold out multiple generators
python main.py --holdout 5 --max-train 4000 --max-eval 1000 --epochs 1   # quick smoke test
```
Repeat with different `--holdout` values to measure sensitivity to generator identity.
Device is auto-selected (CUDA > Apple MPS > CPU). Use `--workers 0` if dataloader workers misbehave.

Outputs go to `results/holdout_<ids>/`: `history.json`, `metrics.json`
(overall / seen / unseen / per-generator / generalization gap), `predictions.npz`, `best.pt`.
The checkpoint is chosen on validation AUC over **seen** generators only; the test set includes all generators.

## Charts
```bash
python scripts/plot_training.py   results/holdout_5     # loss + val metric curves
python scripts/plot_generators.py results/holdout_5     # per-generator bars, ROC, confusion matrices
python scripts/plot_compare.py results/holdout_1 results/holdout_3 results/holdout_5   # gap across runs
```

## Google Colab
Open https://colab.research.google.com/github/BraydenHuguenard/Generalization-of-AI-Generated-Image-Detectors-to-Unseen-Generators/blob/main/colab.ipynb,
set Runtime → Change runtime type → T4 GPU, and run the cells in order. Results are saved to `MyDrive/dl_results/` in Google Drive.
Colab only sees what's pushed to GitHub, so push local changes first.
