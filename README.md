# Domain-Adaptive GNNs for ASD Classification on ABIDE

This repository implements a domain-adaptive training framework for graph neural networks (GNNs) on the ABIDE I dataset, targeting improved generalisation via Leave-One-Site-Out (LOSO) cross-validation. The framework supports both baseline and domain-adaptive versions of models **DSAM**, **BrainGNN**, and **BNT**.

---

## 📁 Directory Structure

```
├── data/
│   ├── loaders/               # ABIDE .1D + .mat loader with T_min clipping and metadata
│   └── preprocessing/         # Optional preproc scripts for raw ABIDE PCP data
│
├── models/
│   ├── dsam/
│   ├── braingnn/
│   ├── bnt/
│   └── domain_adaptation/
│       ├── grl.py             # Gradient Reversal Layer
│       ├── components.py      # DomainDiscriminator, AdversarialLoss
│
├── evaluation/
│   ├── visualisation/
│       ├── roc_plot.py
│       └── tsne_umap.py
│
├── train/
│   ├── loso_trainer.py        # Full LOSO trainer with GRL toggle
│
├── config/
│   └── sweep.yaml             # WandB sweep config (baseline vs GRL)
│
├── scripts/
│   └── run_experiment.py      # Entry point for training a model
│
├── checkpoints/
├── figures/
└── metrics_summary.csv
```

---

## 🧠 Dataset

- **ABIDE I**: 17 sites, multi-site resting-state fMRI dataset.
- **Parcellation**: Any atlas (e.g., `cc200`) via .1D time series.
- **Connectivity**: Pearson matrices (.mat).
- **Labels**: ASD (1) vs TC (2) → Remapped to 0 (TC), 1 (ASD).
- **Metadata**: Age, Sex, ADOS scores embedded in `Data` object.

---

## 🔄 Preprocessing Highlights

- Automatically computes global `T_min` and clips all time series accordingly.
- Injects subject-level metadata into `Data`:
  - `.age` → `float32`
  - `.sex` → `0` (M), `1` (F)
  - `.ados` → `float32`, `-1.0` if missing
- Graph construction uses top-k percentile edge thresholding over correlation matrix.

---

## 🧪 Training Framework: LOSOTrainer

Supports both baseline (no domain adaptation) and domain-adaptive training using a `use_grl` flag.

### Features:
- 🧠 Leave-One-Site-Out (LOSO) cross-validation
- 🧪 Stratified, site-balanced validation split (e.g., 90/10 within each training site)
- 📉 Early stopping based on validation AUC
- 🔥 Learning rate scheduler support
- ✅ Best model checkpointing per site
- 🧬 GRL-based adversarial domain loss (optional)
- 📊 WandB logging for:
  - per-site val AUC/loss/accuracy
  - train loss
  - test AUC/sensitivity/specificity/balanced accuracy
- 📈 ROC + t-SNE + UMAP plots for each site

---

## ⚙️ Configuration (OmegaConf)

Example config:
```yaml
model:
  name: DSAM
  hidden_dim: 64

dataset:
  atlas: cc200
  connectivity: correlation
  num_nodes: 200
  val_split: 0.1
  batch_size: 32
  root: /path/to/ABIDE_preprocessed
  phenotypic_file: /path/to/Phenotypic_V1_0b_preprocessed1.csv

training:
  epochs: 100
  patience: 10
  min_delta: 0.001

domain_adaptation:
  use_grl: true
  grl_lambda: 1.0
  domain_loss_weight: 1.0
```

---

## 🚀 Running Training

```bash
python run_loso.py
```

---

## 🔁 WandB Sweeps

The provided `config/sweep.yaml` runs baseline and domain-adaptive variants for each model:

```yaml
method: grid
parameters:
  model.name:
    values: [DSAM, BrainGNN, BNT]
  domain_adaptation.use_grl:
    values: [true, false]
```

### Launch:

```bash
wandb sweep config/sweep.yaml
wandb agent <your-sweep-id>
```

---

## 📊 Metrics Tracked

For each held-out site and at macro level:
- Accuracy
- AUC
- Sensitivity & Specificity
- F1 score
- Balanced accuracy

---

## 📌 TODOs and Future Work

- [ ] Add support for ABIDE II
- [ ] Multi-task regression head for ADOS prediction
- [ ] GRL scheduling (cosine/linear ramp-up)
- [ ] Integration with DSBN, MixUp, or site-consistency losses

---

## 👤 Authors

Jesse Ssempijja – MSc Advanced Computer Science, University of Sheffield  
Dissertation supervised by Dr. Shuo Zhou (2024–2025)

---

## Citations

@inproceedings{
  kan2022bnt,
  title={BRAIN NETWORK TRANSFORMER},
  author={Xuan Kan and Wei Dai and Hejie Cui and Zilong Zhang and Ying Guo and Carl Yang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}
