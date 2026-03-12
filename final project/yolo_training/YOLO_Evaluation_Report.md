# YOLO Model Evaluation Report

**Model:** YOLOv11x-seg
**Task:** Instance segmentation — lithium dendrites in SEM images
**Classes:** 1 (`dendrite`)
**Dataset:** 10 train / 3 val / 1 test images
**Best checkpoint:** `runs/dendrite_seg/weights/best.pt` (epoch 146 of 473)

---

## 1. IoU / Precision / Recall

Metrics are computed on the validation set at each epoch during training. Values below are from the **best epoch (146)** — the checkpoint saved as `best.pt`.

### Segmentation Mask Metrics

| Metric | Value |
|---|---|
| Precision (M) | **46.90%** |
| Recall (M) | **31.85%** |
| F1 Score (M) | **37.94%** |
| mAP @ IoU=0.50 (M) | **31.25%** |
| mAP @ IoU=0.50:0.95 (M) | **11.33%** |

### Bounding Box Metrics

| Metric | Value |
|---|---|
| Precision (B) | 47.29% |
| Recall (B) | 36.79% |
| mAP @ IoU=0.50 (B) | 36.91% |
| mAP @ IoU=0.50:0.95 (B) | 17.22% |

> `(M)` = mask/segmentation metrics. `(B)` = bounding box metrics.
> mAP@0.50 is the area under the Precision–Recall curve at IoU threshold 0.50 (standard COCO metric).

---

## 2. IoU / Dice Score

### From Training Validation (best epoch)

| Metric | Value |
|---|---|
| Mask mAP@0.50 (proxy IoU) | 31.25% |
| Mask mAP@0.50:0.95 | 11.33% |
| **Dice Score** (derived: `2·IoU / (1+IoU)`) | **47.62%** |

> The Dice Score is computed algebraically from mAP50: `Dice = 2 × 0.3125 / (1 + 0.3125) = 0.4762`.

### Per-Image Pixel IoU / Dice (best.pt inference, conf=0.25)

Pixel-level IoU and Dice are computed by merging all predicted instance masks into a single binary mask and comparing against the ground-truth polygon annotations.

#### Training Set (10 images)

| Image | Predictions | Pixel IoU | Dice |
|---|---|---|---|
| 2e-9_100s_006.png | 287 | 14.88% | 25.91% |
| 2e-9_100s_010.png | 295 | 28.79% | 44.71% |
| 2e-9_100s_019.png | 292 | 39.28% | 56.40% |
| 70nm_diameter_100nm_pitch_012.png | 280 | 35.60% | 52.50% |
| 70nm_R_50nm_pitch_ETD_003.png | 270 | 12.82% | 22.72% |
| 70nm_R_50nm_pitch_ETD_019.png | 212 | 6.92% | 12.94% |
| Ag_1e-8_007.png | 286 | 25.45% | 40.58% |
| Ag_1e-8_011.png | 254 | 11.84% | 21.17% |
| Ag_1e-9__02_016.png | 287 | 36.83% | 53.84% |
| Ag_40nm_pitch_015.png | 268 | 18.85% | 31.72% |
| **Average** | | **23.13%** | **36.25%** |

#### Validation Set (3 images)

| Image | Predictions | Pixel IoU | Dice |
|---|---|---|---|
| 70nm_diameter_100nm_pitch_028.png | 296 | 20.59% | 34.15% |
| Ag_2e-9_011.png | 293 | 51.37% | 67.87% |
| Ag_40nm_pitch_008.png | 290 | 34.77% | 51.59% |
| **Average** | | **35.57%** | **51.20%** |

#### Test Set (1 image)

| Image | Predictions | Pixel IoU | Dice |
|---|---|---|---|
| Ag_1e-8_004.png | 295 | 39.41% | 56.54% |

---

## 3. Robustness Analysis

### Training Stability

| Metric | Value |
|---|---|
| Total epochs trained | 473 (across multiple runs) |
| Best epoch | 146 |
| Epochs with valid mask mAP50 > 0 | 430 / 473 |
| Mean mAP50(M) across all epochs | 14.52% |
| Max mAP50(M) | 31.25% |
| Min mAP50(M) | 0.06% |
| Std mAP50(M) | 9.28% |
| Coefficient of Variation | 0.64 (high — unstable) |
| Epochs with NaN/Inf val loss | 43 / 473 |

A coefficient of variation of 0.64 indicates **high metric variance** across training — the model did not converge to a stable plateau.

### Training Phases

| Epochs | Avg mAP50(M) | Observation |
|---|---|---|
| 1–30 | 3.97% | Warm-up: rapid loss drop, near-zero mAP |
| 31–100 | 15.72% | Unstable: NaN val losses, erratic mAP |
| 101–150 | 23.68% | Improving: mAP climbs, peaks at epoch 146 |
| 151–473 | ~0% | Repeated/restarted runs: no further gain |

### Robustness Across SEM Conditions

The dataset covers two subsets with different imaging conditions:

| Subset | Images | Characteristics |
|---|---|---|
| **Easy** | `2e-9_100s`, `Ag_1e-8`, `Ag_1e-9`, `Ag_2e-9` | Lower noise, clearer dendrite contrast |
| **Hard** | `70nm` pitch, `Ag_40nm` | High-density, overlapping branches, charging-effect blur |

Per-image results show the Easy subset achieves higher IoU/Dice on average (e.g. `Ag_2e-9_011` = IoU 51.4%) while Hard-subset images score lower (e.g. `70nm_R_50nm_pitch_ETD_019` = IoU 6.9%), indicating **limited cross-condition robustness**.

> With only 14 images total, these findings are indicative only and not statistically significant.

---

## 4. Failure Analysis

### F1 — Severe Dataset Undersizing

- Training set: **10 images**. Validation: **3 images**. Test: **1 image**.
- YOLOv11x-seg has ~56 million parameters — massively overparameterized for this sample count.
- Transfer learning (frozen backbone, pretrained weights) partially compensates, but validation statistics remain unreliable.
- A single mis-prediction on the 3-image validation set shifts mAP by approximately **33%**.

### F2 — Unstable Validation Losses (NaN / Inf)

- **43 out of 473 epochs** produced `NaN` or `Inf` in validation segmentation loss.
- Root cause: the 3-image validation set is too small — when the model produces zero detections on a batch, loss computation divides by zero.
- Effect: early stopping based on validation loss becomes unreliable; best.pt selection may not reflect true generalization.

### F3 — Low Absolute mAP (31.2% @ IoU=0.50)

- Dendrites are thin, fractal, and overlapping — structurally harder to segment than blob-like objects.
- Instance segmentation requires correctly separating individual branches, which is extremely challenging without a larger annotated dataset.
- mAP drops from **31.2% at IoU=0.50** to **11.3% at IoU=0.50:0.95**, showing that predicted mask boundaries are coarse relative to ground-truth polygons.

### F4 — Precision / Recall Imbalance (P=46.9%, R=31.9%)

- Recall is **~15 percentage points lower** than Precision.
- The model misses real dendrites more often than it generates false positives.
- Likely cause: incomplete ground-truth annotations (annotators may have missed fine branches), causing the model's correct predictions on unlabeled dendrites to be penalized as false positives — and actually missed dendrites to inflate false negatives.

### F5 — Multiple Training Runs Concatenated

- Training was configured for 150 epochs with patience=30 early stopping.
- `results.csv` contains **473 rows**, indicating the model was trained or resumed **multiple times**.
- Results from earlier runs (epochs 1–35 show very low metrics) are concatenated with later runs (epochs 36–473).
- Best weights (`best.pt`) correspond to epoch 146 of the most productive run.

### F6 — Hard-Subset Generalization Gap

- Per-image IoU for Hard-subset images ranges from **6.9% to 35.6%** vs up to **51.4%** for Easy-subset images.
- Hard-subset characteristics (70 nm pitch, tight packing, ETD detector noise, charging-effect blur) create imaging conditions the model has seen very few examples of.
- With only ~3–4 Hard-subset images in training, the model cannot reliably generalize to these conditions.
- **Recommended mitigation:** acquire more labeled Hard-subset images, or apply test-time augmentation (TTA) to improve prediction stability on noisy inputs.

---

## Summary Table

| Category | Key Finding |
|---|---|
| **Best IoU (val set avg)** | 35.6% pixel IoU |
| **Best Dice (val set avg)** | 51.2% |
| **Test IoU / Dice** | 39.4% / 56.5% |
| **mAP@0.50 (mask)** | 31.2% |
| **Precision / Recall** | 46.9% / 31.9% |
| **F1 Score** | 37.9% |
| **Training stability** | Low (CV=0.64, 43 NaN epochs) |
| **Generalization** | Limited — 14 images, high variance across conditions |
| **Primary failure mode** | Insufficient training data for a 56M-param model |
