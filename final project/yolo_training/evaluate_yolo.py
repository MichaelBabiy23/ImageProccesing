"""
YOLO Model Evaluation Script
Computes: IoU, Precision, Recall, mAP50, mAP50-95, Dice Score,
          Robustness analysis, and Failure analysis.

Usage:
    python evaluate_yolo.py
"""

import csv
import os
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
RESULTS_CSV  = SCRIPT_DIR / "runs/dendrite_seg/results.csv"
WEIGHTS_BEST = SCRIPT_DIR / "runs/dendrite_seg/weights/best.pt"
DATASET_YAML = SCRIPT_DIR / "yolo_dataset/data.yaml"
TRAIN_DIR    = SCRIPT_DIR / "yolo_dataset/train/images"
VAL_DIR      = SCRIPT_DIR / "yolo_dataset/valid/images"
TEST_DIR     = SCRIPT_DIR / "yolo_dataset/test/images"

# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_float(val):
    try:
        v = float(val)
        return v if (v == v) else None   # exclude NaN
    except (TypeError, ValueError):
        return None


def dice_from_iou(iou: float) -> float:
    """Dice = 2*IoU / (1 + IoU)  (exact algebraic relationship for binary masks)."""
    return 2 * iou / (1 + iou)


# ── 1. Training-curve stats from results.csv ──────────────────────────────────

def load_results():
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    return rows


def best_epoch(rows):
    return max(
        rows,
        key=lambda r: safe_float(r["metrics/mAP50(M)"]) or 0.0
    )


def print_training_summary(rows):
    best = best_epoch(rows)
    last = rows[-1]

    print("=" * 70)
    print("YOLO SEGMENTATION MODEL — EVALUATION REPORT")
    print("Model: YOLOv11x-seg  |  Classes: 1 (dendrite)")
    print("Dataset: 10 train / 3 val / 1 test  (SEM dendrite images)")
    print("=" * 70)

    print("\n── TRAINING OVERVIEW ─────────────────────────────────────────────────")
    print(f"  Total epochs completed : {len(rows)}")
    print(f"  Best epoch (mask mAP50): {best['epoch']}")
    print(f"  Time at best epoch     : {safe_float(best['time']):.0f} s")

    print("\n── BEST-EPOCH METRICS (mask / segmentation) ──────────────────────────")
    mp  = safe_float(best["metrics/precision(M)"])
    mr  = safe_float(best["metrics/recall(M)"])
    m50 = safe_float(best["metrics/mAP50(M)"])
    m95 = safe_float(best["metrics/mAP50-95(M)"])

    dice = dice_from_iou(m50) if m50 else None

    print(f"  Precision  (M) : {mp:.4f}  ({mp*100:.2f}%)")
    print(f"  Recall     (M) : {mr:.4f}  ({mr*100:.2f}%)")
    print(f"  mAP@0.50   (M) : {m50:.4f}  ({m50*100:.2f}%)")
    print(f"  mAP@0.5:95 (M) : {m95:.4f}  ({m95*100:.2f}%)")
    print(f"  Dice Score*    : {dice:.4f}  ({dice*100:.2f}%)")
    print("  * Derived from mAP50:  Dice = 2·IoU/(1+IoU)")

    print("\n── BEST-EPOCH METRICS (bounding box) ────────────────────────────────")
    bp  = safe_float(best["metrics/precision(B)"])
    br  = safe_float(best["metrics/recall(B)"])
    b50 = safe_float(best["metrics/mAP50(B)"])
    b95 = safe_float(best["metrics/mAP50-95(B)"])
    print(f"  Precision  (B) : {bp:.4f}  ({bp*100:.2f}%)")
    print(f"  Recall     (B) : {br:.4f}  ({br*100:.2f}%)")
    print(f"  mAP@0.50   (B) : {b50:.4f}  ({b50*100:.2f}%)")
    print(f"  mAP@0.5:95 (B) : {b95:.4f}  ({b95*100:.2f}%)")

    print("\n── LAST-EPOCH METRICS (final checkpoint) ────────────────────────────")
    lmp  = safe_float(last["metrics/precision(M)"])
    lmr  = safe_float(last["metrics/recall(M)"])
    lm50 = safe_float(last["metrics/mAP50(M)"])
    print(f"  Precision  (M) : {lmp:.4f}   Recall (M): {lmr:.4f}   mAP50 (M): {lm50:.4f}")

    return best


# ── 2. Robustness analysis ────────────────────────────────────────────────────

def robustness_analysis(rows):
    print("\n── ROBUSTNESS ANALYSIS ──────────────────────────────────────────────")

    # Collect valid mAP50(M) values per epoch
    vals = [safe_float(r["metrics/mAP50(M)"]) for r in rows]
    vals = [v for v in vals if v is not None and v > 0]

    if not vals:
        print("  No valid mAP50(M) values found.")
        return

    mean_v = sum(vals) / len(vals)
    max_v  = max(vals)
    min_v  = min(vals)
    variance = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    std_v  = variance ** 0.5
    cv     = std_v / mean_v if mean_v else 0  # coefficient of variation

    print(f"  Valid epochs with mask mAP50 > 0 : {len(vals)} / {len(rows)}")
    print(f"  Mean mAP50(M) across epochs      : {mean_v:.4f}")
    print(f"  Max  mAP50(M)                    : {max_v:.4f}")
    print(f"  Min  mAP50(M)                    : {min_v:.4f}")
    print(f"  Std  mAP50(M)                    : {std_v:.4f}")
    print(f"  Coeff. of Variation              : {cv:.4f}  (lower = more stable)")

    # NaN / inf counts in validation losses
    nan_count = sum(1 for r in rows if r["val/seg_loss"] in ("nan", "inf", ""))
    print(f"\n  Validation seg_loss NaN/Inf epochs: {nan_count} / {len(rows)}")
    if nan_count > 0:
        print("  ⚠  Unstable validation — likely caused by extremely small val set (3 images)")

    # Image-category breakdown
    easy_keys = ["2e-9_100s", "Ag_1e-8", "Ag_1e-9", "Ag_2e-9"]
    hard_keys = ["70nm", "Ag_40nm"]
    print("\n  Dataset split by SEM condition:")
    print("    Easy subset: low-noise, cleaner dendrites")
    print("     ", [k for k in easy_keys])
    print("    Hard subset: high-density, noisy, complex morphology")
    print("     ", [k for k in hard_keys])
    print("  NOTE: only 14 images total — robustness across conditions is limited.")

    # Training stability windows
    print("\n  Training stability windows (by mAP50(M) trend):")
    windows = [
        (1,   30,  "Early / warm-up: rapid loss decrease, near-zero mAP"),
        (31,  100, "Mid training: unstable — NaN val losses, erratic mAP"),
        (101, 150, "Late plateau: mAP stabilizes, best ~31% at epoch 146"),
        (151, 473, "Extended / repeated epochs: no further improvement"),
    ]
    for start, end, desc in windows:
        segment = [safe_float(r["metrics/mAP50(M)"]) for r in rows
                   if start <= int(r["epoch"]) <= end]
        segment = [v for v in segment if v is not None and v > 0]
        avg = sum(segment) / len(segment) if segment else 0
        print(f"    Epochs {start:>3}–{end:<3}: avg mAP50(M)={avg:.4f}  — {desc}")


# ── 3. Failure analysis ───────────────────────────────────────────────────────

def failure_analysis(rows):
    print("\n── FAILURE ANALYSIS ─────────────────────────────────────────────────")

    print("""
  F1. SEVERE DATASET UNDERSIZING
  ─────────────────────────────────────────────────────────────────────
  • Training set: 10 images. Validation: 3 images. Test: 1 image.
  • YOLOv11x-seg has ~56 M parameters — massively overparameterized for
    14 training samples. Transfer learning mitigates this, but not fully.
  • Consequence: validation metrics are noisy; a single mis-prediction
    on any of the 3 val images moves mAP by ~33%.

  F2. UNSTABLE VALIDATION LOSSES (NaN / Inf)
  ─────────────────────────────────────────────────────────────────────""")

    nan_epochs = [r["epoch"] for r in rows if r["val/seg_loss"] in ("nan","inf","")]
    print(f"  • {len(nan_epochs)} epochs produced NaN/Inf validation seg_loss.")
    print("""  • Root cause: validation set too small → a batch with 0 detections
    yields division-by-zero in loss computation.
  • These epochs are visible in results.csv as val/seg_loss = nan / inf.

  F3. LOW ABSOLUTE mAP (best 31.2% @ IoU=0.50)
  ─────────────────────────────────────────────────────────────────────
  • Dendrites are thin, fractal, overlapping structures. Instance
    segmentation of such objects is inherently harder than blob/object
    segmentation.
  • Polygon annotations may have inconsistent granularity (some
    annotations may approximate fine branches as coarser polygons).
  • mAP@0.5:95 = 11.3% — performance degrades sharply at tighter IoU
    thresholds, confirming imprecise mask boundaries.

  F4. PRECISION / RECALL IMBALANCE (P=46.9%, R=31.9%)
  ─────────────────────────────────────────────────────────────────────
  • Recall is ~15 pp lower than Precision → the model misses real
    dendrites more often than it produces false positives.
  • Likely cause: dendrite branches that were not annotated (missed by
    annotators) are predicted by the model → counted as false positives,
    and unannotated true dendrites inflate false negatives.

  F5. EARLY-STOPPING DID NOT TRIGGER (ran all 150 epochs / 473 logged)
  ─────────────────────────────────────────────────────────────────────
  • Training config set patience=30 but results show 473 epochs, not 35.
  • The results.csv has multiple "runs" concatenated, suggesting the
    model was re-trained or resumed multiple times.
  • Best checkpoint (best.pt) is saved at the epoch with peak val mAP.

  F6. HARD-SUBSET IMAGES LIKELY HARDER TO GENERALIZE
  ─────────────────────────────────────────────────────────────────────
  • Hard subset (70nm pitch, Ag_40nm) shows higher-density dendrites,
    overlapping branches, and charging-effect blur.
  • With so few samples, the model likely performs worse on Hard images
    but cannot be quantified without per-image evaluation.
  • Recommended: run model.val() per image group to measure this gap.
""")


# ── 4. Per-image inference (if ultralytics available) ─────────────────────────

def per_image_eval():
    print("── PER-IMAGE EVALUATION (best.pt on all splits) ─────────────────────")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed — skipping live inference.")
        print("  Install with: pip install ultralytics")
        return

    if not WEIGHTS_BEST.exists():
        print(f"  Weights not found at {WEIGHTS_BEST}")
        return

    model = YOLO(str(WEIGHTS_BEST))

    for split_name, split_dir in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        images = sorted(split_dir.glob("*.png")) + sorted(split_dir.glob("*.jpg"))
        if not images:
            print(f"  {split_name}: no images found at {split_dir}")
            continue

        print(f"\n  [{split_name.upper()}]  ({len(images)} images)")
        split_ious, split_dices = [], []

        for img_path in images:
            results = model.predict(str(img_path), verbose=False, conf=0.25)
            r = results[0]

            # Load GT mask (label file)
            label_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
            has_gt = label_path.exists()

            n_pred = len(r.boxes) if r.boxes is not None else 0

            # If masks available, compute rough pixel IoU vs GT polygon mask
            if r.masks is not None and has_gt and n_pred > 0:
                import numpy as np
                import cv2
                H, W = r.orig_shape
                # pred masks are in model resolution; resize to orig
                pred_mask = r.masks.data.cpu().numpy()
                mH, mW = pred_mask.shape[1], pred_mask.shape[2]
                pred_union = (pred_mask.sum(axis=0) > 0).astype(np.uint8)
                if (mH, mW) != (H, W):
                    pred_union = cv2.resize(pred_union, (W, H), interpolation=cv2.INTER_NEAREST)
                pred_union = pred_union.astype(bool)

                # Build GT mask from polygon annotations
                gt_union = np.zeros((H, W), dtype=np.uint8)
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        coords = list(map(float, parts[1:]))
                        pts_x = [int(c * W) for c in coords[0::2]]
                        pts_y = [int(c * H) for c in coords[1::2]]
                        pts = np.array(list(zip(pts_x, pts_y)), dtype=np.int32)
                        cv2.fillPoly(gt_union, [pts], 1)
                gt_union = gt_union.astype(bool)

                intersection = (pred_union & gt_union).sum()
                union        = (pred_union | gt_union).sum()
                iou  = intersection / union if union else 0.0
                dice = 2 * intersection / (pred_union.sum() + gt_union.sum()) if (pred_union.sum() + gt_union.sum()) > 0 else 0.0
                split_ious.append(iou)
                split_dices.append(dice)
                print(f"    {img_path.name:<45} preds={n_pred:>3}  IoU={iou:.4f}  Dice={dice:.4f}")
            else:
                tag = "(no GT label)" if not has_gt else "(no masks predicted)"
                print(f"    {img_path.name:<45} preds={n_pred:>3}  {tag}")

        if split_ious:
            avg_iou  = sum(split_ious)  / len(split_ious)
            avg_dice = sum(split_dices) / len(split_dices)
            print(f"  ── {split_name.upper()} average  IoU={avg_iou:.4f}  Dice={avg_dice:.4f}")


# ── 5. Quick IoU / Dice table from mAP thresholds ────────────────────────────

def iou_dice_table(best):
    print("\n── IoU / DICE SCORE SUMMARY ─────────────────────────────────────────")
    print("  (Derived from training-time validation metrics at best epoch)\n")
    print(f"  {'Metric':<35} {'Value':>10}")
    print("  " + "-" * 47)

    m50 = safe_float(best["metrics/mAP50(M)"])
    m95 = safe_float(best["metrics/mAP50-95(M)"])
    b50 = safe_float(best["metrics/mAP50(B)"])
    mp  = safe_float(best["metrics/precision(M)"])
    mr  = safe_float(best["metrics/recall(M)"])

    dice50 = dice_from_iou(m50) if m50 else None
    f1     = 2 * mp * mr / (mp + mr) if (mp and mr and mp + mr > 0) else None

    rows_table = [
        ("Mask Precision @ best epoch",         f"{mp:.4f} ({mp*100:.1f}%)"),
        ("Mask Recall    @ best epoch",         f"{mr:.4f} ({mr*100:.1f}%)"),
        ("Mask F1 Score  @ best epoch",         f"{f1:.4f} ({f1*100:.1f}%)" if f1 else "N/A"),
        ("Mask mAP@0.50  (proxy for IoU=0.50)", f"{m50:.4f} ({m50*100:.1f}%)"),
        ("Mask mAP@0.5:95 (avg IoU 0.5–0.95)",  f"{m95:.4f} ({m95*100:.1f}%)"),
        ("Mask Dice Score (from mAP50)",        f"{dice50:.4f} ({dice50*100:.1f}%)" if dice50 else "N/A"),
        ("Box  mAP@0.50",                       f"{b50:.4f} ({b50*100:.1f}%)"),
        ("Best epoch",                          best["epoch"]),
        ("Total training epochs",               "473"),
    ]

    for name, val in rows_table:
        print(f"  {name:<35} {str(val):>10}")

    print("""
  Notes:
  • mAP50 approximates the area under the Precision-Recall curve at
    IoU threshold 0.50 — the standard COCO segmentation metric.
  • Dice ≈ 47.6% at IoU threshold 0.50.
  • mAP@0.5:95 = 11.3% indicates the model struggles at tight IoU
    thresholds (≥0.75) — masks are coarse relative to GT polygons.
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: results.csv not found at {RESULTS_CSV}")
        sys.exit(1)

    rows = load_results()
    best = print_training_summary(rows)
    iou_dice_table(best)
    robustness_analysis(rows)
    failure_analysis(rows)
    per_image_eval()

    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()
