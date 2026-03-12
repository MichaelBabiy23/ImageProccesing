"""
prepare_and_train.py
====================
Prepares a YOLO-seg dataset from the raw SEM images and trains a YOLOv11
instance-segmentation model.

Dataset layout expected in raw/:
    raw/
        Easy/
            <image>.tif  ...
            annotations/
                <image>.txt  ...   (YOLO polygon format: class x1 y1 x2 y2 ...)
        Hard/
            <image>.tif  ...
            annotations/
                <image>.txt  ...

This script:
  1. Reads all images + labels from both Easy and Hard subsets.
  2. Converts .tif images to 3-channel PNG (required by YOLO pretrained models).
  3. Splits the combined set into train / val / test  (70 / 20 / 10 %).
  4. Writes the YOLO dataset folder structure under yolo_dataset/.
  5. Writes data.yaml.
  6. Launches YOLO training.

Nothing in raw/ is ever modified or deleted.
Run:
    python prepare_and_train.py [options]
    python prepare_and_train.py --help
"""

import argparse
import os
import random
import shutil
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)          # "final project/"

def _find_raw_dir() -> str:
    """
    Locate data/raw/ robustly.

    Normal case:   <project_dir>/data/raw/   exists directly.
    Worktree case: the project is cloned into
                     <main_repo>/.claude/worktrees/<name>/<project_dir>/
                   so raw data (gitignored) is in the main repo.
                   We walk up the directory tree looking for a
                   data/raw/ folder that actually contains Easy/ or Hard/.
    """
    def has_subsets(path):
        """True only if the raw dir has at least one subset with an annotations folder."""
        if not os.path.isdir(path):
            return False
        for s in ("Easy", "Hard"):
            ann = os.path.join(path, s, "annotations")
            if os.path.isdir(ann) and os.listdir(ann):
                return True
        return False

    # Walk upward from SCRIPT_DIR through up to 8 parent directories
    probe = SCRIPT_DIR
    for _ in range(8):
        candidate = os.path.join(probe, "data", "raw")
        if has_subsets(candidate):
            return candidate
        probe = os.path.dirname(probe)

    # Nothing found — return the naive path; error message will be clear
    return os.path.join(PROJECT_DIR, "data", "raw")

RAW_DIR = _find_raw_dir()
DATASET_DIR = os.path.join(SCRIPT_DIR, "yolo_dataset")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "runs")

# ---------------------------------------------------------------------------
# Training defaults  (override via CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_MODEL    = "yolo11x-seg.pt"
DEFAULT_EPOCHS   = 150
DEFAULT_IMGSZ    = 640
DEFAULT_BATCH    = 4          # small; only ~14 images in the dataset
DEFAULT_PATIENCE = 30
DEFAULT_FREEZE   = 10
DEFAULT_LR0      = 0.001
DEFAULT_SEED     = None   # None = truly random each run
DEFAULT_SPLITS   = (0.70, 0.20, 0.10)   # train / val / test

SUBSETS = ["Easy", "Hard"]   # subfolders inside raw/


# ===========================================================================
# Step 1 – collect samples
# ===========================================================================

def collect_samples(raw_dir: str) -> list[tuple[str, str]]:
    """
    Return a list of (image_path, label_path) pairs from all subsets.

    Only pairs where *both* the image and the annotation file exist are
    included.  Missing annotations are reported as warnings.
    """
    samples = []
    for subset in SUBSETS:
        subset_dir  = os.path.join(raw_dir, subset)
        ann_dir     = os.path.join(subset_dir, "annotations")

        if not os.path.isdir(subset_dir):
            print(f"  WARNING: subset directory not found: {subset_dir}")
            continue

        for fname in sorted(os.listdir(subset_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                continue

            img_path = os.path.join(subset_dir, fname)
            lbl_path = os.path.join(ann_dir, stem + ".txt")

            if not os.path.isfile(lbl_path):
                print(f"  WARNING: no annotation for {fname}, skipping")
                continue

            samples.append((img_path, lbl_path))

    return samples


# ===========================================================================
# Step 2 – split
# ===========================================================================

def split_samples(
    samples: list,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list, list, list]:
    """Shuffle and split samples into train / val / test lists."""
    rng = random.Random(seed)  # seed=None uses system time → truly random
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, round(n * train_frac))
    n_val   = max(1, round(n * val_frac))
    # remaining goes to test (may be 0 for tiny datasets)
    n_test  = max(0, n - n_train - n_val)

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    return train, val, test


# ===========================================================================
# Step 3 – write dataset
# ===========================================================================

def write_split(samples: list, split_name: str, dataset_dir: str) -> None:
    """
    Copy images (converted to 3-ch PNG) and labels into
    <dataset_dir>/<split_name>/images/ and .../labels/.
    """
    img_out = os.path.join(dataset_dir, split_name, "images")
    lbl_out = os.path.join(dataset_dir, split_name, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for img_src, lbl_src in samples:
        stem = os.path.splitext(os.path.basename(img_src))[0]
        dst_img = os.path.join(img_out, stem + ".png")
        dst_lbl = os.path.join(lbl_out, stem + ".txt")

        # --- convert image to 3-channel PNG ---
        raw = cv2.imread(img_src, cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"  WARNING: could not read {img_src}, skipping")
            continue

        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.ndim == 3 and raw.shape[2] == 1:
            raw = cv2.cvtColor(raw[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif raw.ndim == 3 and raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        # Normalise to 8-bit if 16-bit TIF
        if raw.dtype != np.uint8:
            raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        cv2.imwrite(dst_img, raw)

        # --- copy label as-is ---
        shutil.copy2(lbl_src, dst_lbl)


def write_data_yaml(dataset_dir: str) -> str:
    """Write data.yaml and return its path."""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    abs_dataset = os.path.abspath(dataset_dir).replace("\\", "/")

    content = (
        f"path: {abs_dataset}\n"
        f"train: train/images\n"
        f"val:   valid/images\n"
        f"test:  test/images\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: dendrite\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    return yaml_path


def prepare_dataset(
    raw_dir:    str,
    dataset_dir: str,
    train_frac: float,
    val_frac:   float,
    test_frac:  float,
    seed:       int,
) -> str:
    """Full dataset preparation pipeline. Returns path to data.yaml."""
    print("\n[1/3] Collecting samples from raw data …")
    samples = collect_samples(raw_dir)
    if not samples:
        sys.exit("ERROR: No valid image/annotation pairs found in raw data.")

    print(f"      Found {len(samples)} image-annotation pairs")
    for img, _ in samples:
        print(f"        {os.path.relpath(img, raw_dir)}")

    print(f"\n[2/3] Splitting {len(samples)} samples "
          f"({train_frac:.0%} / {val_frac:.0%} / {test_frac:.0%}) …")
    train, val, test = split_samples(samples, train_frac, val_frac, test_frac, seed)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    print(f"\n[3/3] Writing YOLO dataset to: {dataset_dir}")
    # Wipe and recreate to keep it fresh on re-runs
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    write_split(train, "train",  dataset_dir)
    write_split(val,   "valid",  dataset_dir)
    write_split(test,  "test",   dataset_dir)

    # Remove stale label caches
    for split in ("train", "valid", "test"):
        cache = os.path.join(dataset_dir, split, "labels.cache")
        if os.path.isfile(cache):
            os.remove(cache)

    yaml_path = write_data_yaml(dataset_dir)
    print(f"      data.yaml written: {yaml_path}")
    return yaml_path


# ===========================================================================
# Step 4 – train
# ===========================================================================

def detect_device() -> str:
    """
    Return the best available device string for YOLO.

    Priority: CUDA GPU > CPU.
    Prints a clear one-line status so the user knows what is being used.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Device     : GPU — {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return "0"   # first CUDA device
        else:
            print("  Device     : CPU  (no CUDA GPU detected — training will be slow)")
            return "cpu"
    except ImportError:
        print("  Device     : CPU  (torch not importable — install pytorch with CUDA for GPU)")
        return "cpu"


def train(
    yaml_path:  str,
    model:      str,
    epochs:     int,
    imgsz:      int,
    batch:      int,
    patience:   int,
    freeze:     int,
    lr0:        float,
    output_dir: str,
) -> None:
    """Launch YOLO training."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ERROR: ultralytics is not installed.\n"
            "Run:  pip install ultralytics"
        )

    # Windows: dataloader workers must be 0 (multiprocessing spawn issues)
    workers = 0
    device  = detect_device()

    print(f"\n{'=' * 60}")
    print(f"  YOLO-Seg Training")
    print(f"  Base model : {model}")
    print(f"  Dataset    : {yaml_path}")
    print(f"  Epochs     : {epochs}  (patience={patience})")
    print(f"  Img size   : {imgsz}")
    print(f"  Batch      : {batch}")
    print(f"  Freeze     : {freeze} backbone layers")
    print(f"  LR0        : {lr0}")
    print(f"  Workers    : {workers}  (Windows requires 0)")
    print(f"  Output     : {output_dir}")
    print(f"{'=' * 60}\n")

    yolo = YOLO(model)
    yolo.train(
        data      = yaml_path,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        patience  = patience,
        freeze    = freeze,
        lr0       = lr0,
        workers   = workers,
        device    = device,   # GPU "0" or "cpu"
        project   = output_dir,
        name      = "dendrite_seg",
        exist_ok  = True,
        verbose   = True,
        # Augmentations tuned for SEM (greyscale, small dataset)
        hsv_h     = 0.0,    # no hue shift (greyscale)
        hsv_s     = 0.0,    # no saturation shift
        hsv_v     = 0.3,    # brightness variation
        fliplr    = 0.5,
        flipud    = 0.5,
        mosaic    = 0.5,    # mosaic helps with small datasets
        copy_paste= 0.2,
    )

    best = os.path.join(output_dir, "dendrite_seg", "weights", "best.pt")
    if os.path.isfile(best):
        print(f"\nDone. Best weights: {best}")
    else:
        print("\nDone. Check runs/ for weights.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    """Parse CLI arguments for dataset preparation and YOLO training."""
    p = argparse.ArgumentParser(
        description="Prepare dataset from raw/ and train YOLO-seg dendrite model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw",      default=RAW_DIR,     help="Path to raw data directory")
    p.add_argument("--dataset",  default=DATASET_DIR, help="Output YOLO dataset directory")
    p.add_argument("--output",   default=OUTPUT_DIR,  help="Training output (runs) directory")
    p.add_argument("--model",    default=DEFAULT_MODEL)
    p.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz",    type=int,   default=DEFAULT_IMGSZ)
    p.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    p.add_argument("--patience", type=int,   default=DEFAULT_PATIENCE)
    p.add_argument("--freeze",   type=int,   default=DEFAULT_FREEZE)
    p.add_argument("--lr0",      type=float, default=DEFAULT_LR0)
    p.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    p.add_argument(
        "--splits", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
        default=list(DEFAULT_SPLITS),
        help="Dataset split fractions (must sum to 1.0)",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Only prepare the dataset, do not run training",
    )
    return p.parse_args()


def main():
    """Entry point: prepare dataset from raw data, then optionally train YOLO."""
    args = parse_args()

    train_f, val_f, test_f = args.splits
    if abs(train_f + val_f + test_f - 1.0) > 1e-6:
        sys.exit(f"ERROR: split fractions must sum to 1.0, got {train_f+val_f+test_f}")

    yaml_path = prepare_dataset(
        raw_dir    = args.raw,
        dataset_dir= args.dataset,
        train_frac = train_f,
        val_frac   = val_f,
        test_frac  = test_f,
        seed       = args.seed,
    )

    if args.skip_train:
        print("\n--skip-train set, exiting after dataset preparation.")
        return

    train(
        yaml_path  = yaml_path,
        model      = args.model,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        patience   = args.patience,
        freeze     = args.freeze,
        lr0        = args.lr0,
        output_dir = args.output,
    )


if __name__ == "__main__":
    main()
