"""
YOLO-Seg pipeline for SEM dendrite segmentation.

Uses ultralytics YOLOv11 instance segmentation with transfer learning
from COCO pretrained weights.  Provides dataset preparation, training,
inference (single and batch), and skeleton extraction.

Usage:
    python yolo_pipeline.py train --data yolo_dataset/data.yaml
    python yolo_pipeline.py predict --model best.pt --source data/raw/Easy
    python yolo_pipeline.py                          # synthetic self-test
"""

import argparse
import cv2
import numpy as np
import os
import sys

from skimage.morphology import skeletonize

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_image, save_image, list_images

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "yolo11n-seg.pt"    # Nano model for fast training
DEFAULT_EPOCHS = 100                # max epochs (early stopping may cut short)
DEFAULT_IMGSZ = 640                 # square input size for YOLO
DEFAULT_BATCH = 8                   # batch size (reduce if GPU OOM)
DEFAULT_PATIENCE = 20               # early stopping: epochs without improvement
DEFAULT_FREEZE = 10                 # freeze first N backbone layers for transfer learning
DEFAULT_LR0 = 0.001                 # initial learning rate
DEFAULT_CONF = 0.25                 # inference confidence threshold
DEFAULT_WORKERS = 0 if sys.platform == "win32" else 8  # dataloader workers


# ===========================================================================
# Dataset preparation
# ===========================================================================

def normalize_dataset_images(dataset_dir):
    """
    Convert all dataset images to 3-channel PNG format.

    YOLO pretrained models expect 3-channel (BGR) input.  SEM images are
    typically grayscale TIF files.  This function:
      1. Converts grayscale/RGBA images to 3-channel BGR
      2. Rewrites non-PNG files as PNG (avoids TIFF metadata issues)
      3. Clears Ultralytics label cache files to force re-indexing

    Parameters
    ----------
    dataset_dir : str
        Root of the YOLO dataset (containing train/valid/test subdirs).
    """
    converted = 0
    total = 0

    for split in ("train", "valid", "test"):
        img_dir = os.path.join(dataset_dir, split, "images")
        if not os.path.isdir(img_dir):
            continue

        for fname in sorted(os.listdir(img_dir)):
            fpath = os.path.join(img_dir, fname)
            if not os.path.isfile(fpath):
                continue

            img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            total += 1

            # Ensure 3 channels
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # Rewrite to PNG if not already
            stem = os.path.splitext(fname)[0]
            ext = os.path.splitext(fname)[1].lower()
            if ext != ".png":
                dst = os.path.join(img_dir, stem + ".png")
                cv2.imwrite(dst, img)
                if os.path.normcase(dst) != os.path.normcase(fpath):
                    os.remove(fpath)
                converted += 1
            else:
                cv2.imwrite(fpath, img)

        # Remove stale cache so Ultralytics re-indexes
        cache = os.path.join(dataset_dir, split, "labels.cache")
        if os.path.isfile(cache):
            os.remove(cache)

    if total > 0:
        print(f"  Dataset normalized: {converted}/{total} images converted to 3-ch PNG")


def validate_dataset(dataset_dir):
    """
    Validate YOLO dataset structure and return path to data.yaml.

    Checks that train/images, train/labels, valid/images, valid/labels
    exist and contain matching file counts.  Normalizes images to 3-channel
    PNG for compatibility with pretrained YOLO backbones.

    Parameters
    ----------
    dataset_dir : str
        Root of the YOLO dataset directory.

    Returns
    -------
    yaml_path : str
        Path to the dataset's data.yaml file.
    """
    for subdir in ("train/images", "train/labels", "valid/images", "valid/labels"):
        full = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(full):
            raise FileNotFoundError(
                f"Missing required directory: {full}\n"
                f"Dataset must have train/ and valid/ splits with images/ and labels/."
            )

    # Count files per split
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        n_img = len([f for f in os.listdir(img_dir) if not f.startswith(".")])
        n_lbl = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt")]) if os.path.isdir(lbl_dir) else 0
        print(f"  {split}: {n_img} images, {n_lbl} labels")
        if n_img != n_lbl and n_lbl > 0:
            print(f"  WARNING: image/label count mismatch in {split}")

    # Normalize images to 3-channel PNG
    normalize_dataset_images(dataset_dir)

    # Verify data.yaml exists
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        # Auto-generate minimal data.yaml
        content = (
            f"path: {os.path.abspath(dataset_dir)}\n"
            f"train: train/images\n"
            f"val: valid/images\n"
            f"\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: dendrite\n"
        )
        with open(yaml_path, "w") as f:
            f.write(content)
        print(f"  Created data.yaml at {yaml_path}")
    else:
        print(f"  Using existing data.yaml: {yaml_path}")

    return yaml_path


# ===========================================================================
# Training
# ===========================================================================

def train_model(dataset_yaml, model=DEFAULT_MODEL, epochs=DEFAULT_EPOCHS,
                imgsz=DEFAULT_IMGSZ, batch=DEFAULT_BATCH,
                patience=DEFAULT_PATIENCE, freeze=DEFAULT_FREEZE,
                lr0=DEFAULT_LR0, workers=DEFAULT_WORKERS, project=None):
    """
    Fine-tune a YOLO-Seg model on the SEM dendrite dataset.

    Uses transfer learning: freezes the first N backbone layers and
    trains the detection/segmentation heads with a low learning rate.
    Early stopping halts training if validation loss plateaus.

    Parameters
    ----------
    dataset_yaml : str
        Path to the dataset's data.yaml file.
    model : str
        Pretrained YOLO model name or path to .pt weights.
    epochs : int
        Maximum training epochs.
    imgsz : int
        Input image size (images are resized to imgsz x imgsz).
    batch : int
        Training batch size.
    patience : int
        Early stopping patience (epochs without val improvement).
    freeze : int
        Number of backbone layers to freeze during training.
    lr0 : float
        Initial learning rate.
    workers : int
        Number of dataloader workers (0 for single-process on Windows).
    project : str or None
        Output directory for training artifacts.

    Returns
    -------
    results : ultralytics.engine.results.Results
        Training results object with metrics and best weights path.
    """
    from ultralytics import YOLO

    if project is None:
        project = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "output", "yolo", "train")

    print(f"\n{'=' * 60}")
    print(f"Training YOLO-Seg Model")
    print(f"  Base model:  {model}")
    print(f"  Dataset:     {dataset_yaml}")
    print(f"  Epochs:      {epochs}")
    print(f"  Image size:  {imgsz}")
    print(f"  Batch:       {batch}")
    print(f"  Patience:    {patience}")
    print(f"  Freeze:      {freeze} layers")
    print(f"  LR:          {lr0}")
    print(f"  Workers:     {workers}")
    print(f"{'=' * 60}\n")

    yolo = YOLO(model)
    results = yolo.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        freeze=freeze,
        lr0=lr0,
        workers=workers,
        project=project,
        name="dendrite_seg",
        exist_ok=True,
        verbose=True,
    )

    best_weights = os.path.join(project, "dendrite_seg", "weights", "best.pt")
    print(f"\nTraining complete. Best weights saved to: {best_weights}")
    return results


# ===========================================================================
# Inference
# ===========================================================================

def extract_mask(model, image_path, conf=DEFAULT_CONF):
    """
    Run YOLO-Seg on a single image and merge all instance masks.

    All detected dendrite instances are combined via logical OR into one
    binary mask covering all dendrite pixels.

    Parameters
    ----------
    model : ultralytics.YOLO
        Loaded YOLO model (pass a pre-loaded model for batch efficiency).
    image_path : str
        Path to the input SEM image.
    conf : float
        Minimum detection confidence threshold.

    Returns
    -------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8, same size as input image.
    """
    preds = model.predict(image_path, conf=conf, verbose=False)

    # Read original image dimensions
    orig = cv2.imread(image_path)
    h, w = orig.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    if preds and preds[0].masks is not None:
        # masks.data is a tensor of shape (N_instances, mask_h, mask_w)
        for inst in preds[0].masks.data.cpu().numpy():
            resized = cv2.resize(inst, (w, h), interpolation=cv2.INTER_LINEAR)
            mask[resized > 0.5] = 255

    return mask


def extract_skeleton(mask):
    """
    Compute single-pixel-width skeleton from a binary mask.

    Uses Zhang-Suen thinning (scikit-image) to extract centerlines.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    skeleton : np.ndarray
        Skeleton image (0 or 255), dtype uint8.
    """
    binary = (mask > 0).astype(bool)
    skel = skeletonize(binary)
    return (skel.astype(np.uint8) * 255)


def run_yolo_pipeline(model_path, image_path, output_dir=None, conf=DEFAULT_CONF):
    """
    Full YOLO inference pipeline on a single image: mask + skeleton + overlays.

    Parallels run_classic_pipeline() so both can be compared side-by-side.

    Parameters
    ----------
    model_path : str
        Path to trained YOLO weights (.pt file).
    image_path : str
        Path to input SEM image.
    output_dir : str or None
        Directory to save results. If None, results are not saved to disk.
    conf : float
        Detection confidence threshold.

    Returns
    -------
    results : dict
        Dictionary with 'mask', 'skeleton', and 'intermediates' keys.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"YOLO predicting: {basename}")

    # Extract mask and skeleton
    mask = extract_mask(model, image_path, conf=conf)
    skeleton = extract_skeleton(mask)

    fg_pixels = int(np.sum(mask > 0))
    skel_pixels = int(np.sum(skeleton > 0))
    print(f"  Mask: {fg_pixels} fg pixels, Skeleton: {skel_pixels} pixels")

    # Save outputs
    if output_dir:
        img_dir = os.path.join(output_dir, basename)
        os.makedirs(img_dir, exist_ok=True)
        save_image(mask, os.path.join(img_dir, "yolo_mask.png"))
        save_image(skeleton, os.path.join(img_dir, "yolo_skeleton.png"))

        # Generate overlay: skeleton in red on original
        orig = load_image(image_path, grayscale=True)
        overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
        overlay[skeleton > 0] = (0, 0, 255)
        save_image(overlay, os.path.join(img_dir, "yolo_overlay.png"))

    return {
        "mask": mask,
        "skeleton": skeleton,
        "intermediates": {"yolo_mask": mask, "yolo_skeleton": skeleton},
    }


def predict_directory(model_path, input_dir, output_dir, conf=DEFAULT_CONF):
    """
    Run YOLO-Seg inference on all images in a directory.

    Loads the model once and reuses it across all images for efficiency.

    Parameters
    ----------
    model_path : str
        Path to trained YOLO weights (.pt file).
    input_dir : str
        Directory of input SEM images.
    output_dir : str
        Directory to save masks, skeletons, and overlays.
    conf : float
        Detection confidence threshold.

    Returns
    -------
    all_results : dict
        Mapping of image basename to results dict.
    """
    from ultralytics import YOLO

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return {}

    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    print(f"YOLO batch inference on {len(image_paths)} images...\n")

    for path in image_paths:
        basename = os.path.splitext(os.path.basename(path))[0]

        mask = extract_mask(model, path, conf=conf)
        skeleton = extract_skeleton(mask)

        # Save mask and skeleton
        save_image(mask, os.path.join(output_dir, f"{basename}_mask.png"))
        save_image(skeleton, os.path.join(output_dir, f"{basename}_skeleton.png"))

        fg = int(np.sum(mask > 0))
        print(f"  {basename}: {fg} fg pixels")
        all_results[basename] = {"mask": mask, "skeleton": skeleton}

    print(f"\nSaved {len(all_results)} results to {output_dir}/")
    return all_results


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    """Command-line interface for the YOLO-Seg pipeline."""
    parser = argparse.ArgumentParser(
        description="YOLO-Seg pipeline for SEM dendrite segmentation"
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    tp = sub.add_parser("train", help="Train YOLO-Seg model on labeled dataset")
    tp.add_argument("--data", required=True,
                    help="Path to data.yaml (or dataset root directory)")
    tp.add_argument("--model", default=DEFAULT_MODEL)
    tp.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    tp.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    tp.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    tp.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    tp.add_argument("--freeze", type=int, default=DEFAULT_FREEZE)
    tp.add_argument("--lr0", type=float, default=DEFAULT_LR0)
    tp.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    tp.add_argument("--project", default=None, help="Output project directory")

    # --- predict ---
    pp = sub.add_parser("predict", help="Run inference on image(s)")
    pp.add_argument("--model", required=True, help="Path to trained .pt weights")
    pp.add_argument("--source", required=True, help="Image file or directory")
    pp.add_argument("--output", default=None, help="Output directory")
    pp.add_argument("--conf", type=float, default=DEFAULT_CONF)

    args = parser.parse_args()

    if args.command == "train":
        # Determine dataset root from data.yaml path
        data_path = args.data
        if os.path.isfile(data_path):
            dataset_dir = os.path.dirname(data_path)
        else:
            dataset_dir = data_path
            data_path = os.path.join(dataset_dir, "data.yaml")

        yaml_path = validate_dataset(dataset_dir)
        train_model(
            yaml_path,
            model=args.model, epochs=args.epochs, imgsz=args.imgsz,
            batch=args.batch, patience=args.patience, freeze=args.freeze,
            lr0=args.lr0, workers=args.workers, project=args.project,
        )

    elif args.command == "predict":
        if not os.path.isfile(args.model):
            print(f"Error: weights not found: {args.model}")
            sys.exit(1)

        out = args.output or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "yolo"
        )

        if os.path.isdir(args.source):
            predict_directory(args.model, args.source, out, conf=args.conf)
        elif os.path.isfile(args.source):
            run_yolo_pipeline(args.model, args.source, out, conf=args.conf)
        else:
            print(f"Error: source not found: {args.source}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Synthetic self-test (no ultralytics required)
        print("=== yolo_pipeline.py — Synthetic Self-Test ===\n")

        # Build a temporary dataset structure
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "output", "_yolo_selftest")
        for sub in ("train/images", "train/labels", "valid/images", "valid/labels"):
            os.makedirs(os.path.join(test_dir, sub), exist_ok=True)

        # Create minimal dummy data
        dummy = np.zeros((64, 64), dtype=np.uint8)
        for i in range(3):
            cv2.imwrite(os.path.join(test_dir, "train/images", f"d{i}.png"), dummy)
            with open(os.path.join(test_dir, "train/labels", f"d{i}.txt"), "w") as f:
                f.write("0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
        cv2.imwrite(os.path.join(test_dir, "valid/images", "d0.png"), dummy)
        with open(os.path.join(test_dir, "valid/labels", "d0.txt"), "w") as f:
            f.write("0 0.3 0.3 0.5 0.3 0.5 0.5 0.3 0.5\n")

        yaml_path = validate_dataset(test_dir)
        print(f"\nValidated dataset, yaml at: {yaml_path}")

        # Test skeleton extraction
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(test_mask, (50, 10), (50, 90), 255, 6)
        cv2.line(test_mask, (50, 50), (90, 50), 255, 4)
        skel = extract_skeleton(test_mask)
        print(f"Skeleton test: mask={np.sum(test_mask > 0)} px, "
              f"skel={np.sum(skel > 0)} px")

        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        print("\nAll YOLO pipeline tests passed.")
