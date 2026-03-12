"""
Apply a pipeline_gui preset to a single SEM image and compare with classic output.

Reads a JSON preset (exported from pipeline_gui.py) containing all tunable
parameters and skip flags, runs the full classic pipeline with those settings
on one image, saves every intermediate stage, and stitches each stage
side-by-side against the existing classic pipeline output for visual comparison.

Usage:
    python run_preset_single_image.py --image data/raw/Easy/img.tif --category Easy
    python run_preset_single_image.py --image img.tif --category Hard --preset my_preset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import clean_sem_image, create_comparison_strip, create_overlay, load_image, save_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for preset pipeline execution."""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run one preset image and stitch it against classic output.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--category", required=True, help="Category name such as Easy or Hard")
    parser.add_argument(
        "--preset",
        default=str(project_dir / "Adjusments" / "Remove_Unused_Functions.json"),
        help="Preset JSON exported from pipeline_gui.py",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_dir / "output" / "classic_remove_unused_functions_side_by_side"),
        help="Root output directory",
    )
    parser.add_argument(
        "--classic-root",
        default=str(project_dir / "output" / "classic"),
        help="Existing classic output root",
    )
    return parser.parse_args()


def run_preset_pipeline(image: np.ndarray, params: dict, skips: dict) -> dict[str, np.ndarray]:
    """
    Run the full classic pipeline using preset parameters and skip flags.

    Mirrors the stages in classic_pipeline.py but uses parameter values from
    a GUI-exported JSON preset instead of the module-level constants. Each
    stage can be individually skipped via the skips dict.

    Parameters
    ----------
    image : np.ndarray
        Raw grayscale SEM image (H, W).
    params : dict
        Pipeline parameter values (e.g. CLAHE_CLIP_LIMIT, ADAPTIVE_BLOCK_SIZE).
    skips : dict
        Boolean flags keyed by stage name (e.g. 'clean', 'clahe', 'bilateral').

    Returns
    -------
    intermediates : dict of str to np.ndarray
        All intermediate images keyed by stage name (01_original through
        10_skeleton plus overlay images).
    """
    def skip(name: str) -> bool:
        return bool(skips.get(name, False))

    cleaned = clean_sem_image(image) if not skip("clean") else image.copy()

    if not skip("normalize"):
        mn, mx = float(cleaned.min()), float(cleaned.max())
        normalized = (
            ((cleaned.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
            if mx > mn else np.zeros_like(cleaned)
        )
    else:
        normalized = cleaned.copy()

    if not skip("clahe"):
        tile = int(params["CLAHE_TILE_SIZE"])
        clahe_obj = cv2.createCLAHE(
            clipLimit=float(params["CLAHE_CLIP_LIMIT"]),
            tileGridSize=(tile, tile),
        )
        clahe_img = clahe_obj.apply(normalized)
    else:
        clahe_img = normalized.copy()

    if not skip("bilateral"):
        bilateral = clahe_img
        for _ in range(int(params["BILATERAL_PASSES"])):
            bilateral = cv2.bilateralFilter(
                bilateral,
                int(params["BILATERAL_D"]),
                float(params["BILATERAL_SIGMA_COLOR"]),
                float(params["BILATERAL_SIGMA_SPACE"]),
            )
    else:
        bilateral = clahe_img.copy()

    block = max(3, int(params["ADAPTIVE_BLOCK_SIZE"]) | 1)
    adaptive = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        float(params["ADAPTIVE_C"]),
    )
    _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def norm_polarity(mask: np.ndarray) -> np.ndarray:
        return cv2.bitwise_not(mask) if np.sum(mask > 0) / mask.size > 0.5 else mask

    adaptive = norm_polarity(adaptive)
    otsu = norm_polarity(otsu)

    if not skip("segmentation"):
        top = adaptive[:max(1, int(adaptive.shape[0] * 0.8)), :]
        adaptive_fg = np.sum(top > 0) / top.size
        seg_mask = adaptive if 0.01 <= adaptive_fg <= 0.55 else otsu
    else:
        seg_mask = np.zeros_like(bilateral)

    height, _ = seg_mask.shape
    y_start = min(height - 1, int(round(height * float(params["BASELINE_DETECT_SEARCH_START_RATIO"]))))
    row_ratio = np.mean(seg_mask > 0, axis=1)
    hits = np.flatnonzero(row_ratio[y_start:] >= float(params["BASELINE_DETECT_MIN_ROW_RATIO"]))
    baseline_row = (y_start + int(hits[0])) if hits.size > 0 else None

    after_baseline = seg_mask.copy()
    if not skip("baseline_cut") and baseline_row is not None:
        after_baseline[int(baseline_row):, :] = 0

    if not skip("small_components"):
        min_area = int(params["MIN_COMPONENT_AREA"])
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(after_baseline, connectivity=8)
        cleaned_mask = np.zeros_like(after_baseline)
        band_top = (
            max(0, int(baseline_row) - int(params["SMALL_TREE_BAND_HEIGHT"]))
            if baseline_row is not None else None
        )
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            top = int(stats[label, cv2.CC_STAT_TOP])
            bottom = top + int(stats[label, cv2.CC_STAT_HEIGHT]) - 1
            if band_top is not None and band_top <= bottom <= (baseline_row or height):
                cleaned_mask[labels == label] = 255
                continue
            if area >= min_area:
                cleaned_mask[labels == label] = 255
    else:
        cleaned_mask = after_baseline.copy()

    if not skip("separation") and np.sum(cleaned_mask) > 0:
        dist = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist,
            float(params["DISTANCE_THRESHOLD"]) * dist.max(),
            255,
            cv2.THRESH_BINARY,
        )
        sure_fg = sure_fg.astype(np.uint8)
        sure_bg = cv2.dilate(
            cleaned_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=3,
        )
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR), markers)
        separated = cleaned_mask.copy()
        separated[markers == -1] = 0
    else:
        separated = cleaned_mask.copy()

    if not skip("skeleton"):
        smoothed = cv2.morphologyEx(
            separated,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        skeleton = skeletonize((smoothed > 0).astype(bool)).astype(np.uint8) * 255
        min_branch = int(params["SKELETON_MIN_BRANCH_LENGTH"])
        horizontal_min = int(params["SKELETON_HORIZONTAL_LINE_MIN_WIDTH"])
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        for label in range(1, num_labels):
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_branch or (height <= 3 and width >= horizontal_min):
                skeleton[labels == label] = 0

        spur_len = int(params["SKELETON_SPUR_LENGTH"])
        neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        for _ in range(5):
            binary = (skeleton > 0).astype(np.uint8)
            neighbor_counts = cv2.filter2D(binary, cv2.CV_16S, neighbor_kernel).astype(np.int16)
            endpoints = list(zip(*np.where((binary == 1) & (neighbor_counts == 1))))
            if not endpoints:
                break

            removed_any = False
            for row, col in endpoints:
                if skeleton[row, col] == 0:
                    continue
                path = [(row, col)]
                cur_row, cur_col = row, col
                visited = {(cur_row, cur_col)}

                while True:
                    found = False
                    for d_row in (-1, 0, 1):
                        for d_col in (-1, 0, 1):
                            if d_row == 0 and d_col == 0:
                                continue
                            next_row, next_col = cur_row + d_row, cur_col + d_col
                            if (next_row, next_col) in visited:
                                continue
                            in_bounds = 0 <= next_row < skeleton.shape[0] and 0 <= next_col < skeleton.shape[1]
                            if not in_bounds or skeleton[next_row, next_col] == 0:
                                continue
                            if int(neighbor_counts[next_row, next_col]) >= 3:
                                found = False
                                break
                            path.append((next_row, next_col))
                            visited.add((next_row, next_col))
                            cur_row, cur_col = next_row, next_col
                            found = True
                            break
                        else:
                            continue
                        break
                    if not found:
                        break

                if len(path) <= spur_len:
                    for path_row, path_col in path:
                        skeleton[path_row, path_col] = 0
                    removed_any = True

            if not removed_any:
                break
    else:
        skeleton = np.zeros_like(separated)

    overlay_mask = create_overlay(image, separated, color=(0, 255, 0), alpha=0.55)
    overlay_skel = create_overlay(image, skeleton, color=(0, 0, 255), alpha=0.70)

    return {
        "01_original": image,
        "02_cleaned": cleaned,
        "03_normalized": normalized,
        "04_clahe": clahe_img,
        "05_bilateral": bilateral,
        "06_segmented": seg_mask,
        "07_after_baseline_cut": after_baseline,
        "08_small_removed": cleaned_mask,
        "09_separated": separated,
        "10_skeleton": skeleton,
        "overlay_mask_on_orig": overlay_mask,
        "overlay_skel_on_orig": overlay_skel,
    }


def main() -> int:
    """Load preset, run pipeline on one image, save stages and comparison strips."""
    args = parse_args()
    image_path = Path(args.image).resolve()
    output_root = Path(args.output_root).resolve()
    classic_root = Path(args.classic_root).resolve()
    preset_path = Path(args.preset).resolve()

    with open(preset_path, "r", encoding="utf-8") as f:
        preset = json.load(f)

    basename = image_path.stem
    out_dir = output_root / args.category / basename
    compare_dir = output_root / "comparisons_vs_classic" / args.category / basename
    classic_dir = classic_root / args.category / basename

    if out_dir.exists() or compare_dir.exists():
        raise SystemExit(f"Output already exists for {basename}")

    out_dir.mkdir(parents=True, exist_ok=False)
    compare_dir.mkdir(parents=True, exist_ok=False)

    image = load_image(str(image_path), grayscale=True)
    results = run_preset_pipeline(image, dict(preset["params"]), dict(preset["skips"]))

    for name, img in results.items():
        save_image(img, str(out_dir / f"{name}.png"))

    for preset_file in sorted(out_dir.glob("*.png")):
        classic_file = classic_dir / preset_file.name
        if not classic_file.exists():
            continue
        classic_img = load_image(str(classic_file), grayscale=False)
        preset_img = load_image(str(preset_file), grayscale=False)
        strip = create_comparison_strip(
            [classic_img, preset_img],
            ["Classic", preset_path.stem],
            height=400,
        )
        save_image(strip, str(compare_dir / preset_file.name))

    print(
        f"[{args.category}] {basename}: preset skeleton nonzero={int(np.count_nonzero(results['10_skeleton']))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
