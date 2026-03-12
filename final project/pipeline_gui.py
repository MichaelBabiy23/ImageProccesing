"""
Interactive GUI for exploring the classic SEM dendrite segmentation pipeline.

Open an image, tweak parameters with sliders/spinboxes, and see every pipeline
stage update in real time.  Completely self-contained — does not modify any
other project file.

Features
--------
- Live parameter editing with 250 ms debounce
- Skip checkboxes per pipeline step
- Single-stage view or all-stages grid
- Compare mode: A vs B side-by-side in every view mode
  - Optional difference highlighting overlays changed regions in each pane
  - Tab A and Tab B each have independent params + skips
  - "Copy A → B" copies A's current settings into B
  - Divider line between the two panels is draggable

Usage:
    python pipeline_gui.py
    python pipeline_gui.py path/to/image.tif
"""

import sys
import os
import json
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import classic_pipeline as classic_cfg
from skimage.morphology import skeletonize
from utils import load_image, clean_sem_image, create_overlay

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QSplitter,
        QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
        QComboBox, QScrollArea, QGroupBox, QFileDialog,
        QSizePolicy, QFrame, QTabWidget, QCheckBox, QProgressBar,
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt5.QtGui import QImage, QPixmap, QColor
except ImportError:
    print("PyQt5 is required.  Install it with:  pip install PyQt5")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Default parameter values
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_PARAM_NAMES = (
    "CLAHE_CLIP_LIMIT",
    "CLAHE_TILE_SIZE",
    "BILATERAL_D",
    "BILATERAL_SIGMA_COLOR",
    "BILATERAL_SIGMA_SPACE",
    "BILATERAL_PASSES",
    "ADAPTIVE_BLOCK_SIZE",
    "ADAPTIVE_C",
    "MIN_COMPONENT_AREA",
    "BASELINE_DETECT_MIN_ROW_RATIO",
    "BASELINE_DETECT_SEARCH_START_RATIO",
    "SMALL_TREE_BAND_HEIGHT",
    "DISTANCE_THRESHOLD",
    "SKELETON_MIN_BRANCH_LENGTH",
    "SKELETON_SPUR_LENGTH",
    "SKELETON_HORIZONTAL_LINE_MIN_WIDTH",
)

DEFAULTS = {name: getattr(classic_cfg, name) for name in _DEFAULT_PARAM_NAMES}

PARAM_SPECS = [
    # (name, lo, hi, step, is_float, group)
    ("CLAHE_CLIP_LIMIT",          0.1, 40.0,  0.1,  True,  "Stage A · Pre-processing"),
    ("CLAHE_TILE_SIZE",           2,   64,     1,    False, "Stage A · Pre-processing"),
    ("BILATERAL_D",               1,   25,     1,    False, "Stage A · Pre-processing"),
    ("BILATERAL_SIGMA_COLOR",     1.0, 250.0,  1.0,  True,  "Stage A · Pre-processing"),
    ("BILATERAL_SIGMA_SPACE",     1.0, 250.0,  1.0,  True,  "Stage A · Pre-processing"),
    ("BILATERAL_PASSES",          1,   5,      1,    False, "Stage A · Pre-processing"),
    ("ADAPTIVE_BLOCK_SIZE",       3,   401,    2,    False, "Stage B · Segmentation"),
    ("ADAPTIVE_C",               -50,  50,     1,    False, "Stage B · Segmentation"),
    ("MIN_COMPONENT_AREA",        1,   2000,   5,    False, "Stage C · Post-processing"),
    ("BASELINE_DETECT_MIN_ROW_RATIO",      0.0, 1.0, 0.01, True, "Stage C · Post-processing"),
    ("BASELINE_DETECT_SEARCH_START_RATIO", 0.0, 1.0, 0.01, True, "Stage C · Post-processing"),
    ("SMALL_TREE_BAND_HEIGHT",    0,   200,    1,    False, "Stage C · Post-processing"),
    ("DISTANCE_THRESHOLD",        0.0, 1.0,   0.01, True,  "Stage D · Separation"),
    ("SKELETON_MIN_BRANCH_LENGTH",1,   100,    1,    False, "Skeleton"),
    ("SKELETON_SPUR_LENGTH",      0,   50,     1,    False, "Skeleton"),
    ("SKELETON_HORIZONTAL_LINE_MIN_WIDTH", 1, 500, 5, False, "Skeleton"),
]

SKIP_SPECS = [
    ("clean",            "Skip: Clean (text/scale-bar removal)",  "Stage A · Pre-processing"),
    ("normalize",        "Skip: Histogram normalisation",          "Stage A · Pre-processing"),
    ("clahe",            "Skip: CLAHE",                            "Stage A · Pre-processing"),
    ("bilateral",        "Skip: Bilateral filter",                 "Stage A · Pre-processing"),
    ("segmentation",     "Skip: Segmentation (→ empty mask)",      "Stage B · Segmentation"),
    ("baseline_cut",     "Skip: Baseline cut",                     "Stage C · Post-processing"),
    ("small_components", "Skip: Small-component removal",          "Stage C · Post-processing"),
    ("separation",       "Skip: Branch separation (watershed)",    "Stage D · Separation"),
    ("skeleton",         "Skip: Skeletonisation",                  "Skeleton"),
]

_SKIP_BY_GROUP: dict[str, list] = {}
for _sk, _sl, _sg in SKIP_SPECS:
    _SKIP_BY_GROUP.setdefault(_sg, []).append((_sk, _sl))

STAGE_LABELS = [
    ("01_original",                       "01 · Original"),
    ("02_cleaned",                        "02 · Cleaned"),
    ("03_normalized",                     "03 · Normalized"),
    ("04_clahe",                          "04 · CLAHE"),
    ("05_bilateral",                      "05 · Bilateral"),
    ("06_segmented",                      "06 · Segmented"),
    ("07_after_baseline_cut",             "07 · Baseline Cut"),
    ("08_small_removed",                  "08 · Small Removed"),
    ("09_separated",                      "09 · Separated"),
    ("10_skeleton",                       "10 · Skeleton"),
    ("overlay_mask",                      "Overlay · Mask"),
    ("overlay_skel",                      "Overlay · Skeleton"),
]
STAGE_KEYS = [k for k, _ in STAGE_LABELS]


class _PipelineCancelled(Exception):
    """Internal control-flow exception used to stop stale worker runs."""
    pass

PARAM_HELP = {
    "CLAHE_CLIP_LIMIT": (
        "Limits how aggressively CLAHE boosts local contrast inside each tile.",
        "Increasing it from 0.5 to 2.0 usually makes faint dendrites pop more, but it also boosts SEM grain and halos.",
        "Decreasing it from 0.5 to 0.2 keeps the image calmer, but weak thin branches can stay washed out.",
    ),
    "CLAHE_TILE_SIZE": (
        "Sets how fine the CLAHE tiling is, so it controls how local the contrast enhancement becomes.",
        "Increasing it from 8 to 16 uses more, smaller tiles and can reveal tiny local detail, but it can also make patchiness and noise more obvious.",
        "Decreasing it from 8 to 4 makes the enhancement more global and smoother, but subtle local contrast differences can be missed.",
    ),
    "BILATERAL_D": (
        "Sets the pixel neighborhood size used by the bilateral filter.",
        "Increasing it from 9 to 15 smooths over a wider area and can suppress more grain, but it can round off thin branch edges.",
        "Decreasing it from 9 to 5 keeps edges sharper and more local detail intact, but more SEM speckle remains.",
    ),
    "BILATERAL_SIGMA_COLOR": (
        "Controls how willing the bilateral filter is to smooth across intensity differences.",
        "Increasing it from 50 to 100 blends pixels with larger brightness gaps, which reduces noise but can blur bright dendrite boundaries.",
        "Decreasing it from 50 to 20 preserves contrast changes more strongly, but noisy texture is less suppressed.",
    ),
    "BILATERAL_SIGMA_SPACE": (
        "Controls how far the bilateral filter reaches in space around each pixel.",
        "Increasing it from 50 to 100 makes the filter consider a broader area and smooth more broadly, but small structures can soften.",
        "Decreasing it from 50 to 20 keeps the smoothing tight and local, but grainy regions can remain uneven.",
    ),
    "BILATERAL_PASSES": (
        "Repeats the bilateral filter multiple times to accumulate denoising.",
        "Increasing it from 1 to 3 can clean noisy SEM backgrounds better, but repeated passes can over-smooth thin branches.",
        "Keeping it at 1 or decreasing it from 3 to 1 preserves more raw detail, but more noise reaches thresholding.",
    ),
    "ADAPTIVE_BLOCK_SIZE": (
        "Sets the local window size used to compute the adaptive threshold.",
        "Increasing it from 67 to 121 makes the threshold follow broader illumination trends, which is steadier on uneven backgrounds but less sensitive to tiny local detail.",
        "Decreasing it from 67 to 31 makes the threshold more local and reactive, which can recover fine branches but also increases speckle and fragmentation.",
    ),
    "ADAPTIVE_C": (
        "Offsets the adaptive threshold by subtracting this value from the local mean before binarization.",
        "Increasing it from -12 toward 0 lowers the effective threshold, so more pixels become foreground and the mask usually grows noisier and thicker.",
        "Decreasing it from -12 to -20 raises the effective threshold, so only brighter structures survive and faint branches are more likely to disappear.",
    ),
    "MIN_COMPONENT_AREA": (
        "Defines the smallest connected component area that is kept after cleanup.",
        "Increasing it from 90 to 200 removes more tiny blobs and debris, but short real branches can be dropped too.",
        "Decreasing it from 90 to 30 keeps more small structures, but isolated noise specks are more likely to survive.",
    ),
    "BASELINE_DETECT_MIN_ROW_RATIO": (
        "Defines how full a row must be to count as the dense baseline band near the bottom.",
        "Increasing it from 0.8 to 0.9 makes baseline detection stricter, so the cut happens less often or lower in the image.",
        "Decreasing it from 0.8 to 0.6 makes detection easier, so the lower band is removed sooner but lower dendrite trunks can get clipped.",
    ),
    "BASELINE_DETECT_SEARCH_START_RATIO": (
        "Sets how far down the image the baseline search begins.",
        "Increasing it from 0.6 to 0.75 starts the search lower, which avoids early false hits but can miss a baseline that begins higher up.",
        "Decreasing it from 0.6 to 0.4 starts the search earlier, which can catch higher artifacts but may cut into valid lower structures.",
    ),
    "SMALL_TREE_BAND_HEIGHT": (
        "Protects a band above the detected baseline so tiny components there are not removed by the area filter.",
        "Increasing it from 30 to 60 keeps more short branches attached near the bottom, but it also preserves more debris in that band.",
        "Decreasing it from 30 to 10 makes the cleanup stricter near the baseline, but genuine small branches in that zone can vanish.",
    ),
    "DISTANCE_THRESHOLD": (
        "Controls how strong a distance-transform peak must be to become a watershed foreground marker.",
        "Increasing it from 0.35 to 0.6 creates fewer, smaller markers, so touching branches are less likely to split apart.",
        "Decreasing it from 0.35 to 0.2 creates broader markers, which can separate more touching structures but can also over-fragment one branch into many pieces.",
    ),
    "SKELETON_MIN_BRANCH_LENGTH": (
        "Defines the minimum skeleton component size kept after pruning.",
        "Increasing it from 8 to 20 removes more short stubs and isolated fragments, but small real side branches can be lost.",
        "Decreasing it from 8 to 3 preserves more tiny skeleton pieces, but the result gets noisier.",
    ),
    "SKELETON_SPUR_LENGTH": (
        "Sets the longest endpoint spur that will be trimmed from the skeleton.",
        "Increasing it from 6 to 12 removes longer hooks and whiskers, but it can also shorten real terminal branches.",
        "Decreasing it from 6 to 2 only removes the tiniest burrs, but more spiky artifacts remain.",
    ),
    "SKELETON_HORIZONTAL_LINE_MIN_WIDTH": (
        "Sets how wide a near-horizontal skeleton fragment must be before it is treated as an artifact and removed.",
        "Increasing it from 40 to 80 keeps more medium-width horizontal segments and only strips very long lines.",
        "Decreasing it from 40 to 20 removes shorter horizontal fragments too, which can clean artifacts but may cut real lateral branches.",
    ),
}

SKIP_HELP = {
    "clean": (
        "Skips the text and scale-bar cleanup stage before any enhancement runs.",
        "Checked example: the scale bar or labels stay in the image and can be mistaken for foreground later.",
        "Unchecked example: those overlays are removed before the rest of the pipeline starts.",
    ),
    "normalize": (
        "Skips the histogram normalization step that stretches the image to the full intensity range.",
        "Checked example: a low-contrast input stays flat and later stages have less separation to work with.",
        "Unchecked example: the grayscale range is expanded before CLAHE and thresholding.",
    ),
    "clahe": (
        "Skips local contrast enhancement with CLAHE.",
        "Checked example: faint dendrites stay softer and some local detail may never stand out.",
        "Unchecked example: local contrast is boosted before denoising and segmentation.",
    ),
    "bilateral": (
        "Skips the edge-preserving denoising stage.",
        "Checked example: more SEM grain reaches thresholding and the mask often gets noisier.",
        "Unchecked example: background noise is smoothed while most strong edges stay intact.",
    ),
    "segmentation": (
        "Skips threshold-based segmentation and forces an empty mask for the later stages.",
        "Checked example: the downstream cleanup and skeleton views become empty because there is no foreground.",
        "Unchecked example: adaptive thresholding and Otsu fallback generate the initial mask.",
    ),
    "baseline_cut": (
        "Skips the cutoff that removes dense rows at the detected lower baseline.",
        "Checked example: a bright lower band can survive and connect unrelated components.",
        "Unchecked example: rows at and below the detected baseline are zeroed out.",
    ),
    "small_components": (
        "Skips connected-component cleanup by area.",
        "Checked example: tiny specks and fragments remain in the mask.",
        "Unchecked example: only components above the size rule, plus the protected baseline band, are kept.",
    ),
    "separation": (
        "Skips watershed-based splitting of touching branches.",
        "Checked example: merged blobs stay connected as one piece.",
        "Unchecked example: distance-transform markers are used to split touching structures where possible.",
    ),
    "skeleton": (
        "Skips skeletonization and all skeleton pruning.",
        "Checked example: the skeleton view becomes empty even if the mask exists.",
        "Unchecked example: the cleaned mask is thinned to a one-pixel skeleton and pruned.",
    ),
}


def _format_param_tooltip(name: str) -> str:
    desc, inc, dec = PARAM_HELP[name]
    return f"{desc}\nIncrease example: {inc}\nDecrease example: {dec}"


def _format_skip_tooltip(name: str) -> str:
    desc, checked, unchecked = SKIP_HELP[name]
    return f"{desc}\nChecked example: {checked}\nUnchecked example: {unchecked}"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(image: np.ndarray, p: dict, skips: dict | None = None,
                  progress_cb=None) -> dict:
    """
    progress_cb: callable(pct: int, stage_name: str) or None.
    Called at the start of each named stage so the UI can update a progress bar.
    """
    if skips is None:
        skips = {}

    def _skip(key):
        return skips.get(key, False)

    def _prog(pct: int, name: str):
        if progress_cb is not None:
            keep_going = progress_cb(pct, name)
            if keep_going is False:
                raise _PipelineCancelled()

    # Stage A
    _prog(0, "Clean")
    cleaned = clean_sem_image(image) if not _skip("clean") else image.copy()

    _prog(5, "Normalize")
    if not _skip("normalize"):
        mn, mx = float(cleaned.min()), float(cleaned.max())
        normalized = ((cleaned.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(cleaned)
    else:
        normalized = cleaned.copy()

    _prog(12, "CLAHE")
    if not _skip("clahe"):
        tile = int(p["CLAHE_TILE_SIZE"])
        clahe_obj = cv2.createCLAHE(clipLimit=float(p["CLAHE_CLIP_LIMIT"]), tileGridSize=(tile, tile))
        clahe_img = clahe_obj.apply(normalized)
    else:
        clahe_img = normalized.copy()

    _prog(20, "Bilateral Filter")
    if not _skip("bilateral"):
        bilateral = clahe_img
        for _ in range(int(p["BILATERAL_PASSES"])):
            bilateral = cv2.bilateralFilter(bilateral, int(p["BILATERAL_D"]),
                                            float(p["BILATERAL_SIGMA_COLOR"]),
                                            float(p["BILATERAL_SIGMA_SPACE"]))
    else:
        bilateral = clahe_img.copy()

    # Stage B
    _prog(32, "Segmentation")
    block = max(3, int(p["ADAPTIVE_BLOCK_SIZE"]) | 1)
    adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, float(p["ADAPTIVE_C"]))
    _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def _norm_pol(m):
        return cv2.bitwise_not(m) if np.sum(m > 0) / m.size > 0.5 else m

    adaptive = _norm_pol(adaptive)
    otsu = _norm_pol(otsu)

    if not _skip("segmentation"):
        top = adaptive[:max(1, int(adaptive.shape[0] * 0.8)), :]
        a_fg = np.sum(top > 0) / top.size
        seg_mask = adaptive if 0.01 <= a_fg <= 0.55 else otsu
    else:
        seg_mask = np.zeros_like(bilateral)

    # Stage C — baseline cut + small component removal
    _prog(44, "Baseline Cut")
    h, _ = seg_mask.shape
    y_start = min(h - 1, int(round(h * float(p["BASELINE_DETECT_SEARCH_START_RATIO"]))))
    row_ratio = np.mean(seg_mask > 0, axis=1)
    idx = np.flatnonzero(row_ratio[y_start:] >= float(p["BASELINE_DETECT_MIN_ROW_RATIO"]))
    baseline_row = (y_start + int(idx[0])) if idx.size > 0 else None

    after_baseline = seg_mask.copy()
    if not _skip("baseline_cut") and baseline_row is not None:
        after_baseline[int(baseline_row):, :] = 0

    _prog(58, "Small Component Removal")
    if not _skip("small_components"):
        min_area = int(p["MIN_COMPONENT_AREA"])
        n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(after_baseline, connectivity=8)
        cleaned_mask = np.zeros_like(after_baseline)
        band_top = max(0, int(baseline_row) - int(p["SMALL_TREE_BAND_HEIGHT"])) if baseline_row is not None else None
        for i2 in range(1, n2):
            area2 = int(stats2[i2, cv2.CC_STAT_AREA])
            top2 = int(stats2[i2, cv2.CC_STAT_TOP])
            bot2 = top2 + int(stats2[i2, cv2.CC_STAT_HEIGHT]) - 1
            if band_top is not None and band_top <= bot2 <= (baseline_row or h):
                cleaned_mask[labels2 == i2] = 255; continue
            if area2 >= min_area:
                cleaned_mask[labels2 == i2] = 255
    else:
        cleaned_mask = after_baseline.copy()

    # Stage D
    _prog(78, "Branch Separation")
    if not _skip("separation") and np.sum(cleaned_mask) > 0:
        dist = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, float(p["DISTANCE_THRESHOLD"]) * dist.max(), 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        sure_bg = cv2.dilate(cleaned_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1; markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR), markers)
        separated = cleaned_mask.copy(); separated[markers == -1] = 0
    else:
        separated = cleaned_mask.copy()

    _prog(88, "Skeletonisation")
    # Skeleton
    if not _skip("skeleton"):
        smoothed = cv2.morphologyEx(separated, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        skeleton = skeletonize((smoothed > 0).astype(bool)).astype(np.uint8) * 255
        # prune
        min_branch = int(p["SKELETON_MIN_BRANCH_LENGTH"])
        horiz_min = int(p["SKELETON_HORIZONTAL_LINE_MIN_WIDTH"])
        n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        for i2 in range(1, n2):
            sw = stats2[i2, cv2.CC_STAT_WIDTH]; sh = stats2[i2, cv2.CC_STAT_HEIGHT]
            if stats2[i2, cv2.CC_STAT_AREA] < min_branch or (sh <= 3 and sw >= horiz_min):
                skeleton[labels2 == i2] = 0
        # spurs
        spur_len = int(p["SKELETON_SPUR_LENGTH"])
        nb_k = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        for _ in range(5):
            b2 = (skeleton > 0).astype(np.uint8)
            nbc = cv2.filter2D(b2, cv2.CV_16S, nb_k).astype(np.int16)
            ep_coords = list(zip(*np.where((b2 == 1) & (nbc == 1))))
            if not ep_coords: break
            removed = False
            for r2, c2 in ep_coords:
                if skeleton[r2, c2] == 0: continue
                path = [(r2, c2)]; cr2, cc2 = r2, c2; visited = {(cr2, cc2)}
                while True:
                    found = False
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0: continue
                            nr2, nc2 = cr2 + dr, cc2 + dc
                            if (nr2, nc2) in visited: continue
                            if 0 <= nr2 < skeleton.shape[0] and 0 <= nc2 < skeleton.shape[1] and skeleton[nr2, nc2] > 0:
                                if int(nbc[nr2, nc2]) >= 3: found = False; break
                                path.append((nr2, nc2)); visited.add((nr2, nc2)); cr2, cc2 = nr2, nc2; found = True; break
                        else: continue
                        break
                    if not found: break
                if len(path) <= spur_len:
                    for pr2, pc2 in path: skeleton[pr2, pc2] = 0
                    removed = True
            if not removed: break
    else:
        skeleton = np.zeros_like(separated)

    _prog(96, "Overlays")
    overlay_mask = create_overlay(image, separated, color=(0, 255, 0), alpha=0.55)
    overlay_skel = create_overlay(image, skeleton, color=(0, 0, 255), alpha=0.70)
    _prog(100, "Done")

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
        "overlay_mask": overlay_mask,
        "overlay_skel": overlay_skel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _as_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    if np.issubdtype(img.dtype, np.floating):
        if img.size and np.nanmax(img) <= 1.0 and np.nanmin(img) >= 0.0:
            return np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return np.clip(img, 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        return (64, 64, 255)
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return (b, g, r)


def _difference_mask(img_a: np.ndarray, img_b: np.ndarray,
                     threshold: int = 12) -> np.ndarray | None:
    if img_a is None or img_b is None:
        return None

    a = _to_bgr(_as_uint8(img_a))
    b = _to_bgr(_as_uint8(img_b))
    if b.shape[:2] != a.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)

    diff = cv2.absdiff(a, b)
    mask = (diff.max(axis=2) >= threshold).astype(np.uint8) * 255
    if not np.any(mask):
        return mask

    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def _highlight_differences(img: np.ndarray, compare_img: np.ndarray,
                           highlight_color: tuple[int, int, int]) -> np.ndarray:
    base = _to_bgr(_as_uint8(img))
    mask = _difference_mask(img, compare_img)
    if mask is None or not np.any(mask):
        return base

    solid = np.full_like(base, highlight_color)
    blended = cv2.addWeighted(base, 0.55, solid, 0.45, 0)
    out = base.copy()
    out[mask > 0] = blended[mask > 0]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, highlight_color, 1, cv2.LINE_AA)
    return out


def _make_thumb(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    img = _to_bgr(img)
    ih, iw = img.shape[:2]
    scale = min(tw / iw, th / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    yo, xo = (th - nh) // 2, (tw - nw) // 2
    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas


def _np_to_pixmap(img: np.ndarray, w: int, h: int) -> QPixmap:
    if img is None:
        return QPixmap()
    if img.ndim == 2:
        ih, iw = img.shape
        # ensure contiguous
        img = np.ascontiguousarray(img)
        q = QImage(img.data, iw, ih, iw, QImage.Format_Grayscale8)
    else:
        ih, iw = img.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        q = QImage(rgb.data, iw, ih, iw * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(q).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _build_grid_canvas(results: dict, cols: int = 4,
                       cell_w: int = 320, cell_h: int = 260,
                       compare_results: dict | None = None,
                       highlight_color: tuple[int, int, int] | None = None) -> np.ndarray:
    keys = [k for k in STAGE_KEYS if k in results]
    rows = (len(keys) + cols - 1) // cols
    label_h, pad = 20, 4
    total_w = cols * (cell_w + pad) + pad
    total_h = rows * (cell_h + label_h + pad) + pad
    canvas = np.full((total_h, total_w, 3), 40, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, key in enumerate(keys):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + label_h + pad)
        img = results[key]
        if compare_results is not None and highlight_color is not None:
            img = _highlight_differences(img, compare_results.get(key), highlight_color)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = _make_thumb(img, cell_w, cell_h)
        lbl = next((l for k2, l in STAGE_LABELS if k2 == key), key)
        cv2.putText(canvas, lbl, (x0 + 2, y0 + cell_h + label_h - 4),
                    font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Background worker thread
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    """Runs _run_pipeline on a background thread; emits Qt signals for progress."""
    progress   = pyqtSignal(int, str)   # (percent 0-100, stage name)
    result_ready = pyqtSignal(str, dict)  # (slot "a"/"b", results dict)
    error      = pyqtSignal(str, str)   # (slot, error message)

    def __init__(self, slot: str, image: np.ndarray, params: dict, skips: dict):
        super().__init__()
        self.slot   = slot
        self.image  = image
        self.params = params
        self.skips  = skips
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        def _cb(pct: int, name: str):
            if self._cancelled:
                return False
            self.progress.emit(pct, name)
            return True

        try:
            results = _run_pipeline(self.image, self.params, self.skips, _cb)
            if not self._cancelled:
                self.result_ready.emit(self.slot, results)
        except _PipelineCancelled:
            pass
        except Exception as e:
            if not self._cancelled:
                self.error.emit(self.slot, f"{e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# ParamRow widget
# ─────────────────────────────────────────────────────────────────────────────

class ParamRow(QWidget):
    value_changed = pyqtSignal(str, object)

    def __init__(self, name, default, lo, hi, step, is_float=False, tooltip="", parent=None):
        super().__init__(parent)
        self.name = name
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 1, 2, 1)

        lbl = QLabel(name)
        lbl.setFixedWidth(240)
        lbl.setToolTip(tooltip or name)
        lay.addWidget(lbl)

        if is_float:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(3)
            self.spin.setSingleStep(step)
        else:
            self.spin = QSpinBox()
            self.spin.setSingleStep(int(step))

        self.spin.setMinimum(lo)
        self.spin.setMaximum(hi)
        self.spin.setValue(default)
        self.spin.setFixedWidth(90)
        self.spin.setToolTip(tooltip or name)
        lay.addWidget(self.spin)

        rst = QPushButton("↺")
        rst.setFixedWidth(26)
        rst_tip = f"Reset to {default}"
        if tooltip:
            rst_tip = f"{tooltip}\n{rst_tip}"
        rst.setToolTip(rst_tip)
        rst.clicked.connect(lambda: self.spin.setValue(default))
        lay.addWidget(rst)

        self.setToolTip(tooltip or name)
        self.spin.valueChanged.connect(lambda v: self.value_changed.emit(self.name, v))

    def get_value(self):
        return self.spin.value()

    def set_value(self, v):
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)


# ─────────────────────────────────────────────────────────────────────────────
# ParamPanel — one full parameter panel (used for both A and B)
# ─────────────────────────────────────────────────────────────────────────────

class ParamPanel(QScrollArea):
    """Scrollable panel containing all param rows + skip checkboxes for one slot."""
    changed = pyqtSignal()   # emitted whenever any value changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)

        self._rows: dict[str, ParamRow] = {}
        self._skip_cbs: dict[str, QCheckBox] = {}

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setSpacing(4)
        lay.setContentsMargins(4, 4, 4, 4)

        current_group = None
        group_lay = None

        for spec in PARAM_SPECS:
            name, lo, hi, step, is_float, group = spec

            if group != current_group:
                current_group = group
                gb = QGroupBox(group)
                group_lay = QVBoxLayout(gb)
                group_lay.setSpacing(2)
                group_lay.setContentsMargins(4, 8, 4, 4)
                lay.addWidget(gb)

                for skip_key, skip_label in _SKIP_BY_GROUP.get(group, []):
                    cb = QCheckBox(skip_label)
                    cb.setChecked(False)
                    cb.setStyleSheet("color: #f38ba8; font-style: italic;")
                    cb.setToolTip(_format_skip_tooltip(skip_key))
                    cb.stateChanged.connect(self._emit_changed)
                    self._skip_cbs[skip_key] = cb
                    group_lay.addWidget(cb)

                if _SKIP_BY_GROUP.get(group):
                    line = QFrame()
                    line.setFrameShape(QFrame.HLine)
                    line.setStyleSheet("color: #45475a;")
                    group_lay.addWidget(line)

            row = ParamRow(
                name,
                DEFAULTS[name],
                lo,
                hi,
                step,
                is_float,
                tooltip=_format_param_tooltip(name),
            )
            row.value_changed.connect(self._emit_changed)
            self._rows[name] = row
            group_lay.addWidget(row)

        lay.addStretch()
        self.setWidget(inner)

    def _emit_changed(self, *_):
        self.changed.emit()

    def get_params(self) -> dict:
        return {name: row.get_value() for name, row in self._rows.items()}

    def get_skips(self) -> dict:
        return {key: cb.isChecked() for key, cb in self._skip_cbs.items()}

    def get_state(self) -> dict:
        return {
            "params": self.get_params(),
            "skips": self.get_skips(),
        }

    def set_params(self, params: dict):
        for name, row in self._rows.items():
            if name in params:
                row.set_value(params[name])

    def set_skips(self, skips: dict):
        for key, cb in self._skip_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(skips.get(key, False))
            cb.blockSignals(False)

    def reset_all(self):
        for name, row in self._rows.items():
            row.set_value(DEFAULTS[name])
        for cb in self._skip_cbs.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def apply_state(self, state: dict):
        self.set_params(state.get("params", {}))
        self.set_skips(state.get("skips", {}))


# ─────────────────────────────────────────────────────────────────────────────
# ImagePane — a display pane for one result set
# ─────────────────────────────────────────────────────────────────────────────

class ImagePane(QWidget):
    """Display area for one slot's pipeline results, with a progress bar."""

    def __init__(self, label_text: str, label_color: str, parent=None):
        super().__init__(parent)
        self._accent = label_color
        self._accent_bgr = _hex_to_bgr(label_color)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # ── header row: slot label + stage name
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)

        self.header = QLabel(label_text)
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setFixedWidth(36)
        self.header.setFixedHeight(22)
        self.header.setStyleSheet(
            f"background-color: {label_color}; color: #1e1e2e; "
            f"font-weight: bold; font-size: 13px; border-radius: 3px;"
        )
        header_row.addWidget(self.header)

        self.stage_label = QLabel("")
        self.stage_label.setStyleSheet(f"color: {label_color}; font-size: 11px;")
        header_row.addWidget(self.stage_label, stretch=1)

        lay.addLayout(header_row)

        # ── progress bar (hidden when idle)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #313244;
                border: none;
                border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background-color: {label_color};
                border-radius: 3px;
            }}
        """)
        self.progress_bar.setVisible(False)
        lay.addWidget(self.progress_bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel("No result")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll.setWidget(self.img_label)
        lay.addWidget(self.scroll)

        self.results: dict = {}

    def set_running(self, running: bool):
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setValue(0)
            self.stage_label.setText("Starting…")

    def set_progress(self, pct: int, name: str):
        self.progress_bar.setValue(pct)
        self.stage_label.setText(name)

    def set_done(self):
        self.progress_bar.setValue(100)
        # brief flash then hide
        QTimer.singleShot(600, lambda: (
            self.progress_bar.setVisible(False),
            self.stage_label.setText(""),
        ))

    def set_results(self, results: dict):
        self.results = results

    def refresh(self, mode: str, stage_key: str, compare_results: dict | None = None):
        if not self.results:
            return
        vw = self.scroll.viewport().width() - 4
        vh = self.scroll.viewport().height() - 4
        if vw < 2 or vh < 2:
            return

        if mode == "single":
            img = self.results.get(stage_key)
            if img is None:
                return
            if compare_results is not None:
                img = _highlight_differences(img, compare_results.get(stage_key), self._accent_bgr)
            pix = _np_to_pixmap(img, vw, vh)
        else:
            canvas = _build_grid_canvas(
                self.results,
                compare_results=compare_results,
                highlight_color=self._accent_bgr if compare_results is not None else None,
            )
            scale = min(vw / canvas.shape[1], vh / canvas.shape[0], 1.0)
            if scale < 1.0:
                nw = max(1, int(canvas.shape[1] * scale))
                nh = max(1, int(canvas.shape[0] * scale))
                canvas = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            h3, w3 = rgb.shape[:2]
            q = QImage(rgb.data, w3, h3, w3 * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q)

        self.img_label.setPixmap(pix)
        self.img_label.setFixedSize(pix.size())


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    # Accent colours for slot A (blue) and slot B (amber)
    _COLOR_A = "#89b4fa"
    _COLOR_B = "#fab387"

    def __init__(self, initial_image_path=None):
        super().__init__()
        self.setWindowTitle("SEM Dendrite Pipeline — Interactive Explorer")
        self.resize(1700, 950)

        self.image: np.ndarray | None = None
        self._results_a: dict = {}
        self._results_b: dict = {}
        self._compare_mode = False
        self._worker_a: PipelineWorker | None = None
        self._worker_b: PipelineWorker | None = None
        self._pending_rerun = {"a": False, "b": False}
        self._close_requested = False

        # Two independent debounce timers — one per slot
        self._timer_a = QTimer(); self._timer_a.setSingleShot(True)
        self._timer_a.timeout.connect(lambda: self._run_slot("a"))
        self._timer_b = QTimer(); self._timer_b.setSingleShot(True)
        self._timer_b.timeout.connect(lambda: self._run_slot("b"))

        self._build_ui()
        self._apply_dark_theme()

        if initial_image_path and os.path.isfile(initial_image_path):
            self._load_image(initial_image_path)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QHBoxLayout(central)
        root_lay.setContentsMargins(6, 6, 6, 6)
        root_lay.setSpacing(6)

        # ── Left: tab panel A / B ─────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(430)
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        # Image open + status
        open_btn = QPushButton("📂  Open Image…")
        open_btn.setFixedHeight(36)
        open_btn.clicked.connect(self._open_image_dialog)
        left_lay.addWidget(open_btn)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        left_lay.addWidget(self.image_label)

        # Compare toggle
        self.compare_btn = QPushButton("⚡  Enable Compare A vs B")
        self.compare_btn.setCheckable(True)
        self.compare_btn.toggled.connect(self._toggle_compare)
        left_lay.addWidget(self.compare_btn)

        self.highlight_diff_cb = QCheckBox("Highlight differences")
        self.highlight_diff_cb.setChecked(True)
        self.highlight_diff_cb.setEnabled(False)
        self.highlight_diff_cb.toggled.connect(self._refresh_display)
        left_lay.addWidget(self.highlight_diff_cb)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # --- Tab A ---
        tab_a_widget = QWidget()
        tab_a_lay = QVBoxLayout(tab_a_widget)
        tab_a_lay.setContentsMargins(2, 2, 2, 2)
        tab_a_lay.setSpacing(3)

        btns_a = QHBoxLayout()
        reset_a = QPushButton("↺ Reset A")
        reset_a.clicked.connect(lambda: self._reset_slot("a"))
        save_a = QPushButton("Save A…")
        save_a.clicked.connect(lambda: self._save_slot_preset("a"))
        load_a = QPushButton("Load A…")
        load_a.clicked.connect(lambda: self._load_slot_preset("a"))
        copy_to_b = QPushButton("Copy A → B")
        copy_to_b.clicked.connect(self._copy_a_to_b)
        btns_a.addWidget(reset_a)
        btns_a.addWidget(save_a)
        btns_a.addWidget(load_a)
        btns_a.addWidget(copy_to_b)
        tab_a_lay.addLayout(btns_a)

        self.live_cb = QCheckBox("Live update")
        self.live_cb.setChecked(True)
        tab_a_lay.addWidget(self.live_cb)
        run_a = QPushButton("▶  Run A")
        run_a.clicked.connect(lambda: self._schedule("a", 0))
        tab_a_lay.addWidget(run_a)

        self.panel_a = ParamPanel()
        self.panel_a.changed.connect(lambda: self._on_panel_changed("a"))
        tab_a_lay.addWidget(self.panel_a)
        self.tabs.addTab(tab_a_widget, "A")
        self.tabs.tabBar().setTabTextColor(0, QColor(self._COLOR_A))

        # --- Tab B ---
        tab_b_widget = QWidget()
        tab_b_lay = QVBoxLayout(tab_b_widget)
        tab_b_lay.setContentsMargins(2, 2, 2, 2)
        tab_b_lay.setSpacing(3)

        btns_b = QHBoxLayout()
        reset_b = QPushButton("↺ Reset B")
        reset_b.clicked.connect(lambda: self._reset_slot("b"))
        save_b = QPushButton("Save B…")
        save_b.clicked.connect(lambda: self._save_slot_preset("b"))
        load_b = QPushButton("Load B…")
        load_b.clicked.connect(lambda: self._load_slot_preset("b"))
        copy_to_a = QPushButton("Copy B → A")
        copy_to_a.clicked.connect(self._copy_b_to_a)
        btns_b.addWidget(reset_b)
        btns_b.addWidget(save_b)
        btns_b.addWidget(load_b)
        btns_b.addWidget(copy_to_a)
        tab_b_lay.addLayout(btns_b)

        run_b = QPushButton("▶  Run B")
        run_b.clicked.connect(lambda: self._schedule("b", 0))
        tab_b_lay.addWidget(run_b)

        self.panel_b = ParamPanel()
        self.panel_b.changed.connect(lambda: self._on_panel_changed("b"))
        tab_b_lay.addWidget(self.panel_b)
        self.tabs.addTab(tab_b_widget, "B")
        self.tabs.tabBar().setTabTextColor(1, QColor(self._COLOR_B))

        left_lay.addWidget(self.tabs)
        root_lay.addWidget(left)

        # ── Right: viewer ─────────────────────────────────────────────────
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)

        # View controls toolbar
        ctrl = QHBoxLayout()

        self.status_label = QLabel("Ready")
        ctrl.addWidget(self.status_label)
        ctrl.addStretch()

        ctrl.addWidget(QLabel("Mode:"))
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Single stage", "All stages (grid)"])
        self.view_mode.currentIndexChanged.connect(self._refresh_display)
        ctrl.addWidget(self.view_mode)

        ctrl.addWidget(QLabel("Stage:"))
        self.stage_combo = QComboBox()
        for key, label in STAGE_LABELS:
            self.stage_combo.addItem(label, key)
        self.stage_combo.currentIndexChanged.connect(self._refresh_display)
        ctrl.addWidget(self.stage_combo)

        right_lay.addLayout(ctrl)

        # The display area: a splitter holding pane A (always) and pane B (only in compare)
        self.view_splitter = QSplitter(Qt.Horizontal)

        self.pane_a = ImagePane("  A  ", self._COLOR_A)
        self.pane_b = ImagePane("  B  ", self._COLOR_B)

        self.view_splitter.addWidget(self.pane_a)
        self.view_splitter.addWidget(self.pane_b)
        self.pane_b.setVisible(False)

        right_lay.addWidget(self.view_splitter)
        root_lay.addWidget(right, stretch=1)

    # ── Compare toggle ────────────────────────────────────────────────────────

    def _toggle_compare(self, on: bool):
        self._compare_mode = on
        self.pane_b.setVisible(on)
        self.highlight_diff_cb.setEnabled(on)
        self.compare_btn.setText(
            "✖  Disable Compare" if on else "⚡  Enable Compare A vs B"
        )
        if on and not self._results_b and self.image is not None:
            self._schedule("b", 0)
        self._refresh_display()

    # ── Slot logic ────────────────────────────────────────────────────────────

    def _on_panel_changed(self, slot: str):
        if self.live_cb.isChecked():
            self._schedule(slot, 250)

    def _schedule(self, slot: str, delay: int):
        if self._close_requested:
            return
        if slot == "a":
            self._timer_a.start(delay)
        else:
            self._timer_b.start(delay)

    def _run_slot(self, slot: str):
        if self.image is None or self._close_requested:
            return

        panel = self.panel_a if slot == "a" else self.panel_b
        pane  = self.pane_a  if slot == "a" else self.pane_b
        current_worker = self._worker_a if slot == "a" else self._worker_b

        # Never destroy an in-flight QThread. Mark it stale and queue one rerun.
        if current_worker is not None:
            if current_worker.isRunning():
                current_worker.cancel()
                self._pending_rerun[slot] = True
                pane.set_running(True)
                pane.stage_label.setText("Updating…")
                self.status_label.setText(f"Updating {slot.upper()}…")
                return
            if slot == "a":
                self._worker_a = None
            else:
                self._worker_b = None

        worker = PipelineWorker(slot, self.image, panel.get_params(), panel.get_skips())

        worker.progress.connect(lambda pct, name, p=pane: p.set_progress(pct, name))
        worker.result_ready.connect(self._on_worker_result)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(lambda s=slot, w=worker: self._on_worker_thread_finished(s, w))

        if slot == "a":
            self._worker_a = worker
        else:
            self._worker_b = worker

        self._pending_rerun[slot] = False
        pane.set_running(True)
        self.status_label.setText(f"Running {slot.upper()}…")
        worker.start()

    def _on_worker_result(self, slot: str, results: dict):
        pane = self.pane_a if slot == "a" else self.pane_b
        pane.set_done()
        pane.set_results(results)
        if slot == "a":
            self._results_a = results
        else:
            self._results_b = results
        self.status_label.setText("Done ✓")
        self._refresh_display()

    def _on_worker_error(self, slot: str, msg: str):
        pane = self.pane_a if slot == "a" else self.pane_b
        pane.set_running(False)
        pane.stage_label.setText("Error")
        self.status_label.setText(f"Error ({slot.upper()}) — see console")
        print(f"[Pipeline error — slot {slot.upper()}]\n{msg}")

    def _on_worker_thread_finished(self, slot: str, worker: PipelineWorker):
        current_worker = self._worker_a if slot == "a" else self._worker_b
        if current_worker is worker:
            if slot == "a":
                self._worker_a = None
            else:
                self._worker_b = None

        worker.deleteLater()

        if self._close_requested:
            if not self._has_running_workers():
                self.close()
            return

        if self._pending_rerun[slot] and self.image is not None:
            self._pending_rerun[slot] = False
            self._run_slot(slot)

    def _has_running_workers(self) -> bool:
        return any(
            worker is not None and worker.isRunning()
            for worker in (self._worker_a, self._worker_b)
        )

    def _reset_slot(self, slot: str):
        panel = self.panel_a if slot == "a" else self.panel_b
        panel.reset_all()
        self._schedule(slot, 0)

    def _copy_a_to_b(self):
        self.panel_b.set_params(self.panel_a.get_params())
        self.panel_b.set_skips(self.panel_a.get_skips())
        self._schedule("b", 0)

    def _copy_b_to_a(self):
        self.panel_a.set_params(self.panel_b.get_params())
        self.panel_a.set_skips(self.panel_b.get_skips())
        self._schedule("a", 0)

    def _slot_panel(self, slot: str) -> ParamPanel:
        return self.panel_a if slot == "a" else self.panel_b

    def _default_preset_path(self, slot: str) -> str:
        return os.path.join(os.path.dirname(__file__), f"pipeline_gui_preset_{slot}.json")

    def _preset_payload(self, slot: str) -> dict:
        panel = self._slot_panel(slot)
        return {
            "schema_version": 1,
            "slot": slot,
            "params": panel.get_params(),
            "skips": panel.get_skips(),
        }

    def _validate_preset_payload(self, payload: dict) -> tuple[dict, str | None]:
        if not isinstance(payload, dict):
            return {}, "Preset file must contain a JSON object."

        params = payload.get("params")
        skips = payload.get("skips")
        if not isinstance(params, dict):
            return {}, "Preset is missing a valid 'params' object."
        if not isinstance(skips, dict):
            return {}, "Preset is missing a valid 'skips' object."

        clean_params = {}
        for name in DEFAULTS:
            if name not in params:
                continue
            value = params[name]
            if not isinstance(value, (int, float)):
                return {}, f"Invalid value for parameter '{name}'."
            clean_params[name] = value

        clean_skips = {}
        known_skip_keys = {key for key, _, _ in SKIP_SPECS}
        for key, value in skips.items():
            if key not in known_skip_keys:
                continue
            if not isinstance(value, bool):
                return {}, f"Invalid value for skip flag '{key}'."
            clean_skips[key] = value

        return {"params": clean_params, "skips": clean_skips}, None

    def _save_slot_preset(self, slot: str):
        suggested = self._default_preset_path(slot)
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save Preset for {slot.upper()}",
            suggested,
            "JSON Files (*.json);;All files (*.*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._preset_payload(slot), f, indent=2, sort_keys=True)
            self.status_label.setText(f"Saved preset {os.path.basename(path)}")
        except Exception as e:
            self.status_label.setText(f"Save failed: {e}")

    def _load_slot_preset(self, slot: str):
        suggested = self._default_preset_path(slot)
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load Preset into {slot.upper()}",
            suggested,
            "JSON Files (*.json);;All files (*.*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            state, err = self._validate_preset_payload(payload)
            if err is not None:
                self.status_label.setText(f"Load failed: {err}")
                return
            self._slot_panel(slot).apply_state(state)
            self.status_label.setText(f"Loaded preset {os.path.basename(path)} into {slot.upper()}")
            self._schedule(slot, 0)
        except Exception as e:
            self.status_label.setText(f"Load failed: {e}")

    # ── Image loading ─────────────────────────────────────────────────────────

    def _open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SEM Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*.*)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        try:
            self.image = load_image(path, grayscale=True)
            self.image_label.setText(os.path.basename(path))
            self.status_label.setText("Image loaded — running pipeline…")
            self._schedule("a", 0)
            if self._compare_mode:
                self._schedule("b", 0)
        except Exception as e:
            self.status_label.setText(f"Error loading: {e}")

    # ── Display refresh ───────────────────────────────────────────────────────

    def _refresh_display(self):
        mode = "single" if self.view_mode.currentText() == "Single stage" else "grid"
        stage_key = self.stage_combo.currentData()
        highlight_diffs = self._compare_mode and self.highlight_diff_cb.isChecked()

        self.pane_a.refresh(
            mode,
            stage_key,
            compare_results=self._results_b if highlight_diffs else None,
        )
        if self._compare_mode:
            self.pane_b.refresh(
                mode,
                stage_key,
                compare_results=self._results_a if highlight_diffs else None,
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_display()

    def closeEvent(self, event):
        self._timer_a.stop()
        self._timer_b.stop()
        self._pending_rerun["a"] = False
        self._pending_rerun["b"] = False

        active = []
        for worker in (self._worker_a, self._worker_b):
            if worker is not None and worker.isRunning():
                worker.cancel()
                active.append(worker)

        if active:
            self._close_requested = True
            self.status_label.setText("Stopping pipeline threads…")
            event.ignore()
            return

        super().closeEvent(event)

    # ── Dark theme ────────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #313244;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 4px 10px;
                color: #cdd6f4;
            }
            QPushButton:hover { background-color: #45475a; }
            QPushButton:pressed { background-color: #585b70; }
            QPushButton:checked { background-color: #45475a; border-color: #89b4fa; color: #89b4fa; }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #313244;
                border: 1px solid #585b70;
                border-radius: 3px;
                color: #cdd6f4;
                padding: 2px;
            }
            QTabWidget::pane { border: 1px solid #45475a; }
            QTabBar::tab {
                background: #313244;
                color: #cdd6f4;
                padding: 5px 20px;
                border: 1px solid #45475a;
            }
            QTabBar::tab:selected { background: #45475a; }
            QScrollArea { border: none; }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #181825; width: 10px; height: 10px;
            }
            QScrollBar::handle { background: #585b70; border-radius: 5px; min-height: 20px; }
            QCheckBox { color: #a6e3a1; }
            QSplitter::handle { background: #45475a; width: 4px; }
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    win = MainWindow(initial_image_path=initial)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
