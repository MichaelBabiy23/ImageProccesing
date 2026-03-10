"""
Classic CV pipeline for SEM dendrite segmentation.

Four-stage pipeline:
  A. Pre-processing  — histogram normalization, CLAHE, bilateral filter
  B. Segmentation    — adaptive thresholding (primary), Otsu (fallback)
  C. Post-processing — morphological reconstruction,
                       small component removal, substrate band removal
                       (two-stage: smoothed FG + intensity profile),
                       top band removal, edge noise removal,
                       horizontal artifact removal
  D. Separation      — distance transform + watershed for touching branches

Plus skeletonization via Zhang-Suen thinning with pruning and spur removal.

Usage:
  python classic_pipeline.py image.tif              # single image
  python classic_pipeline.py --input data/raw/Easy  # batch directory
  python classic_pipeline.py                        # synthetic self-test or real data
"""

import argparse
import cv2
import numpy as np
import os
import sys

from skimage.morphology import reconstruction, skeletonize

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_image, save_image, list_images, clean_sem_image, create_overlay

# ---------------------------------------------------------------------------
# Tunable parameters (all constants at top for easy adjustment)
# ---------------------------------------------------------------------------

# Stage A: Pre-processing
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8

BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 50  # TODO setting it this high with a BILATERAL_D of 9 basically means nothing.
BILATERAL_SIGMA_SPACE = 50
BILATERAL_PASSES = 1

# Stage B: Segmentation
ADAPTIVE_BLOCK_SIZE = 67
ADAPTIVE_C = -12

# Stage C: Post-processing
EROSION_KERNEL_SIZE = 3
EROSION_ITERATIONS = 1
MIN_COMPONENT_AREA = 90
MIN_COMPONENT_AREA_FRACTION = 0.00005
MIN_COMPONENT_AREA_MAX = 320
NOISE_COMPONENT_COUNT_THRESHOLD = 25000
RECON_MIN_KEEP_RATIO = 0.72
BASELINE_DETECT_MIN_ROW_RATIO = 0.80
BASELINE_DETECT_SEARCH_START_RATIO = 0.60
SMALL_TREE_BAND_HEIGHT = 30

# Substrate suppression (bottom bright electrode region — "ground of the trees")
SUBSTRATE_ROW_FG_THRESHOLD = 0.50       # smoothed row-FG threshold for Stage 1
SUBSTRATE_MIN_HEIGHT_FRACTION = 0.06    # minimum substrate band height as fraction of image
SUBSTRATE_MARGIN_ROWS = 2              # extra rows to remove above detected cutoff
SUBSTRATE_FG_WINDOW = 20               # moving-average window (rows) for FG smoothing
SUBSTRATE_INTENSITY_JUMP_RATIO = 0.25  # min contrast ratio (bottom vs top) for Stage 2
SUBSTRATE_INTENSITY_SMOOTH_K = 41      # kernel size for row-mean intensity smoothing
SUBSTRATE_INTENSITY_MIN_RUN = 50       # min contiguous bright rows for intensity detection
SUBSTRATE_INTENSITY_MIN_CUTOFF_RATIO = 0.55   # Stage 2 cutoff must stay in lower 45%
SUBSTRATE_INTENSITY_MIN_KEEP_RATIO = 0.60     # reject Stage 2 if it erases too much mask
SUBSTRATE_INTENSITY_MAX_BOTTOM_FG_RATIO = 0.80  # bottom band must be visibly sparser
SUBSTRATE_MAX_ROWS_ABOVE_BASELINE = 8  # cap how far substrate removal can climb above baseline

# Top-band suppression (bright header / nanopore pattern at image top)
TOP_BAND_FG_THRESHOLD = 0.30           # row-FG threshold for top-band detection
TOP_BAND_MIN_ROWS = 20                 # minimum run of dense rows to trigger removal
TOP_BAND_MAX_FRACTION = 0.15           # cap: never remove more than 15% of image height

# Edge noise removal (tiny specks actually touching the left/right borders)
EDGE_COMPONENT_MAX_AREA = 80           # max area for a component to be considered edge noise
EDGE_COMPONENT_MAX_WIDTH_FRACTION = 0.015
EDGE_COMPONENT_MAX_HEIGHT_FRACTION = 0.05

# Stage D: Separation
DISTANCE_THRESHOLD = 0.35  # fraction of max distance for watershed markers

# Skeleton pruning
SKELETON_MIN_BRANCH_LENGTH = 8
SKELETON_SPUR_LENGTH = 6
SKELETON_HORIZONTAL_LINE_MIN_WIDTH = 40


# ===========================================================================
# Stage A: Pre-processing
# ===========================================================================

def normalize_histogram(image):
    """
    Linear stretch of pixel values to the full [0, 255] range.
    Ensures a common basis across images with different exposures.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.

    Returns
    -------
    normalized : np.ndarray
        Image with values stretched to [0, 255].
    """
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val == min_val:
        return np.zeros_like(image)
    normalized = ((image.astype(np.float64) - min_val) / (max_val - min_val) * 255)
    return normalized.astype(np.uint8)


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    Divides the image into tiles and equalizes each independently with
    a clip limit to prevent noise amplification.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.

    Returns
    -------
    enhanced : np.ndarray
        Contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
    )
    return clahe.apply(image)


def apply_bilateral_filter(image):
    """
    Edge-preserving denoising via bilateral filtering.
    Applies multiple passes to suppress SEM shot noise while preserving
    sharp dendrite edges.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.

    Returns
    -------
    filtered : np.ndarray
        Denoised image with edges preserved.
    """
    result = image
    for _ in range(BILATERAL_PASSES):
        result = cv2.bilateralFilter(
            result, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
        )
    return result


def preprocess(image):
    """
    Full pre-processing pipeline: clean, normalize, CLAHE, bilateral.

    Parameters
    ----------
    image : np.ndarray
        Raw grayscale SEM image (H, W).

    Returns
    -------
    result : np.ndarray
        Pre-processed image ready for segmentation.
    intermediates : dict
        Dictionary of intermediate images for visualization.
    """
    cleaned = clean_sem_image(image)
    normalized = normalize_histogram(cleaned)
    clahe_img = apply_clahe(normalized)
    bilateral_img = apply_bilateral_filter(clahe_img)

    intermediates = {
        "01_original": image,
        "02_cleaned": cleaned,
        "03_normalized": normalized,
        "04_clahe": clahe_img,
        "05_bilateral": bilateral_img,
    }
    return bilateral_img, intermediates


# ===========================================================================
# Stage B: Segmentation
# ===========================================================================

def segment_adaptive(image):
    """
    Adaptive thresholding — computes a local threshold per pixel based on
    neighborhood mean. Preferred for SEM images with non-uniform illumination.

    Parameters
    ----------
    image : np.ndarray
        Pre-processed grayscale image (H, W).

    Returns
    -------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.
    """
    mask = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C # we got to C after intense testing (landed on -12)
    )
    return mask


def segment_otsu(image):
    """
    Otsu's binarization — finds the optimal global threshold minimizing
    within-class variance. Fallback for uniformly illuminated images.

    Parameters
    ----------
    image : np.ndarray
        Pre-processed grayscale image (H, W).

    Returns
    -------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.
    """
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment(image):
    """
    Segment bright dendrites from dark background.
    Produces both adaptive and Otsu masks, then selects the better one using
    simple quality heuristics:
      - adaptive is chosen only if it has plausible foreground ratio and low
        small-component noise in the upper image region
      - otherwise Otsu is used
    Auto-corrects inverted masks (dendrites should be minority foreground).

    Parameters
    ----------
    image : np.ndarray
        Pre-processed grayscale image (H, W).

    Returns
    -------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.  Dendrites = 255.
    """
    def _normalize_polarity(binary_mask):
        """Ensure dendrites are foreground (minority). Invert if FG > 50%."""
        fg_ratio = np.sum(binary_mask > 0) / binary_mask.size
        if fg_ratio > 0.5:
            binary_mask = cv2.bitwise_not(binary_mask)
        return binary_mask

    def _mask_quality(binary_mask):
        """Compute FG ratio and noise ratio in the upper 80% of the image."""
        # Evaluate only the upper region where dendrites are expected
        top = binary_mask[:max(1, int(binary_mask.shape[0] * 0.8)), :]
        top_fg_ratio = np.sum(top > 0) / top.size

        # Noise proxy: ratio of tiny (<20px) components to substantial ones
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(top, connectivity=8)
        small = 0
        large = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:
                small += 1
            else:
                large += 1
        noise_ratio = small / max(1, large)
        return top_fg_ratio, noise_ratio

    # Generate both masks and normalize polarity (dendrites = white)
    adaptive = _normalize_polarity(segment_adaptive(image))
    otsu = _normalize_polarity(segment_otsu(image))

    # Evaluate quality of each mask
    a_fg, a_noise = _mask_quality(adaptive)
    o_fg, _ = _mask_quality(otsu)

    # Prefer adaptive whenever FG ratio is plausible for SEM dendrites.
    # Tiny-component specks are handled in post-processing.
    adaptive_plausible = 0.01 <= a_fg <= 0.55
    if adaptive_plausible:
        return adaptive

    return otsu


# ===========================================================================
# Stage C: Post-processing
# ===========================================================================

def morphological_reconstruction(mask):
    """
    Geodesic dilation-based reconstruction to remove noise while preserving
    dendrite structure.

    Process:
      1. Aggressively erode the mask to keep only thick branch cores (marker)
      2. Use the original mask as the limit (mask image)
      3. Dilate the marker iteratively within the mask boundaries

    This removes noise without damaging thin dendrite branches (unlike
    standard morphological opening).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    reconstructed : np.ndarray
        Cleaned binary mask (0 or 255), dtype uint8.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE)
    )
    # Create marker by aggressive erosion — only thick cores remain
    marker = cv2.erode(mask, kernel, iterations=EROSION_ITERATIONS)

    # skimage reconstruction expects float images in [0, 1]
    marker_f = (marker / 255.0).astype(np.float64)
    mask_f = (mask / 255.0).astype(np.float64)

    # Geodesic dilation: grow marker within mask boundaries
    reconstructed_f = reconstruction(marker_f, mask_f, method='dilation')

    reconstructed = (reconstructed_f * 255).astype(np.uint8)
    return reconstructed


def remove_small_components(mask, min_area=None, baseline_row=None):
    """
    Remove connected components smaller than min_area pixels.
    Based on the physical assumption that dendrites are large,
    continuous structures.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.
    min_area : int or None
        Minimum component area in pixels. Uses MIN_COMPONENT_AREA if None.
    baseline_row : int or None
        Optional baseline row. Components whose bottom lies in a small band
        above this row are preserved even if they are below min_area.

    Returns
    -------
    cleaned : np.ndarray
        Mask with small components removed.
    """
    if min_area is None:
        min_area = MIN_COMPONENT_AREA

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    band_top = None
    if baseline_row is not None:
        band_top = max(0, int(baseline_row) - SMALL_TREE_BAND_HEIGHT)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        top = int(stats[i, cv2.CC_STAT_TOP])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        bottom = top + height - 1

        # Preserve small tree stumps right above the substrate baseline.
        if band_top is not None and band_top <= bottom <= int(baseline_row):
            cleaned[labels == i] = 255
            continue

        if area >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def choose_min_component_area(mask):
    """
    Choose an adaptive small-component threshold for the current image/mask.

    The goal is to avoid removing thin dendrite fragments on sparse images
    while still suppressing heavy speckle noise on high-resolution/noisy images.
    """
    base = max(MIN_COMPONENT_AREA, int(mask.size * MIN_COMPONENT_AREA_FRACTION))
    fg_ratio = np.count_nonzero(mask) / float(mask.size)

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_count = max(0, num_labels - 1)

    # Dense masks and huge component counts indicate noisy images.
    if fg_ratio > 0.25:
        base = max(base, int(MIN_COMPONENT_AREA * 1.5))
    if component_count > NOISE_COMPONENT_COUNT_THRESHOLD:
        base = max(base, int(MIN_COMPONENT_AREA * 1.4))

    return int(min(base, MIN_COMPONENT_AREA_MAX))


def detect_baseline_row(mask):
    """
    Detect a baseline row where the bottom substrate region starts.
    """
    if mask is None or mask.ndim != 2 or mask.size == 0:
        return None

    h = mask.shape[0]
    y_start = min(h - 1, int(round(h * BASELINE_DETECT_SEARCH_START_RATIO)))
    row_ratio = np.mean(mask > 0, axis=1)
    idx = np.flatnonzero(row_ratio[y_start:] >= BASELINE_DETECT_MIN_ROW_RATIO)
    if idx.size == 0:
        return None
    return y_start + int(idx[0])


def zero_below_baseline(mask, baseline_row):
    """
    Zero baseline row and everything below it.
    """
    if baseline_row is None:
        return mask
    out = mask.copy()
    out[int(baseline_row):, :] = 0
    return out


def remove_substrate_band(mask, preprocessed=None, baseline_row=None):
    """
    Remove the bright substrate/electrode base ("ground of the trees") where
    dendrites grow from.  Uses a two-stage approach to handle both dense
    (Easy images) and grain-boundary (Hard Group B) substrate regions.

    Stage 1 — Smoothed foreground scan:
        Smooth per-row FG ratio with a moving average, then scan from the
        bottom upward looking for a contiguous run of rows whose smoothed
        FG >= SUBSTRATE_ROW_FG_THRESHOLD (0.50).  The moving average lets us
        catch grain-boundary regions where raw per-row FG oscillates 30-70%.

    Stage 2 — Intensity-profile detection (only if Stage 1 found nothing):
        If a preprocessed image is provided, compute the row-mean intensity
        profile and look for a bright substrate region at the bottom whose
        average intensity is significantly higher than the dark upper region.
        The transition row where brightness rises is used as the cutoff.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.
    preprocessed : np.ndarray or None
        Pre-processed grayscale image (H, W), used for Stage 2 intensity
        analysis.  If None, only Stage 1 is applied (backward compatible).

    baseline_row : int or None
        Optional detected baseline row. If available, substrate removal is
        prevented from climbing too far above that baseline.

    Returns
    -------
    cleaned : np.ndarray
        Mask with bottom substrate band removed if detected.
    """
    binary = mask > 0
    h, w = binary.shape
    row_fg = np.mean(binary, axis=1)  # per-row foreground ratio [0, 1]

    def clamp_cutoff(cutoff):
        if baseline_row is None:
            return cutoff
        return max(int(cutoff), max(0, int(baseline_row) - SUBSTRATE_MAX_ROWS_ABOVE_BASELINE))

    # --- Stage 1: Smoothed FG scan from bottom ---
    # Apply moving average to smooth out oscillations in grain boundaries
    win = SUBSTRATE_FG_WINDOW
    if h >= win:
        kernel = np.ones(win) / win
        smoothed_fg = np.convolve(row_fg, kernel, mode='same')
    else:
        smoothed_fg = row_fg

    # Scan from bottom: find contiguous run where smoothed FG >= threshold
    i = h - 1
    run = 0
    while i >= 0 and smoothed_fg[i] >= SUBSTRATE_ROW_FG_THRESHOLD:
        run += 1
        i -= 1

    min_rows = max(8, int(h * SUBSTRATE_MIN_HEIGHT_FRACTION))

    if run >= min_rows:
        # Stage 1 found substrate — zero everything below cutoff
        cutoff = clamp_cutoff(max(0, (i + 1) - SUBSTRATE_MARGIN_ROWS))
        cleaned = mask.copy()
        cleaned[cutoff:, :] = 0
        return cleaned

    # --- Stage 2: Intensity-profile detection (needs preprocessed image) ---
    if preprocessed is None:
        return mask  # no preprocessed image, nothing more we can do

    # Compute row-mean intensity and smooth it
    row_intensity = np.mean(preprocessed.astype(np.float64), axis=1)
    k = min(SUBSTRATE_INTENSITY_SMOOTH_K, h)
    if k % 2 == 0:
        k = max(1, k - 1)  # ensure odd kernel size
    smooth_kernel = np.ones(k) / k
    smoothed_intensity = np.convolve(row_intensity, smooth_kernel, mode='same')

    # Compare top 20% (dark dendrite region) vs bottom 60-88% (potential substrate)
    top_end = int(h * 0.20)
    bot_start = int(h * 0.60)
    bot_end = int(h * 0.88)

    if top_end < 1 or bot_start >= bot_end:
        return mask  # image too small for meaningful analysis

    top_median = np.median(smoothed_intensity[:top_end])
    bot_median = np.median(smoothed_intensity[bot_start:bot_end])

    # Check if bottom is significantly brighter than top
    if top_median <= 0 or (bot_median - top_median) / max(1.0, bot_median) < SUBSTRATE_INTENSITY_JUMP_RATIO:
        return mask  # not enough contrast — no clear substrate

    # Find transition row: scan upward from bottom, find where intensity
    # drops below the dark region's baseline + a margin.
    threshold_intensity = top_median + 0.35 * (bot_median - top_median)
    transition = h - 1
    run_count = 0
    for r in range(h - 1, -1, -1):
        if smoothed_intensity[r] >= threshold_intensity:
            run_count += 1
        else:
            if run_count >= SUBSTRATE_INTENSITY_MIN_RUN:
                transition = r + 1
                break
            run_count = 0
    else:
        # Scanned all the way to top without breaking
        if run_count >= SUBSTRATE_INTENSITY_MIN_RUN:
            transition = 0

    if transition >= h - min_rows:
        return mask  # transition too close to bottom, not meaningful

    cutoff = clamp_cutoff(max(0, transition - SUBSTRATE_MARGIN_ROWS))
    cutoff_ratio = cutoff / float(h)

    # Stage 2 is prone to cutting through real trees on Hard images.
    # Only trust it when the candidate band is low enough in the frame,
    # materially sparser than the region above, and does not erase most
    # of the foreground mask.
    if cutoff_ratio < SUBSTRATE_INTENSITY_MIN_CUTOFF_RATIO:
        return mask

    above_fg = float(np.mean(row_fg[:cutoff])) if cutoff > 0 else 0.0
    bottom_fg = float(np.mean(row_fg[cutoff:])) if cutoff < h else 0.0
    if above_fg <= 0:
        return mask
    if bottom_fg > above_fg * SUBSTRATE_INTENSITY_MAX_BOTTOM_FG_RATIO:
        return mask

    cleaned = mask.copy()
    cleaned[cutoff:, :] = 0
    keep_ratio = np.count_nonzero(cleaned) / float(max(1, np.count_nonzero(mask)))
    if keep_ratio < SUBSTRATE_INTENSITY_MIN_KEEP_RATIO:
        return mask
    return cleaned


def remove_top_band(mask):
    """
    Remove a dense foreground band at the top of the image.

    Some Hard images (e.g., 70nm_pitch_035) have a bright nanopore/nanowire
    pattern in the top rows that gets falsely segmented.  This function scans
    from row 0 downward, finds a contiguous run of rows with high FG ratio,
    and zeros them out.

    Safety cap: never removes more than TOP_BAND_MAX_FRACTION (15%) of image
    height to avoid accidentally deleting real dendrites.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    cleaned : np.ndarray
        Mask with top band removed if detected.
    """
    binary = mask > 0
    h = binary.shape[0]
    row_fg = np.mean(binary, axis=1)

    # Scan from top: find contiguous run where FG >= threshold
    run = 0
    for i in range(h):
        if row_fg[i] >= TOP_BAND_FG_THRESHOLD:
            run += 1
        else:
            break  # first gap ends the run

    if run < TOP_BAND_MIN_ROWS:
        return mask  # no significant top band found

    # Cap removal at a fraction of image height to prevent over-removal
    max_remove = int(h * TOP_BAND_MAX_FRACTION)
    cutoff = min(run, max_remove)

    cleaned = mask.copy()
    cleaned[:cutoff, :] = 0
    return cleaned


def remove_edge_noise(mask):
    """
    Remove tiny connected components that actually touch the left or right
    image borders.

    SEM images sometimes have noise or annotation artifacts near the edges.
    Real dendrites can legitimately approach the sides, so this filter is
    deliberately conservative: a component must touch a border, be tiny,
    and remain spatially compact to be removed.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    cleaned : np.ndarray
        Mask with edge-hugging small components removed.
    """
    h, w = mask.shape[:2]
    max_w = max(12, int(w * EDGE_COMPONENT_MAX_WIDTH_FRACTION))
    max_h = max(20, int(h * EDGE_COMPONENT_MAX_HEIGHT_FRACTION))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = mask.copy()

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        touches_left = x <= 1
        touches_right = (x + comp_w) >= (w - 1)
        small_enough = area <= EDGE_COMPONENT_MAX_AREA
        compact = comp_w <= max_w and comp_h <= max_h

        if (touches_left or touches_right) and small_enough and compact:
            cleaned[labels == i] = 0

    return cleaned


def remove_bottom_horizontal_artifacts(mask):
    """
    Remove thin horizontal artifacts near the bottom that span most of image
    width (typically residual substrate/interface lines).

    Targets three patterns:
      - Components that span the full image width (edge-to-edge)
      - Thin horizontal strips (height <= 4, width >= 60% of image)
      - Short flat stubs in the lower 35% (height <= 3, width >= 20)

    All must originate in the bottom half of the image.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    cleaned : np.ndarray
        Mask with bottom spanning-line artifacts removed.
    """
    h, w = mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = mask.copy()

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        # Geometric tests for horizontal line-like components
        near_bottom = y >= int(h * 0.5)          # bottom half of image
        lower_band = y >= int(h * 0.65)           # bottom 35%
        spans_width = (x <= 1) and (x + comp_w >= w - 1)  # edge to edge
        thin_horizontal = comp_h <= 4 and comp_w >= int(w * 0.60)
        short_flat_stub = comp_h <= 3 and comp_w >= 20

        if near_bottom and (spans_width or thin_horizontal or (lower_band and short_flat_stub)):
            cleaned[labels == i] = 0

    return cleaned


def postprocess(mask, preprocessed=None):
    """
    Full post-processing pipeline:
    reconstruction → component cleanup → band removal.

    The cleanup chain after reconstruction is:
      1. detect baseline + cut below it
      2. remove substrate/top/edge/horizontal artifacts
      3. remove_small_components (with baseline-band preservation)

    Parameters
    ----------
    mask : np.ndarray
        Raw binary segmentation mask (0 or 255).
    preprocessed : np.ndarray or None
        Pre-processed grayscale image, passed to remove_substrate_band()
        for intensity-based substrate detection (Stage 2).

    Returns
    -------
    result : np.ndarray
        Cleaned binary mask.
    intermediates : dict
        Dictionary of intermediate masks for visualization.
    """
    recon = morphological_reconstruction(mask)
    pre_recon_area = int(np.count_nonzero(mask))
    recon_area = int(np.count_nonzero(recon))
    keep_ratio = (recon_area / float(pre_recon_area)) if pre_recon_area > 0 else 1.0
    if keep_ratio < RECON_MIN_KEEP_RATIO:
        # Reconstruction can erase thin trees on sparse images.
        # If too much is lost, keep the pre-reconstruction mask.
        recon = mask.copy()
        print(
            f"  Reconstruction fallback: keep={keep_ratio:.1%} < "
            f"{RECON_MIN_KEEP_RATIO:.0%}; using pre-reconstruction mask."
        )

    # Remove known geometric artifacts before size filtering.
    baseline_row = detect_baseline_row(recon)
    after_baseline = zero_below_baseline(recon, baseline_row)
    after_substrate = remove_substrate_band(
        after_baseline,
        preprocessed,
        baseline_row=baseline_row,
    )
    after_top = remove_top_band(after_substrate)
    after_edge = remove_edge_noise(after_top)
    after_bottom_artifacts = remove_bottom_horizontal_artifacts(after_edge)

    # Adaptive small-component filtering to preserve sparse dendrite tips.
    min_area = choose_min_component_area(after_bottom_artifacts)
    cleaned = remove_small_components(
        after_bottom_artifacts,
        min_area=min_area,
        baseline_row=baseline_row,
    )

    intermediates = {
        "07_reconstructed": recon,
        "08a_after_baseline_cut": after_baseline,
        "08b_after_substrate_removed": after_substrate,
        "08c_after_top_removed": after_top,
        "08d_after_edge_removed": after_edge,
        "08e_after_bottom_artifact_removed": after_bottom_artifacts,
        "08_small_removed": cleaned,
    }
    return cleaned, intermediates


# ===========================================================================
# Stage D: Separation (Distance Transform + Watershed)
# ===========================================================================

def separate_branches(mask):
    """
    Separate touching dendrite branches using distance transform and watershed.

    Process:
      1. Compute distance transform of the binary mask
      2. Threshold at a fraction of the maximum distance → foreground markers
      3. Identify background (far from any foreground)
      4. Label markers with connected components
      5. Run watershed to find boundaries between touching branches

    Parameters
    ----------
    mask : np.ndarray
        Clean binary mask (0 or 255), dtype uint8.

    Returns
    -------
    separated : np.ndarray
        Binary mask with touching branches separated.
    """
    if np.sum(mask) == 0:
        return mask.copy()

    # Distance transform
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Threshold to find sure foreground (branch cores)
    _, sure_fg = cv2.threshold(
        dist, DISTANCE_THRESHOLD * dist.max(), 255, cv2.THRESH_BINARY
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background — region far from any foreground
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Unknown region — between sure foreground and sure background
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers for watershed
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # background = 1, not 0
    markers[unknown == 255] = 0  # unknown = 0 (watershed will determine)

    # Watershed needs 3-channel input
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_color, markers)

    # Build separated mask: watershed boundaries are marked as -1
    separated = mask.copy()
    separated[markers == -1] = 0

    return separated


# ===========================================================================
# Skeletonization
# ===========================================================================

def skeletonize_mask(mask):
    """
    Extract single-pixel-width centerline skeleton from binary mask.
    Uses Zhang-Suen thinning algorithm via scikit-image.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    skeleton : np.ndarray
        Skeleton image (0 or 255), dtype uint8.
    """
    # Smooth jagged mask edges with a small closing to prevent diamond/loop
    # artifacts in the skeleton.  Closing only adds pixels (fills 1px notches)
    # so it does not alter branch topology.
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_kernel)
    binary = (smoothed > 0).astype(bool)
    skel = skeletonize(binary)
    skeleton = skel.astype(np.uint8) * 255
    skeleton = prune_skeleton(skeleton)
    skeleton = remove_spurs(skeleton)
    return skeleton


def prune_skeleton(skeleton):
    """
    Remove skeleton artifacts: tiny fragments and horizontal line remnants.

    Removes connected components that are either:
      - Smaller than SKELETON_MIN_BRANCH_LENGTH pixels (tiny stubs), or
      - Horizontal lines (height <= 3 and width >= SKELETON_HORIZONTAL_LINE_MIN_WIDTH).

    Parameters
    ----------
    skeleton : np.ndarray
        Skeleton image (0 or 255), dtype uint8.

    Returns
    -------
    pruned : np.ndarray
        Cleaned skeleton image (0 or 255), dtype uint8.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8
    )
    pruned = skeleton.copy()
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        is_small = area < SKELETON_MIN_BRANCH_LENGTH
        is_horizontal_line = h <= 3 and w >= SKELETON_HORIZONTAL_LINE_MIN_WIDTH
        if is_small or is_horizontal_line:
            pruned[labels == i] = 0
    return pruned


def remove_spurs(skeleton):
    """
    Iteratively remove short terminal branches (spurs) from a skeleton.

    A spur is a path from an endpoint (1 neighbor) to the nearest junction
    (3+ neighbors).  If that path is <= SKELETON_SPUR_LENGTH pixels the spur
    is erased (the junction pixel itself is kept).

    Iteration is needed because removing a spur may turn a former junction
    into a new endpoint, revealing another spur.

    Parameters
    ----------
    skeleton : np.ndarray
        Skeleton image (0 or 255), dtype uint8.

    Returns
    -------
    cleaned : np.ndarray
        Skeleton with short spurs removed, same dtype.
    """
    skel = skeleton.copy()
    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], dtype=np.uint8)

    while True:
        binary = (skel > 0).astype(np.uint8)
        neighbor_count = cv2.filter2D(binary, cv2.CV_16S, neighbor_kernel)
        neighbor_count = neighbor_count.astype(np.int16)

        # Endpoints: skeleton pixel with exactly 1 neighbor
        endpoints = (binary == 1) & (neighbor_count == 1)
        ep_coords = list(zip(*np.where(endpoints)))
        if not ep_coords:
            break

        removed_any = False
        for r, c in ep_coords:
            if skel[r, c] == 0:
                continue  # already removed in this pass

            # Trace from endpoint toward junction
            path = [(r, c)]
            cr, cc = r, c
            visited = {(cr, cc)}

            while True:
                # Look at 8-connected neighbors
                found_next = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (nr, nc) in visited:
                            continue
                        if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] and skel[nr, nc] > 0:
                            nb = int(neighbor_count[nr, nc])
                            if nb >= 3:
                                # Reached a junction — stop (don't include it)
                                found_next = False
                                break
                            # Regular continuation pixel
                            path.append((nr, nc))
                            visited.add((nr, nc))
                            cr, cc = nr, nc
                            found_next = True
                            break
                    else:
                        continue
                    break

                if not found_next:
                    break

            if len(path) <= SKELETON_SPUR_LENGTH:
                for pr, pc in path:
                    skel[pr, pc] = 0
                removed_any = True

        if not removed_any:
            break

    return skel


# ===========================================================================
# Pipeline orchestration
# ===========================================================================

def run_classic_pipeline(image_path, output_dir=None, save_intermediates=True):
    """
    Run the full classic segmentation pipeline on a single SEM image.

    Parameters
    ----------
    image_path : str
        Path to input SEM image.
    output_dir : str or None
        Directory to save results. If None, results are not saved.
    save_intermediates : bool
        If True, save every intermediate image for analysis/reporting.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'mask': final binary segmentation mask
        - 'skeleton': single-pixel skeleton
        - 'separated': mask after branch separation
        - 'intermediates': dict of all intermediate images
    """
    # Load image
    image = load_image(image_path, grayscale=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing: {basename} ({image.shape[1]}x{image.shape[0]})")

    # Stage A: Pre-processing
    preprocessed, preprocess_ints = preprocess(image)

    # Stage B: Segmentation (adaptive-first with fallback + auto-inversion)
    seg_mask = segment(preprocessed)
    preprocess_ints["06_segmented"] = seg_mask

    # Stage C: Post-processing (pass preprocessed for intensity-based substrate detection)
    clean_mask, postprocess_ints = postprocess(seg_mask, preprocessed)

    # Stage D: Separation
    separated = separate_branches(clean_mask)

    # Skeletonization
    skeleton = skeletonize_mask(separated)

    # Collect all intermediates
    all_intermediates = {}
    all_intermediates.update(preprocess_ints)
    all_intermediates.update(postprocess_ints)
    all_intermediates["09_separated"] = separated
    all_intermediates["10_skeleton"] = skeleton

    # Overlays: skeleton (red) and mask (green) on original
    overlay_skel = create_overlay(image, skeleton, color=(0, 0, 255), alpha=0.70)
    overlay_mask = create_overlay(image, separated, color=(0, 255, 0), alpha=0.55)

    # Save results
    if output_dir and save_intermediates:
        img_out_dir = os.path.join(output_dir, basename)
        os.makedirs(img_out_dir, exist_ok=True)
        for name, img in all_intermediates.items():
            save_image(img, os.path.join(img_out_dir, f"{name}.png"))
        save_image(overlay_skel, os.path.join(img_out_dir, "overlay_skel_on_orig.png"))
        save_image(overlay_mask, os.path.join(img_out_dir, "overlay_mask_on_orig.png"))
        print(f"  Saved {len(all_intermediates) + 2} intermediate images to {img_out_dir}/")
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_image(separated, os.path.join(output_dir, f"{basename}_mask.png"))
        save_image(skeleton, os.path.join(output_dir, f"{basename}_skeleton.png"))
        save_image(overlay_skel, os.path.join(output_dir, f"{basename}_overlay_skel.png"))
        save_image(overlay_mask, os.path.join(output_dir, f"{basename}_overlay_mask.png"))

    results = {
        "mask": separated,
        "skeleton": skeleton,
        "separated": separated,
        "intermediates": all_intermediates,
    }
    return results


def process_all_images(input_dir, output_dir):
    """
    Batch-process all SEM images in a directory through the classic pipeline.

    Parameters
    ----------
    input_dir : str
        Directory containing input SEM images.
    output_dir : str
        Directory to save all outputs.

    Returns
    -------
    all_results : dict
        Mapping of image basename to pipeline results.
    """
    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return {}

    print(f"Found {len(image_paths)} images in {input_dir}\n")
    all_results = {}
    for path in image_paths:
        basename = os.path.splitext(os.path.basename(path))[0]
        results = run_classic_pipeline(path, output_dir, save_intermediates=True)
        all_results[basename] = results
        print()

    print(f"Batch processing complete. Results saved to {output_dir}/")
    return all_results


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    """Argparse CLI for the classic segmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="Classic CV pipeline for SEM dendrite segmentation"
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to a single SEM image (omit for batch mode with --input)"
    )
    parser.add_argument(
        "--input", default=None,
        help="Directory of SEM images for batch processing"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: output/classic/)"
    )
    parser.add_argument(
        "--no-intermediates", action="store_true",
        help="Only save final mask and skeleton, not intermediate stages"
    )

    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output or os.path.join(project_dir, "output", "classic")

    if args.image:
        # Single image mode
        if not os.path.isfile(args.image):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        run_classic_pipeline(
            args.image, output_dir,
            save_intermediates=not args.no_intermediates
        )
    elif args.input:
        # Batch mode
        if not os.path.isdir(args.input):
            print(f"Error: Directory not found: {args.input}")
            sys.exit(1)
        process_all_images(args.input, output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # No CLI args: try real data directories, fall back to synthetic test
        project_dir = os.path.dirname(os.path.abspath(__file__))
        easy_dir = os.path.join(project_dir, "data", "raw", "Easy")
        hard_dir = os.path.join(project_dir, "data", "raw", "Hard")
        output_dir = os.path.join(project_dir, "output", "classic")

        if os.path.isdir(easy_dir) or os.path.isdir(hard_dir):
            print("=== classic_pipeline.py — Processing Real Data ===\n")
            for d in [easy_dir, hard_dir]:
                if os.path.isdir(d):
                    category = os.path.basename(d)
                    cat_output = os.path.join(output_dir, category)
                    print(f"--- {category} ---")
                    process_all_images(d, cat_output)
                    print()
        else:
            # Synthetic self-test
            print("=== classic_pipeline.py — Synthetic Self-Test ===\n")

            np.random.seed(42)
            h, w = 512, 512
            synth = np.random.randint(30, 80, (h, w), dtype=np.uint8)

            # Draw some bright "dendrite" structures
            cv2.line(synth, (100, 50), (100, 400), 200, 3)
            cv2.line(synth, (100, 200), (250, 150), 190, 2)
            cv2.line(synth, (100, 300), (200, 350), 185, 2)
            cv2.line(synth, (300, 100), (300, 450), 210, 4)
            cv2.line(synth, (300, 250), (400, 200), 195, 2)
            cv2.line(synth, (300, 350), (450, 400), 180, 2)

            # Add a bright "scale bar" at bottom
            synth[460:, :] = 230

            # Save synthetic image, then process it
            test_img_path = os.path.join(project_dir, "output", "synth_dendrites.png")
            os.makedirs(os.path.dirname(test_img_path), exist_ok=True)
            cv2.imwrite(test_img_path, synth)

            out_dir = os.path.join(project_dir, "output", "classic")
            results = run_classic_pipeline(test_img_path, out_dir, save_intermediates=True)

            print(f"\nFinal mask — non-zero pixels: {np.sum(results['mask'] > 0)}")
            print(f"Skeleton   — non-zero pixels: {np.sum(results['skeleton'] > 0)}")

            for name, img in results["intermediates"].items():
                print(f"  {name}: shape={img.shape}, range=[{img.min()}, {img.max()}]")

            print("\nAll classic pipeline tests passed.")
