"""
Classic CV pipeline for SEM dendrite segmentation.

Four-stage pipeline:
  A. Pre-processing  — histogram normalization, CLAHE, bilateral filter
  B. Segmentation    — adaptive thresholding (primary), Otsu (fallback)
  C. Post-processing — morphological reconstruction, closing, small component removal
  D. Separation      — distance transform + watershed for touching branches

Plus skeletonization via Zhang-Suen thinning.
"""

import argparse
import cv2
import numpy as np
import os
import sys

from skimage.morphology import reconstruction, skeletonize

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_image, save_image, list_images, clean_sem_image

# ---------------------------------------------------------------------------
# Tunable parameters (all constants at top for easy adjustment)
# ---------------------------------------------------------------------------

# Stage A: Pre-processing
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = 8

BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
BILATERAL_PASSES = 2

GAUSSIAN_KSIZE = 5

# Stage B: Segmentation
ADAPTIVE_BLOCK_SIZE = 51
ADAPTIVE_C = 5

# Stage C: Post-processing
OPENING_KERNEL_SIZE = 3
EROSION_KERNEL_SIZE = 3
EROSION_ITERATIONS = 2
CLOSING_KERNEL_SIZE = 5
MIN_COMPONENT_AREA = 50

# Substrate suppression (bottom bright electrode region)
SUBSTRATE_ROW_FG_THRESHOLD = 0.85
SUBSTRATE_MIN_HEIGHT_FRACTION = 0.06
SUBSTRATE_MARGIN_ROWS = 2

# Stage D: Separation
DISTANCE_THRESHOLD = 0.4  # fraction of max distance for watershed markers


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


def apply_gaussian_blur(image):
    """
    Moderate Gaussian smoothing to further suppress noise after bilateral.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.

    Returns
    -------
    blurred : np.ndarray
        Smoothed image.
    """
    return cv2.GaussianBlur(image, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0)


def preprocess(image):
    """
    Full pre-processing pipeline: clean → normalize → CLAHE → bilateral.

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
    smoothed = apply_gaussian_blur(bilateral_img)

    intermediates = {
        "01_original": image,
        "02_cleaned": cleaned,
        "03_normalized": normalized,
        "04_clahe": clahe_img,
        "05_bilateral": bilateral_img,
        "06_smoothed": smoothed,
    }
    return smoothed, intermediates


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
        ADAPTIVE_C
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
        fg_ratio = np.sum(binary_mask > 0) / binary_mask.size
        if fg_ratio > 0.5:
            binary_mask = cv2.bitwise_not(binary_mask)
        return binary_mask

    def _mask_quality(binary_mask):
        # Evaluate only the upper region where dendrites are expected.
        top = binary_mask[:max(1, int(binary_mask.shape[0] * 0.8)), :]
        top_fg_ratio = np.sum(top > 0) / top.size

        # Noise proxy: ratio of tiny connected components to substantial ones.
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

    adaptive = _normalize_polarity(segment_adaptive(image))
    otsu = _normalize_polarity(segment_otsu(image))

    a_fg, a_noise = _mask_quality(adaptive)
    o_fg, _ = _mask_quality(otsu)

    adaptive_plausible = (0.02 <= a_fg <= 0.55) and (a_noise <= 0.40)
    if adaptive_plausible and (a_fg >= 0.80 * o_fg or o_fg < 0.08):
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


def apply_closing(mask):
    """
    Morphological closing to fill small holes and ensure branch continuity.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    closed : np.ndarray
        Closed binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def remove_small_components(mask, min_area=None):
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
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def remove_substrate_band(mask):
    """
    Remove a dense bottom foreground band caused by bright substrate leakage.

    Detects a contiguous trailing run of rows near the image bottom with very
    high foreground occupancy and zeros it out.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 or 255), dtype uint8.

    Returns
    -------
    cleaned : np.ndarray
        Mask with bottom substrate band removed if detected.
    """
    binary = mask > 0
    h = binary.shape[0]
    row_fg = np.mean(binary, axis=1)

    i = h - 1
    run = 0
    while i >= 0 and row_fg[i] >= SUBSTRATE_ROW_FG_THRESHOLD:
        run += 1
        i -= 1

    min_rows = max(8, int(h * SUBSTRATE_MIN_HEIGHT_FRACTION))
    if run < min_rows:
        return mask

    cutoff = i + 1
    cutoff = max(0, cutoff - SUBSTRATE_MARGIN_ROWS)
    cleaned = mask.copy()
    cleaned[cutoff:, :] = 0
    return cleaned


def remove_bottom_horizontal_artifacts(mask):
    """
    Remove thin horizontal artifacts near the bottom that span most of image
    width (typically residual substrate/interface lines).

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

        near_bottom = y >= int(h * 0.5)
        lower_band = y >= int(h * 0.65)
        spans_width = (x <= 1) and (x + comp_w >= w - 1)
        thin_horizontal = comp_h <= 4 and comp_w >= int(w * 0.60)
        short_flat_stub = comp_h <= 3 and comp_w >= 20

        if near_bottom and (spans_width or thin_horizontal or (lower_band and short_flat_stub)):
            cleaned[labels == i] = 0

    return cleaned


def postprocess(mask):
    """
    Full post-processing pipeline:
    opening → reconstruction → closing → small component removal.

    Parameters
    ----------
    mask : np.ndarray
        Raw binary segmentation mask.

    Returns
    -------
    result : np.ndarray
        Cleaned binary mask.
    intermediates : dict
        Dictionary of intermediate masks.
    """
    # Kill isolated noise pixels with morphological opening
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    recon = morphological_reconstruction(opened)
    closed = apply_closing(recon)

    # Scale minimum area with image size (at least 0.01% of pixels)
    min_area = max(MIN_COMPONENT_AREA, int(mask.size * 0.0001))
    cleaned = remove_small_components(closed, min_area=min_area)
    cleaned = remove_substrate_band(cleaned)
    cleaned = remove_bottom_horizontal_artifacts(cleaned)

    intermediates = {
        "08_opened": opened,
        "09_reconstructed": recon,
        "10_closed": closed,
        "11_small_removed": cleaned,
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
    binary = (mask > 0).astype(bool)
    skel = skeletonize(binary)
    return (skel.astype(np.uint8) * 255)


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
    preprocess_ints["07_segmented"] = seg_mask

    # Stage C: Post-processing
    clean_mask, postprocess_ints = postprocess(seg_mask)

    # Stage D: Separation
    separated = separate_branches(clean_mask)

    # Skeletonization
    skeleton = skeletonize_mask(separated)

    # Collect all intermediates
    all_intermediates = {}
    all_intermediates.update(preprocess_ints)
    all_intermediates.update(postprocess_ints)
    all_intermediates["12_separated"] = separated
    all_intermediates["13_skeleton"] = skeleton

    # Save results
    if output_dir and save_intermediates:
        img_out_dir = os.path.join(output_dir, basename)
        os.makedirs(img_out_dir, exist_ok=True)
        for name, img in all_intermediates.items():
            save_image(img, os.path.join(img_out_dir, f"{name}.png"))
        print(f"  Saved {len(all_intermediates)} intermediate images to {img_out_dir}/")
    elif output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_image(separated, os.path.join(output_dir, f"{basename}_mask.png"))
        save_image(skeleton, os.path.join(output_dir, f"{basename}_skeleton.png"))

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
