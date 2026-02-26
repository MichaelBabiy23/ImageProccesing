# Student Code Explanation

This document explains every file, constant, function, and key line of code I wrote for the SEM dendrite segmentation project. The goal is to show I understand what each part does and why I chose to write it that way.

---

## Table of Contents

1. [requirements.txt](#1-requirementstxt)
2. [utils.py](#2-utilspy)
3. [classic_pipeline.py](#3-classic_pipelinepy)
4. [yolo_pipeline.py](#4-yolo_pipelinepy)
5. [evaluate.py](#5-evaluatepy)
6. [run_all.py](#6-run_allpy)

---

## 1. requirements.txt

This file lists the four Python packages the project depends on:

- **numpy** — the core numerical library. I use it everywhere for array operations on images (images are just 2D numpy arrays of pixel values).
- **opencv-python** — the `cv2` module. I use it for image I/O (`imread`, `imwrite`), filtering (bilateral filter, CLAHE), morphological operations (erode, dilate, close), thresholding (adaptive, Otsu), connected components analysis, distance transform, and watershed segmentation.
- **scikit-image** — I use two specific functions from it: `skimage.morphology.reconstruction` for geodesic dilation (morphological reconstruction) and `skimage.morphology.skeletonize` for Zhang-Suen thinning to extract single-pixel skeletons.
- **ultralytics** — the YOLO library. I use it to load a pretrained YOLOv11 segmentation model, fine-tune it on my labeled SEM dendrite dataset, and run inference to get instance segmentation masks.

---

## 2. utils.py

**Purpose:** Shared helper functions used by all other files. Handles image I/O, SEM-specific image cleaning (removing scale bars and text overlays), and visualization utilities for comparison figures.

### Constants

```python
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
```
A tuple of file extensions I consider to be images. I use this when scanning a directory to filter out non-image files. I included `.tif` and `.tiff` because SEM images sometimes come in TIFF format.

```python
SCALE_BAR_FRACTION = 0.12
```
SEM images from the microscope have a metadata strip at the bottom (showing magnification, voltage, scale bar, etc.). I found that this strip is roughly the bottom 12% of the image. This constant controls how much of the bottom gets removed during cleaning.

### Functions

#### `load_image(path, grayscale=True)`

Loads an image from disk using OpenCV.

```python
flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
```
I pick the OpenCV flag based on whether I want a single-channel grayscale image (which I do for SEM processing) or a 3-channel color image.

```python
image = cv2.imread(path, flag)
if image is None:
    raise FileNotFoundError(f"Could not load image: {path}")
```
`cv2.imread` returns `None` silently when it can't find a file (instead of raising an error), so I check for that and raise a proper error with the file path so I know which image failed.

**Returns:** A numpy array — shape `(H, W)` for grayscale or `(H, W, 3)` for color.

---

#### `save_image(image, path)`

Saves an image to disk.

```python
os.makedirs(os.path.dirname(path), exist_ok=True)
```
Before saving, I create any missing parent directories. `exist_ok=True` means it won't raise an error if the directory already exists.

```python
cv2.imwrite(path, image)
```
Writes the numpy array to the file path. OpenCV picks the format (PNG, JPG, etc.) based on the file extension.

---

#### `list_images(directory)`

Lists all image files in a directory, sorted alphabetically.

```python
if not os.path.isdir(directory):
    raise FileNotFoundError(f"Directory not found: {directory}")
```
I check the directory exists first so the error message is clear.

```python
for f in sorted(os.listdir(directory)):
    if f.lower().endswith(IMAGE_EXTENSIONS):
        files.append(os.path.join(directory, f))
```
I iterate over all files in the directory (sorted so the order is deterministic), convert each filename to lowercase, and check if it ends with one of my image extensions. If it does, I build the full path with `os.path.join` and add it to the list.

**Returns:** A sorted list of full file paths to images.

---

#### `remove_scale_bar(image)`

Masks the bottom region of an SEM image where the microscope puts its metadata (magnification, scale bar, voltage text).

```python
h, w = image.shape[:2]
cutoff = int(h * (1 - SCALE_BAR_FRACTION))
```
I calculate where to cut. For a 512-pixel-tall image with `SCALE_BAR_FRACTION = 0.12`, the cutoff is at row `int(512 * 0.88) = 450`. Everything below row 450 is the metadata strip.

```python
ref_strip = image[max(0, cutoff - 20):cutoff, :]
fill_value = int(np.median(ref_strip))
```
Instead of filling with black or white (which would create a hard edge), I take the median pixel value of a thin strip (20 pixels tall) just above the metadata region. This gives me the typical background intensity in that area, so the replacement blends in smoothly.

```python
cleaned = image.copy()
cleaned[cutoff:, :] = fill_value
```
I copy the image (so I don't modify the original) and fill the bottom region with that median value.

**Returns:** The cleaned image with the bottom metadata replaced.

---

#### `remove_text_overlay(image)`

Detects and removes bright text annotations that the SEM software sometimes overlays on the image (like labels or measurements).

```python
thresh = int(np.percentile(image, 99.5))
text_mask = (image >= thresh).astype(np.uint8) * 255
```
I find the pixel intensity at the 99.5th percentile — meaning only the brightest 0.5% of pixels are above this threshold. SEM text annotations are typically near-white and stand out from the gray SEM content.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
text_mask = cv2.dilate(text_mask, kernel, iterations=1)
```
I dilate the detected bright pixels by one iteration with a 3x3 rectangular kernel. This expands the text mask slightly to cover the edges of text characters that might not be quite as bright as the centers.

```python
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
```
I run connected component analysis to find groups of connected bright pixels. Each group gets a label, and I get statistics (area, bounding box) for each group. I use 8-connectivity (including diagonal neighbors).

```python
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if 5 < area < 2000:
        filtered_mask[labels == i] = 255
```
I keep only components between 5 and 2000 pixels. Text characters are small but not tiny — this range filters out single-pixel noise (< 5 px) and large bright regions that are actually part of the SEM content (> 2000 px), like bright dendrite surfaces.

```python
if np.sum(filtered_mask) == 0:
    return image
```
If no text was detected, I return the original image unchanged.

```python
cleaned = cv2.inpaint(image, filtered_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
```
I use OpenCV's `inpaint` function with the Telea algorithm (`INPAINT_TELEA`) to fill in the text regions. It replaces each text pixel by propagating nearby pixel values inward. The `inpaintRadius=5` means it looks 5 pixels around each masked pixel for reference values.

**Returns:** The image with text overlays replaced by inpainted background.

---

#### `clean_sem_image(image)`

Convenience function that runs both cleaning steps in sequence.

```python
cleaned = remove_scale_bar(image)
cleaned = remove_text_overlay(cleaned)
```
First removes the scale bar (bottom metadata), then removes any remaining text overlays. Order matters — removing the scale bar first means we don't accidentally detect the scale bar text as an overlay.

**Returns:** Fully cleaned image.

---

#### `create_overlay(image, mask, color=(0, 255, 0), alpha=0.4)`

Creates a semi-transparent colored overlay showing a mask on top of an image. I use this for visualization — it lets me see the segmentation result on top of the original image.

```python
if image.ndim == 2:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
else:
    base = image.copy()
```
If the input is grayscale (2D array), I convert it to 3-channel BGR so I can overlay color on it. If it's already color, I just copy it.

```python
overlay = base.copy()
mask_bool = mask > 0
overlay[mask_bool] = color
```
I create a copy of the base image and set all pixels where the mask is non-zero to the overlay color. Default is green `(0, 255, 0)` in BGR.

```python
result = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)
```
`addWeighted` blends two images: `result = alpha * overlay + (1-alpha) * base + 0`. With `alpha=0.4`, the colored mask is 40% visible and the original image is 60% visible, creating a transparent overlay effect.

**Returns:** A 3-channel color image with the mask overlay.

---

#### `create_comparison_strip(images, titles, height=400)`

Creates a horizontal strip of images with title bars for side-by-side visual comparison. I use this to generate the required "source -> classic mask -> YOLO mask -> skeleton" comparison figures.

```python
if img.ndim == 2:
    panel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
```
I convert each grayscale image to color so they can all be stacked together (you can't stack grayscale and color images).

```python
scale = height / h
new_w = int(w * scale)
panel = cv2.resize(panel, (new_w, height))
```
I resize each image to the target height while preserving the aspect ratio. The width is calculated proportionally.

```python
title_bar = np.zeros((40, new_w, 3), dtype=np.uint8)
cv2.putText(title_bar, title, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
```
I create a black 40-pixel-tall title bar and draw white text on it. `FONT_HERSHEY_SIMPLEX` is a clean font, `0.7` is the font scale, `(255,255,255)` is white in BGR, and `LINE_AA` enables anti-aliased text.

```python
panel = np.vstack([title_bar, panel])
```
I stack the title bar on top of the image vertically.

```python
max_h = max(p.shape[0] for p in panels)
```
I find the tallest panel (they might differ slightly due to rounding in aspect ratio calculations).

```python
if p.shape[0] < max_h:
    pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
    p = np.vstack([p, pad])
```
I pad shorter panels with black pixels at the bottom so they all have the same height — necessary for horizontal stacking.

```python
strip = np.hstack(padded)
```
Finally I stack all panels side-by-side horizontally into one wide image.

**Returns:** A single image with all panels side by side, each with a title.

---

#### `__main__` Self-Test Block

When I run `python utils.py` directly, it executes a synthetic self-test:

1. Creates a 512x512 random grayscale image simulating SEM noise
2. Adds a bright bottom strip (simulating a scale bar) and bright pixel groups (simulating text)
3. Tests `clean_sem_image` and verifies the bottom row intensity changed
4. Tests `create_overlay` with a square mask and checks the output shape
5. Tests `create_comparison_strip` with three images
6. Saves all outputs to the `output/` directory

This lets me verify all utilities work without needing real SEM images.

---

## 3. classic_pipeline.py

**Purpose:** The classic computer vision pipeline for dendrite segmentation — no neural networks. It follows the four-stage architecture required by the project: pre-processing, segmentation, post-processing, and branch separation, plus skeletonization.

### Constants

#### Stage A: Pre-processing

```python
CLAHE_CLIP_LIMIT = 3.0
```
The clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization). This limits how much contrast enhancement is applied per tile. Higher values give more contrast but also amplify noise. I chose 3.0 as a balance — enough to bring out dendrite detail without making noise unbearable.

```python
CLAHE_TILE_SIZE = 8
```
CLAHE divides the image into tiles and equalizes each one independently. An 8x8 grid means each tile covers roughly 1/64 of the image. Smaller tiles give more local adaptation (good for SEM images with uneven illumination) but can create visible tile boundaries.

```python
BILATERAL_D = 9
```
The diameter of the bilateral filter neighborhood — each pixel considers a 9x9 area around it. Larger means more smoothing.

```python
BILATERAL_SIGMA_COLOR = 75
```
How much difference in intensity is tolerated before the bilateral filter stops averaging. A value of 75 means pixels whose intensities differ by more than ~75 gray levels are treated as belonging to different regions (e.g., a dendrite edge). This is what makes bilateral filtering edge-preserving.

```python
BILATERAL_SIGMA_SPACE = 75
```
How much spatial distance matters. Pixels farther than ~75 pixels away have less influence. Together with `BILATERAL_D`, this controls the spatial extent of smoothing.

#### Stage B: Segmentation

```python
ADAPTIVE_BLOCK_SIZE = 51
```
The neighborhood size for adaptive thresholding — each pixel's threshold is computed from a 51x51 window around it. Must be odd. I chose 51 because dendrite branches in our SEM images are typically 5-30 pixels wide, so a 51-pixel window is large enough to capture local background variation without being thrown off by the dendrite itself.

```python
ADAPTIVE_C = 5
```
A constant subtracted from the computed local mean threshold. This shifts the threshold slightly, making the algorithm less sensitive — it requires a pixel to be 5 gray levels above its local mean to be classified as foreground. This helps reduce noise in flat background regions.

#### Stage C: Post-processing

```python
EROSION_KERNEL_SIZE = 5
```
The kernel size for the erosion step in morphological reconstruction. A 5x5 elliptical kernel is used to aggressively erode the mask.

```python
EROSION_ITERATIONS = 3
```
I erode 3 times — this removes thin noise strands while keeping the thick cores of real dendrite branches. The eroded result becomes the "marker" for reconstruction.

```python
CLOSING_KERNEL_SIZE = 5
```
The kernel size for morphological closing. A 5x5 elliptical kernel fills small gaps and holes in the dendrite mask.

```python
MIN_COMPONENT_AREA = 50
```
Connected components smaller than 50 pixels are removed as noise. Real dendrite structures are much larger than 50 pixels, so this safely eliminates speckle noise.

#### Stage D: Separation

```python
DISTANCE_THRESHOLD = 0.4
```
Fraction of the maximum distance transform value used to find "sure foreground" markers for watershed. At 0.4 (40%), only pixels deep inside a branch are marked as definite foreground. Lower values make watershed more aggressive in splitting touching branches.

### Functions

#### `normalize_histogram(image)`

Linear stretch of pixel intensities to fill the full [0, 255] range.

```python
min_val = float(image.min())
max_val = float(image.max())
if max_val == min_val:
    return np.zeros_like(image)
```
I find the minimum and maximum pixel values. If the image is completely flat (all same value), I return an all-black image to avoid division by zero.

```python
normalized = ((image.astype(np.float64) - min_val) / (max_val - min_val) * 255)
return normalized.astype(np.uint8)
```
The formula maps `min_val -> 0` and `max_val -> 255`, stretching everything in between linearly. I cast to float64 first to avoid integer overflow during arithmetic, then back to uint8 for the result.

**Why:** SEM images often use only a narrow range of the 0-255 scale (e.g., 40-180). Stretching to the full range ensures subsequent algorithms (CLAHE, thresholding) work with maximum dynamic range.

---

#### `apply_clahe(image)`

Applies Contrast Limited Adaptive Histogram Equalization.

```python
clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP_LIMIT,
    tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
)
return clahe.apply(image)
```
I create a CLAHE object with my clip limit and tile size, then apply it to the image. Unlike regular histogram equalization (which uses one global mapping), CLAHE divides the image into tiles and equalizes each independently. The clip limit prevents over-amplification of noise in flat regions — if any histogram bin exceeds the clip limit, the excess is redistributed evenly.

**Why:** SEM images have non-uniform illumination (charging effects, detector geometry). CLAHE enhances local contrast so dendrites stand out from the background everywhere, not just in well-lit regions.

---

#### `apply_bilateral_filter(image)`

Edge-preserving denoising.

```python
return cv2.bilateralFilter(
    image, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
)
```
The bilateral filter is like a Gaussian blur, but it weights pixels by both spatial distance AND intensity similarity. Pixels that are close spatially AND similar in intensity get averaged together. Pixels across an edge (different intensities) are kept separate.

**Why:** SEM images have shot noise (random pixel fluctuations). A regular Gaussian blur would smooth out noise but also blur dendrite edges. The bilateral filter smooths flat regions while keeping the sharp edges where dendrites meet the background.

---

#### `preprocess(image)`

Full pre-processing pipeline — runs all four steps in sequence.

```python
cleaned = clean_sem_image(image)
normalized = normalize_histogram(cleaned)
clahe_img = apply_clahe(normalized)
bilateral_img = apply_bilateral_filter(clahe_img)
```
The order matters:
1. **Clean** — remove scale bar and text first (they would confuse later steps)
2. **Normalize** — stretch to full range so CLAHE has full dynamic range to work with
3. **CLAHE** — enhance local contrast
4. **Bilateral** — denoise without blurring edges (done last so CLAHE has clean input)

```python
intermediates = {
    "01_original": image,
    "02_cleaned": cleaned,
    ...
}
```
I save every intermediate image in a dictionary so I can visualize the pipeline stages for the report. The numeric prefixes ensure they sort in pipeline order.

**Returns:** The final pre-processed image and a dictionary of all intermediates.

---

#### `segment_adaptive(image)`

Adaptive thresholding — the primary segmentation method.

```python
mask = cv2.adaptiveThreshold(
    image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C
)
```
For each pixel, OpenCV computes the Gaussian-weighted mean of the `ADAPTIVE_BLOCK_SIZE x ADAPTIVE_BLOCK_SIZE` neighborhood around it, subtracts `ADAPTIVE_C`, and uses that as the local threshold. If the pixel is brighter than this local threshold, it becomes 255 (white/foreground); otherwise 0 (black/background).

`ADAPTIVE_THRESH_GAUSSIAN_C` means the local mean is Gaussian-weighted (center pixels contribute more than edge pixels), which gives smoother thresholds than a simple mean.

**Why:** SEM images have non-uniform illumination. A single global threshold (like Otsu) would work well in bright regions but miss dendrites in darker areas. Adaptive thresholding adjusts to local brightness, so it finds dendrites everywhere in the image.

---

#### `segment_otsu(image)`

Otsu's binarization — a fallback segmentation method.

```python
_, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```
Otsu's method finds the single global threshold that minimizes the within-class variance — it looks at the histogram and finds the best split point between the two peaks (background and foreground). I pass `0` as the threshold because Otsu ignores it and computes its own. The `+` combines the binary and Otsu flags.

**Why:** Otsu works well when the image has a clear bimodal histogram (two distinct peaks). For uniformly illuminated SEM images, it can be simpler and faster than adaptive thresholding. I keep it as a fallback option.

---

#### `morphological_reconstruction(mask)`

Geodesic dilation-based reconstruction to remove noise while preserving dendrite structure.

```python
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (EROSION_KERNEL_SIZE, EROSION_KERNEL_SIZE)
)
marker = cv2.erode(mask, kernel, iterations=EROSION_ITERATIONS)
```
I create a 5x5 elliptical structuring element and erode the mask 3 times. This is aggressive — only the thick cores of real dendrites survive. Thin noise strands get completely eroded away. The result is my "marker" — seed points I know are definitely dendrite.

```python
marker_f = (marker / 255.0).astype(np.float64)
mask_f = (mask / 255.0).astype(np.float64)
```
Scikit-image's `reconstruction` function expects float images in [0, 1], so I convert from [0, 255] uint8.

```python
reconstructed_f = reconstruction(marker_f, mask_f, method='dilation')
```
Geodesic dilation grows the marker iteratively, but it can never exceed the original mask. Think of it like water filling a basin — the marker is the water source, and the mask is the basin walls. The water (marker) expands until it fills the entire basin (mask). This recovers the full dendrite shape from the eroded cores, while noise that was completely eroded away stays gone.

```python
reconstructed = (reconstructed_f * 255).astype(np.uint8)
```
I convert back to [0, 255] uint8 format.

**Why:** This is better than standard morphological opening (erode then dilate) because opening tends to shrink thin branches. Reconstruction grows the marker back to the original mask boundary, so thin branches that are connected to thick cores are fully recovered.

---

#### `apply_closing(mask)`

Morphological closing to fill small holes.

```python
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE)
)
return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```
Closing = dilation followed by erosion. Dilation fills small holes and gaps inside dendrites. The subsequent erosion restores the original boundary size. I use an elliptical kernel because dendrites are organic shapes (no sharp corners).

**Why:** After reconstruction, the mask might have small holes inside dendrite branches (from noise or uneven intensity). Closing fills these without changing the overall shape.

---

#### `remove_small_components(mask, min_area=None)`

Removes connected components smaller than a threshold.

```python
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    mask, connectivity=8
)
```
Connected component analysis finds all groups of connected white pixels. With 8-connectivity, pixels touching diagonally are considered connected. `stats` gives the area, bounding box, and centroid of each component.

```python
cleaned = np.zeros_like(mask)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        cleaned[labels == i] = 255
```
I start from label 1 (label 0 is the background). For each component, if its area (in pixels) is at least `min_area` (default 50), I keep it in the output. Otherwise it's discarded as noise.

**Why:** After morphological operations, there might still be small speckle noise. Real dendrites are large structures — removing anything under 50 pixels is safe and cleans up the final mask.

---

#### `postprocess(mask)`

Full post-processing pipeline.

```python
recon = morphological_reconstruction(mask)
closed = apply_closing(recon)
cleaned = remove_small_components(closed)
```
Three steps in sequence:
1. **Reconstruction** — remove noise while preserving connected dendrite structure
2. **Closing** — fill small holes inside dendrites
3. **Small component removal** — eliminate any remaining speckle

**Returns:** The cleaned mask and a dictionary of intermediates.

---

#### `separate_branches(mask)`

Separates touching dendrite branches using distance transform and watershed.

```python
if np.sum(mask) == 0:
    return mask.copy()
```
If the mask is empty (no foreground), I return immediately to avoid division by zero.

```python
dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
```
The distance transform computes, for each white pixel, its Euclidean distance to the nearest black pixel (nearest edge). Pixels deep inside a thick branch get high values; pixels near edges get low values. `cv2.DIST_L2` means Euclidean distance, and `5` is the mask size for the distance computation.

```python
_, sure_fg = cv2.threshold(
    dist, DISTANCE_THRESHOLD * dist.max(), 255, cv2.THRESH_BINARY
)
```
I threshold the distance map at 40% of its maximum value. Only pixels that are deep inside a branch (far from any edge) survive. These are "sure foreground" markers — seeds for watershed. Where two branches touch, the distance values dip (the touching point is close to the edge), so the two branches get separate markers.

```python
sure_bg = cv2.dilate(mask, kernel, iterations=3)
```
I dilate the original mask 3 times to get the "sure background" — a region that definitely includes all background pixels plus a margin around the dendrites.

```python
unknown = cv2.subtract(sure_bg, sure_fg)
```
The "unknown" region is everything between sure foreground and sure background. Watershed will decide whether these pixels belong to one branch or another.

```python
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
```
I label each sure-foreground seed region with a unique integer. I add 1 so the background label becomes 1 (not 0), because watershed uses 0 to mean "unknown/undecided." Then I set all unknown pixels to 0.

```python
mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
markers = cv2.watershed(mask_color, markers)
```
OpenCV's watershed requires a 3-channel image. It treats the markers as seeds and "floods" outward from each seed, stopping where different flood regions meet. The boundaries where different regions meet are marked as -1.

```python
separated = mask.copy()
separated[markers == -1] = 0
```
I set the watershed boundary pixels to black (0), which creates thin gaps between touching branches.

**Returns:** The binary mask with touching branches separated by thin boundary lines.

---

#### `skeletonize_mask(mask)`

Extracts a single-pixel-width centerline skeleton.

```python
binary = (mask > 0).astype(bool)
skel = skeletonize(binary)
return (skel.astype(np.uint8) * 255)
```
I convert the mask to a boolean array (True/False), run scikit-image's `skeletonize` (Zhang-Suen thinning algorithm), and convert back to uint8 (0/255). Thinning repeatedly peels away the outermost layer of pixels until only single-pixel-wide lines remain — the skeleton represents the topological centerline of each branch.

**Why:** The project requires skeleton extraction. Skeletons are useful for measuring dendrite branch length and analyzing branching patterns.

---

#### `run_classic_pipeline(image_path, output_dir=None, save_intermediates=True)`

Runs the full 4-stage pipeline on a single image.

```python
image = load_image(image_path, grayscale=True)
basename = os.path.splitext(os.path.basename(image_path))[0]
```
I load the image in grayscale and extract the filename without extension (e.g., `"dendrite_001"` from `"dendrite_001.png"`) for use in output filenames.

The function then calls each stage in order:
1. `preprocess(image)` — cleaning, normalization, CLAHE, bilateral
2. `segment_adaptive(preprocessed)` — adaptive thresholding
3. `postprocess(seg_mask)` — reconstruction, closing, component removal
4. `separate_branches(clean_mask)` — distance transform + watershed
5. `skeletonize_mask(separated)` — Zhang-Suen thinning

```python
if output_dir and save_intermediates:
    img_out_dir = os.path.join(output_dir, basename)
    for name, img in all_intermediates.items():
        save_image(img, os.path.join(img_out_dir, f"{name}.png"))
```
If saving intermediates is enabled, I create a subdirectory named after the image and save every intermediate stage as a separate PNG. This is essential for the report — I can show exactly what each pipeline stage does.

**Returns:** A dictionary containing the final mask, skeleton, separated mask, and all intermediates.

---

#### `process_all_images(input_dir, output_dir)`

Batch-processes all SEM images in a directory.

```python
image_paths = list_images(input_dir)
for path in image_paths:
    results = run_classic_pipeline(path, output_dir, save_intermediates=True)
    all_results[basename] = results
```
I list all images, then loop through each one and run the full pipeline. Results are stored in a dictionary keyed by image name.

**Returns:** A dictionary mapping image basenames to their pipeline results.

---

#### `main()`

The CLI entry point using argparse. Supports two modes:
- **Single image mode:** `python classic_pipeline.py path/to/image.png`
- **Batch mode:** `python classic_pipeline.py --input path/to/dir/`

The `--output` flag sets the output directory, and `--no-intermediates` skips saving intermediate stages (only saves the final mask and skeleton).

---

#### `__main__` Self-Test Block

When run without CLI arguments, it creates a synthetic 512x512 SEM-like image:
- Random gray noise (30-80 intensity) as background
- Six bright lines drawn with `cv2.line` simulating dendrite branches
- A bright strip at the bottom simulating a scale bar

Then it runs the full pipeline on this synthetic image and prints the number of non-zero pixels in the mask and skeleton, plus the shape and value range of every intermediate.

---

## 4. yolo_pipeline.py

**Purpose:** The deep learning approach — uses YOLOv11 instance segmentation to detect and segment dendrites. Handles dataset validation, model training with transfer learning, and inference (single image and batch).

### Constants

```python
DEFAULT_MODEL = "yolo11n-seg.pt"
```
The pretrained YOLO model to start from. `yolo11n-seg` is the "nano" variant — the smallest and fastest YOLOv11 segmentation model. I chose nano because my SEM dataset is small, so a larger model would overfit. The `-seg` suffix means it's the segmentation variant (not just detection).

```python
DEFAULT_EPOCHS = 100
```
Maximum training epochs. 100 is a reasonable upper bound — early stopping will likely stop training before this.

```python
DEFAULT_IMGSZ = 640
```
Input image size. YOLO resizes all images to 640x640 during training and inference. This is the standard YOLO input size.

```python
DEFAULT_BATCH = 8
```
Batch size — how many images are processed together in one forward/backward pass. 8 is conservative enough to fit in most GPU memory.

```python
DEFAULT_PATIENCE = 20
```
Early stopping patience — if the validation loss doesn't improve for 20 consecutive epochs, training stops. This prevents overfitting and saves time.

```python
DEFAULT_FREEZE = 10
```
Number of backbone layers to freeze during training. Freezing the first 10 layers means they keep their pretrained COCO weights and only the later layers adapt to SEM data. This is standard transfer learning — the early layers learn generic features (edges, textures) that are useful for SEM images too, while the later layers learn domain-specific features.

```python
DEFAULT_LR0 = 0.001
```
Initial learning rate. 0.001 is a conservative starting point for fine-tuning — lower than training from scratch because I don't want to destroy the pretrained weights.

```python
DEFAULT_CONF = 0.25
```
Confidence threshold for inference. YOLO only keeps detections with confidence above 0.25 (25%). Lower means more detections (potentially noisy), higher means fewer (potentially missing dendrites).

### Functions

#### `prepare_yolo_dataset(roboflow_dir, output_yaml=None)`

Validates a Roboflow-exported YOLO segmentation dataset and creates/verifies the `data.yaml` config file.

```python
required = ["train/images", "train/labels", "valid/images", "valid/labels"]
for subdir in required:
    full_path = os.path.join(roboflow_dir, subdir)
    if not os.path.isdir(full_path):
        raise FileNotFoundError(...)
```
I check that the four required subdirectories exist. Roboflow exports in this exact structure when you select "YOLOv8 Segmentation" format.

```python
train_imgs = len(list_images(os.path.join(roboflow_dir, "train/images")))
train_labels = len([f for f in os.listdir(...) if f.endswith('.txt')])
```
I count images and label files separately. Each label file (`.txt`) contains polygon coordinates for the dendrite instances in the corresponding image. A mismatch means some images are missing labels or vice versa.

```python
if train_imgs != train_labels:
    print(f"  WARNING: Image/label count mismatch...")
```
This is a warning, not an error — YOLO can handle missing labels (it treats unlabeled images as negatives).

```python
yaml_content = (
    f"path: {os.path.abspath(roboflow_dir)}\n"
    f"train: train/images\n"
    f"val: valid/images\n"
    f"\n"
    f"nc: 1\n"
    f"names:\n"
    f"  0: dendrite\n"
)
```
If no `data.yaml` exists, I create a minimal one. `nc: 1` means one class (dendrite). The `path`, `train`, and `val` fields tell YOLO where to find the data.

**Returns:** The path to the validated `data.yaml`.

---

#### `train_yolo(dataset_yaml, ...)`

Trains the YOLO segmentation model with transfer learning.

```python
from ultralytics import YOLO
```
I import ultralytics inside the function, not at the top of the file. This way, the rest of the module works even if ultralytics is not installed (useful for the self-test which doesn't need it).

```python
yolo_model = YOLO(model)
```
This loads the pretrained model. If `model` is `"yolo11n-seg.pt"`, ultralytics automatically downloads the pretrained COCO weights from the internet.

```python
results = yolo_model.train(
    data=dataset_yaml,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    patience=patience,
    freeze=freeze,
    lr0=lr0,
    project=project,
    name="dendrite_seg",
    exist_ok=True,
    verbose=True,
)
```
The `train()` call handles everything: data loading, augmentation, forward passes, loss computation, backpropagation, validation, learning rate scheduling, early stopping, and weight saving. Key parameters:
- `freeze=freeze` — freezes the first N backbone layers (transfer learning)
- `patience=patience` — early stopping
- `name="dendrite_seg"` — the run name for organizing outputs
- `exist_ok=True` — allows overwriting previous runs with the same name

**Returns:** The ultralytics Results object, which includes the path to the best model weights.

---

#### `predict_single(model_path, image_path, conf=DEFAULT_CONF)`

Runs inference on a single image and extracts the binary mask.

```python
model = YOLO(model_path)
results = model.predict(image_path, conf=conf, verbose=False)
```
I load the trained model and run prediction. `verbose=False` suppresses the per-image log output.

```python
image = cv2.imread(image_path)
h, w = image.shape[:2]
combined_mask = np.zeros((h, w), dtype=np.uint8)
```
I read the original image to get its dimensions, then create an empty mask of the same size.

```python
if results and results[0].masks is not None:
    masks_data = results[0].masks.data.cpu().numpy()
    for instance_mask in masks_data:
        resized = cv2.resize(instance_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        combined_mask[resized > 0.5] = 255
```
YOLO returns one mask per detected instance (each individual dendrite). `results[0].masks.data` is a PyTorch tensor of shape `(N, mH, mW)` where N is the number of instances. I move it to CPU and convert to numpy. For each instance mask, I resize it to the original image size (YOLO internally uses a smaller resolution) using bilinear interpolation, then threshold at 0.5 and OR it into the combined mask. The result is a single binary mask showing all dendrites.

**Returns:** A binary mask (0/255) the same size as the input image.

---

#### `predict_batch(model_path, input_dir, output_dir, conf=DEFAULT_CONF)`

Runs inference on all images in a directory. Same logic as `predict_single` but in a loop, and it loads the model only once for efficiency.

```python
model = YOLO(model_path)
```
I load the model once outside the loop — loading is expensive (GPU initialization, weight transfer).

For each image, it runs prediction, combines instance masks, saves the result, and prints the foreground pixel count.

**Returns:** A dictionary mapping image basenames to their binary masks.

---

#### `yolo_mask_to_skeleton(mask)`

Extracts a skeleton from a YOLO-generated mask. Same Zhang-Suen thinning as in `classic_pipeline.py`.

```python
binary = (mask > 0).astype(bool)
skel = skeletonize(binary)
return (skel.astype(np.uint8) * 255)
```

---

#### `main()`

CLI entry point with two subcommands:
- `python yolo_pipeline.py train --data data.yaml` — trains the model
- `python yolo_pipeline.py predict --model best.pt --source image.png` — runs inference

The predict command handles both single images and directories.

---

#### `__main__` Self-Test Block

When run without arguments, it tests everything that doesn't require ultralytics:

1. Creates a fake Roboflow directory structure with dummy images and YOLO-format label files
2. Runs `prepare_yolo_dataset` to validate the structure and create `data.yaml`
3. Tests `yolo_mask_to_skeleton` on a synthetic mask with a cross pattern
4. Cleans up the temporary test directory with `shutil.rmtree`

---

## 5. evaluate.py

**Purpose:** Computes pixel-level evaluation metrics between predicted masks and ground truth, generates comparison figures, and produces a metrics summary report with failure analysis.

### Functions

#### `compute_dice(pred, gt)`

Computes the Dice similarity coefficient (also called F1 score for pixels).

```python
pred_bin = (pred > 0).astype(np.float64)
gt_bin = (gt > 0).astype(np.float64)
```
I convert both masks from [0, 255] to [0.0, 1.0] float arrays. Any non-zero pixel becomes 1.0.

```python
if pred_sum + gt_sum == 0:
    return 1.0
```
If both masks are completely empty (no dendrites in either), I define Dice as 1.0 — perfect agreement on "nothing here."

```python
intersection = (pred_bin * gt_bin).sum()
return 2.0 * intersection / (pred_sum + gt_sum)
```
Element-wise multiplication of two binary arrays gives 1 only where both are 1 (the intersection). The Dice formula is `2 * intersection / (sum_pred + sum_gt)`. The factor of 2 normalizes it to [0, 1] — if the masks are identical, `intersection = pred_sum = gt_sum`, so `Dice = 2 * S / (S + S) = 1.0`.

**Why Dice:** It's the standard metric for segmentation evaluation. It penalizes both false positives and false negatives, and handles imbalanced classes (where background dominates) better than simple accuracy.

---

#### `compute_iou(pred, gt)`

Computes Intersection over Union (Jaccard index).

```python
intersection = (pred_bin * gt_bin).sum()
union = pred_bin.sum() + gt_bin.sum() - intersection
```
Union = all pixels that are positive in either mask. I compute it as `A + B - intersection` to avoid double-counting the overlap.

```python
return intersection / union
```
IoU ranges from 0 (no overlap) to 1 (perfect overlap). It's more strict than Dice — for the same overlap, IoU is always lower than or equal to Dice.

---

#### `compute_precision_recall(pred, gt)`

Pixel-level precision and recall.

```python
tp = float(np.logical_and(pred_bin, gt_bin).sum())
fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
fn = float(np.logical_and(~pred_bin, gt_bin).sum())
```
- **TP (True Positive):** pixels that are positive in both prediction and ground truth — correctly detected dendrite pixels
- **FP (False Positive):** pixels positive in prediction but not in ground truth — noise or background falsely marked as dendrite
- **FN (False Negative):** pixels positive in ground truth but not in prediction — dendrite pixels that were missed

```python
precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
```
- **Precision:** of all pixels I called "dendrite", what fraction actually are? High precision = low noise.
- **Recall:** of all actual dendrite pixels, what fraction did I find? High recall = complete coverage.

The edge cases return 1.0 — if there are no positive predictions (tp + fp = 0), precision is undefined and I default to 1.0. Same logic for recall.

---

#### `evaluate_single(pred, gt)`

Convenience function that computes all four metrics at once.

```python
dice = compute_dice(pred, gt)
iou = compute_iou(pred, gt)
precision, recall = compute_precision_recall(pred, gt)
return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}
```

---

#### `create_comparison_figure(source_path, classic_mask, yolo_mask, skeleton, output_path)`

Generates the required 4-panel comparison figure.

```python
source = load_image(source_path, grayscale=True)
skeleton_overlay = create_overlay(source, skeleton, color=(0, 0, 255), alpha=0.6)
```
I load the source image and create a red overlay of the skeleton on it. Red `(0, 0, 255)` in BGR is used because it stands out well against grayscale SEM images. Alpha=0.6 makes the skeleton more prominent.

```python
images = [source, classic_mask, yolo_mask, skeleton_overlay]
titles = ["Source", "Classic Mask", "YOLO Mask", "Skeleton"]
strip = create_comparison_strip(images, titles)
save_image(strip, output_path)
```
I use the comparison strip utility from `utils.py` to arrange all four images side by side with titles.

---

#### `generate_metrics_summary(results, output_path)`

Formats a text report with all metrics and saves it.

```python
header = f"{'Image':<25} {'Method':<10} {'Dice':>6} {'IoU':>6} {'Prec':>6} {'Rec':>6}"
```
I format the header with fixed-width columns using Python string formatting. `<25` means left-aligned in 25 characters, `>6` means right-aligned in 6 characters.

```python
classic_totals = {"dice": [], "iou": [], "precision": [], "recall": []}
```
I collect metrics for each method into lists so I can compute averages at the end.

```python
for name in sorted(results.keys()):
    entry = results[name]
    if "classic" in entry:
        m = entry["classic"]
        lines.append(f"{name:<25} {'Classic':<10} {m['dice']:>6.3f} ...")
```
For each image, I print the metrics for both Classic and YOLO (if available). `:.3f` formats floats to 3 decimal places.

```python
failures = analyze_failures(results)
failure_report = format_failure_report(failures)
lines.append(failure_report)
```
The summary includes the failure analysis report at the end.

The function prints the summary to the console and saves it to a text file.

---

#### `analyze_failures(results, threshold=0.5)`

Identifies segmentation failures and characterizes their root cause.

```python
prec_threshold = 0.6
rec_threshold = 0.6
```
I use 0.6 as thresholds for categorizing precision and recall as "low" or "high."

```python
if m["dice"] >= threshold:
    continue
```
I only analyze images where Dice is below 0.5 (50%) — everything above that is considered acceptable.

The root cause rules:
```python
if prec < prec_threshold and rec >= rec_threshold:
    cause = "Over-segmentation — noise included as dendrite"
```
Low precision (many false positives) + high recall (found most dendrites) means the algorithm is finding the dendrites but also marking noise as dendrite.

```python
elif prec >= prec_threshold and rec < rec_threshold:
    cause = "Under-segmentation — thin branches missed"
```
High precision (what it found is correct) + low recall (missed many pixels) means it's being too conservative — probably missing thin branches or low-contrast areas.

```python
elif prec < prec_threshold and rec < rec_threshold:
    cause = "Fundamental mismatch — wrong region or severe artifacts"
```
Both low means the prediction is completely off — possibly segmenting artifacts instead of dendrites.

```python
other_dice = entry.get(other_key, {}).get("dice")
if other_dice is not None and other_dice >= threshold:
    if method_key == "classic":
        cause += " (YOLO succeeds → likely non-uniform illumination)"
```
Cross-method insight: if the classic pipeline fails but YOLO succeeds on the same image, it suggests the classic pipeline's threshold-based approach is struggling with non-uniform illumination (which YOLO handles via learned features). If YOLO fails but classic succeeds, the image might be out-of-distribution for YOLO's training data.

**Returns:** A list of failure dictionaries, each with the image name, method, metrics, and diagnosed cause.

---

#### `format_failure_report(failures)`

Formats the failure list into readable text.

```python
if not failures:
    lines.append("  No failures detected — all images above threshold.")
else:
    for f in failures:
        lines.append(f"  Image: {f['name']}")
        lines.append(f"    Cause:     {f['cause']}")
```

---

#### `evaluate_all(classic_dir, yolo_dir, gt_dir, image_dir, output_dir)`

Batch evaluation — the main entry point for evaluation.

```python
gt_paths = list_images(gt_dir)
```
I start by listing all ground truth masks. Each ground truth file is named `<name>.png`, and I look for corresponding prediction files.

```python
classic_path = os.path.join(classic_dir, f"{name}_mask.png")
if os.path.exists(classic_path):
    classic_mask = load_image(classic_path, grayscale=True)
    entry["classic"] = evaluate_single(classic_mask, gt_mask)
```
For each ground truth, I look for matching classic and YOLO masks by name convention (`<name>_mask.png`). If found, I compute all metrics.

```python
if source_path and classic_mask is not None and yolo_mask is not None:
    from skimage.morphology import skeletonize
    skel = skeletonize((classic_mask > 0).astype(bool))
    skeleton = (skel.astype(np.uint8) * 255)
    create_comparison_figure(source_path, classic_mask, yolo_mask, skeleton, fig_path)
```
If both masks and the source image exist, I generate a comparison figure. I compute the skeleton from the classic mask for the figure.

```python
if results:
    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    generate_metrics_summary(results, summary_path)
```
At the end, I generate the full metrics summary with failure analysis.

---

#### `__main__` Self-Test Block

Seven tests that verify all evaluation functions:

1. **Perfect overlap** — two identical masks should give Dice=1.0 and IoU=1.0
2. **Partial overlap** — mathematically computed expected values (Dice=0.667, IoU=0.5) verified against actual output
3. **No overlap** — completely disjoint masks should give Dice=0.0 and IoU=0.0
4. **Both empty** — two empty masks should give Dice=1.0 and IoU=1.0 (agree on "nothing")
5. **Comparison figure** — generates a synthetic 4-panel figure and saves it
6. **Metrics summary** — generates a summary report from hardcoded example metrics
7. **Failure analysis** — tests with known failure patterns (over-segmentation, under-segmentation, fundamental mismatch) and asserts the correct root causes are identified

---

## 6. run_all.py

**Purpose:** End-to-end orchestrator that runs both pipelines on the same images and produces all required deliverables: masks, skeletons, comparison figures, and the metrics summary with failure analysis.

### Functions

#### `run_orchestrator(images_dir, gt_dir=None, yolo_model=None, output_dir=None)`

The main orchestrator function that runs the complete pipeline in 5 stages.

```python
project_dir = os.path.dirname(os.path.abspath(__file__))
if output_dir is None:
    output_dir = os.path.join(project_dir, "output")
```
I default the output directory to `output/` inside the project folder.

```python
classic_dir = os.path.join(output_dir, "classic")
yolo_dir = os.path.join(output_dir, "yolo")
compare_dir = os.path.join(output_dir, "comparisons")
eval_dir = os.path.join(output_dir, "evaluation")
```
I set up four subdirectories to organize the output cleanly.

**Stage 1 — Classic Pipeline:**
```python
classic_results = process_all_images(images_dir, classic_dir)
for name, res in classic_results.items():
    mask_path = os.path.join(classic_dir, f"{name}_mask.png")
    save_image(res["mask"], mask_path)
```
I run the classic pipeline on all images, then save top-level `<name>_mask.png` files in the classic directory. These are needed by `evaluate_all()` which looks for files matching this naming convention.

**Stage 2 — YOLO Pipeline (optional):**
```python
if yolo_model:
    if not os.path.isfile(yolo_model):
        print(f"  WARNING: YOLO model not found: {yolo_model} — skipping")
    else:
        from yolo_pipeline import predict_batch
        yolo_results = predict_batch(yolo_model, images_dir, yolo_dir)
```
YOLO is optional — you need to provide a trained `.pt` weights file. If the file doesn't exist, I print a warning instead of crashing. I import `predict_batch` inside the `if` block so the code works even without ultralytics installed (as long as you don't try to use YOLO).

**Stage 3 — Skeletonization:**
```python
for name, res in classic_results.items():
    skeletons[name] = res.get("skeleton", skeletonize_mask(res["mask"]))
```
I extract skeletons from the classic pipeline results. `res.get("skeleton", ...)` tries to use the skeleton already computed by the classic pipeline; if it's not there (shouldn't happen), it computes it from the mask as a fallback.

**Stage 4 — Comparison Figures:**
```python
for img_path in image_paths:
    name = os.path.splitext(os.path.basename(img_path))[0]
    classic_mask = classic_results[name]["mask"]
    if name in yolo_results:
        yolo_mask = yolo_results[name]
    else:
        yolo_mask = np.zeros_like(classic_mask)
    create_comparison_figure(img_path, classic_mask, yolo_mask, skeleton, fig_path)
```
For each image, I create the 4-panel comparison figure. If YOLO wasn't run, I use a blank (all-black) mask as a placeholder so the figure still generates.

```python
if isinstance(yolo_mask, dict):
    yolo_mask = yolo_mask.get("mask", np.zeros_like(classic_mask))
```
Safety check — in case YOLO results are stored as a dict instead of a plain mask array.

**Stage 5 — Evaluation (optional):**
```python
if gt_dir and os.path.isdir(gt_dir):
    eval_results = evaluate_all(
        classic_dir=classic_dir, yolo_dir=yolo_dir,
        gt_dir=gt_dir, image_dir=images_dir, output_dir=eval_dir,
    )
```
If ground truth masks are provided, I run the full evaluation. This computes metrics, generates comparison figures in the evaluation directory, and writes the summary report.

At the end, the function prints a summary showing how many images were processed, where outputs are saved, and the average Dice scores for each method.

**Returns:** A summary dictionary with counts, paths, and evaluation results.

---

#### `main()`

CLI entry point using argparse:
- `--images` (required) — directory with source SEM images
- `--gt` (optional) — directory with ground truth masks
- `--yolo-model` (optional) — path to trained YOLO weights
- `--output` (optional) — output directory

Example usage:
```bash
python run_all.py --images data/images --gt data/ground_truth --yolo-model weights/best.pt
```

---

#### `__main__` Self-Test Block

When run without arguments, it performs a full end-to-end test:

1. Creates 3 synthetic SEM images with different dendrite patterns (vertical, diagonal, star)
2. Creates matching ground truth masks (slightly thicker lines to simulate manual annotation)
3. Runs the orchestrator with classic pipeline only (no YOLO model)
4. Verifies:
   - All 3 images were processed
   - All 3 comparison figures were created
   - The metrics summary file exists and contains the "Failure Analysis" section
   - All 3 images have Classic evaluation results
5. Cleans up the temporary test directory

This test proves the entire system works end-to-end without needing real SEM images or trained YOLO weights.
