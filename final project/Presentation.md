# SEM Dendrite Segmentation -- Presentation and Algorithm Explanation

Student ID: 323073734

---

## 1. Problem Statement (30 sec)

### What Are Dendrites?

Lithium dendrites are fractal, branch-like metallic microstructures that grow on battery anodes during charging cycles. They are a primary cause of short circuits in lithium-ion batteries and a major safety concern.

### Why Segment Them?

Quantifying dendrite morphology (length, branching, coverage) from SEM images helps researchers understand growth mechanisms and evaluate protective coatings. Manual annotation is slow and subjective -- automatic segmentation enables reproducible, large-scale analysis.

### SEM Image Challenges

- **Low signal-to-noise ratio** -- SEM shot noise creates grainy backgrounds
- **Non-uniform illumination** -- brightness varies across the field of view
- **Saturation artifacts** -- overly bright regions near the substrate electrode
- **Charging effects** -- blur caused by electrostatic charge buildup
- **Soft gradients** -- dendrite edges blend into background rather than being sharp
- **Scale bar and text overlays** -- must be removed before processing

---

## 2. Classic Pipeline Overview (1 min)

The pipeline processes each SEM image through four sequential stages:

```
Original Image
    |
    v
[A] PRE-PROCESSING
    Clean (remove scale bar/text)
    -> Normalize histogram
    -> CLAHE (local contrast)
    -> Bilateral filter (denoise, preserve edges)
    -> Gaussian smooth
    |
    v
[B] SEGMENTATION
    Adaptive threshold (local, for non-uniform illumination)
    vs. Otsu threshold (global, fallback)
    -> Auto-select better mask via quality heuristics
    -> Auto-correct polarity (dendrites = minority foreground)
    |
    v
[C] POST-PROCESSING
    Morphological opening (kill pixel noise)
    -> Morphological reconstruction (geodesic dilation)
    -> Closing (fill small holes)
    -> Remove small components (< 0.01% image area)
    -> Remove substrate band (2-stage: FG scan + intensity profile)
    -> Remove top band (dense header patterns)
    -> Remove edge noise (margin-hugging artifacts)
    -> Remove horizontal artifacts (interface lines)
    |
    v
[D] SEPARATION + SKELETON
    Distance transform + Watershed (separate touching branches)
    -> Zhang-Suen thinning (single-pixel skeleton)
    -> Prune tiny fragments
    -> Remove short spurs
    |
    v
Final: Binary Mask + Skeleton
```

---

## 3. Stage-by-Stage Algorithm Explanation (2 min)

### Stage A: Pre-processing

**Why CLAHE instead of global histogram equalization?**

Global histogram equalization redistributes all pixel intensities uniformly -- this amplifies noise in dark regions and can wash out detail in bright regions. CLAHE (Contrast Limited Adaptive Histogram Equalization) divides the image into tiles (8x8 grid) and equalizes each tile independently. The "clip limit" (set to 3.0) prevents noise amplification by capping how much any intensity bin can grow.

**Why bilateral filter instead of Gaussian blur alone?**

Gaussian blur reduces noise but also blurs dendrite edges. The bilateral filter is edge-preserving: it averages pixels only with neighbors that are both spatially close AND similar in intensity. This smooths the noisy background while keeping sharp dendrite boundaries intact. We apply 2 passes for stronger denoising.

**Pipeline order matters:** Normalize first (establish consistent intensity range), then CLAHE (enhance local contrast), then bilateral (denoise while preserving the enhanced edges).

### Stage B: Segmentation

**Why adaptive threshold, not Otsu alone?**

Otsu's method finds a single global threshold that minimizes within-class variance. This works for images with uniform illumination, but SEM images often have brightness gradients -- a threshold that works for the left side misses dendrites on the right side.

Adaptive thresholding computes a local threshold for each pixel based on the mean of its 51x51 neighborhood, minus a constant C=5. This adapts to local brightness variations automatically.

**Automatic method selection:** We generate both masks and pick the better one using quality heuristics:
- Foreground ratio should be between 2-55% (dendrites are the minority)
- Noise ratio (tiny components / substantial components) should be below 40%
- If adaptive passes these checks, we use it; otherwise Otsu is the fallback

**Polarity correction:** Since adaptive threshold can produce inverted masks depending on local contrast, we auto-detect if foreground exceeds 50% and invert if necessary.

### Stage C: Post-processing

**Why morphological reconstruction instead of opening?**

Standard morphological opening (erode then dilate) removes noise but also erodes thin dendrite branches, potentially breaking them. Morphological reconstruction uses a different approach:
1. Aggressively erode the mask to create a "marker" -- only the thick branch cores survive
2. Use the original mask as the "limit"
3. Geodesically dilate the marker: it grows back within the original mask boundaries

The result: noise that was disconnected from thick branches is eliminated, but thin branches connected to thick cores are perfectly preserved.

**Substrate detection -- the "ground of the trees" problem:**

Dendrites grow upward from a bright electrode substrate at the bottom of the image. This substrate region gets falsely segmented as foreground. We detect and remove it with a two-stage approach:

- *Stage 1 (Smoothed FG scan):* Compute per-row foreground ratio, smooth with a 20-row moving average, scan from bottom upward. If a contiguous run of rows has smoothed FG >= 50%, that is the substrate. The smoothing catches grain-boundary regions where raw per-row FG oscillates between 30-70%.

- *Stage 2 (Intensity profile):* If Stage 1 finds nothing, analyze the preprocessed image's row-mean intensity. Compare the dark upper region (top 20%) to the bright lower region (60-88%). If contrast ratio >= 25%, find the transition row where brightness rises and zero everything below it.

**Top band removal:** Some Hard images have bright nanopore patterns in the top rows. We scan from row 0 down, find contiguous dense rows (FG >= 30%), and remove them (capped at 15% of image height).

**Edge noise removal:** Small components (< 2000px) with centroids within 5% of left/right edges are removed -- these are typically SEM annotation artifacts.

### Stage D: Separation and Skeletonization

**Why distance transform + watershed?**

When two dendrite branches touch, they appear as a single connected component. The distance transform computes, for each foreground pixel, the distance to the nearest background pixel. Branch cores (far from edges) have high distance values; the touching point between branches has a low value (it is a narrow bridge).

We threshold at 40% of the maximum distance to find "sure foreground" markers (one per branch core), then use watershed segmentation to find the boundary between them. This splits touching branches at their narrowest connection point.

**Skeletonization (Zhang-Suen thinning):** Iteratively peels away boundary pixels while preserving topology, producing a single-pixel-wide centerline. We apply a small morphological closing before thinning to smooth jagged mask edges (prevents diamond/loop artifacts).

**Pruning:** Remove tiny skeleton fragments (< 15 pixels) and horizontal line remnants. Then iteratively remove short spurs (terminal branches <= 10 pixels) -- these are typically noise-induced stubs, not real dendrite branches.

---

## 4. Results (1 min)

### Easy Images (10 Ag_2e-9 images)

The classic pipeline performs well on Easy images:
- Masks capture dendrite structures accurately
- Skeletons trace centerlines correctly, following branch topology
- Substrate region at the bottom is detected and removed cleanly
- Few false positives after post-processing

### Hard Images -- Group B (7 Ag_40nm_pitch images)

These images feature a bright substrate with grain boundary patterns:
- **Upper half (dendrite region):** Segmented well -- dendrites are properly captured
- **Lower half (substrate):** The two-stage substrate detection removes the bright electrode region, eliminating 70-90% of false positives in images 001, 003, 004, 005, 006
- Images 002 and 007 have ambiguous contrast, making substrate detection less effective

### Hard Images -- Group A (3 70nm_pitch images)

- **Image 035:** Partially salvageable -- top band removal removes the nanopore pattern header, and the dendrite (brighter than background) is captured
- **Images 036, surface_036:** Genuine failure cases (see Failure Analysis below)

---

## 5. Failure Analysis (30 sec)

### Where the Classic Pipeline Fails

**Periodic nanopore/nanowire backgrounds (70nm_pitch_036, surface_036):**
The background has a regular pattern of bright dots/lines at the same spatial frequency as dendrite features. Adaptive thresholding cannot distinguish between "bright dendrite on dark background" and "bright pattern element on dark gap" because both have the same local contrast profile. This is a fundamental limitation of pixel-level thresholding -- it has no concept of object shape or semantic meaning.

**Low-contrast substrates (Ag_40nm_pitch_002, _007):**
When the intensity difference between the dendrite region and the substrate is small, neither the smoothed FG scan nor the intensity profile detects a clear transition. The substrate region remains partially segmented.

### Why These Cases Are Hard

The root cause is the same: classic CV methods operate on local pixel statistics (intensity, gradient, neighborhood mean). They cannot learn that "dendrites are tree-shaped structures" or "this regular pattern is background, not an object." These are semantic distinctions that require learned features.

### How YOLO Handles These Cases Differently

A trained YOLO model learns shape and texture features from labeled examples. It can distinguish a dendrite branch from a nanopore pattern because it has seen both during training. The deep learning approach handles:
- Semantic understanding of "dendrite vs. background pattern"
- Robustness to varying illumination (learned invariance)
- Instance-level segmentation (each dendrite is a separate object)

This is the fundamental trade-off: classic methods are interpretable and need no training data, but fail on semantically ambiguous images. Deep learning methods require labeled data and are less interpretable, but can learn the distinction.
