# Final Project: Morphological Segmentation of SEM Dendrites

## Scientific Background

Dendrites are microscopic metallic structures with fractal, branch-like geometry that grow on the anode during cyclic charging of lithium batteries. They form when lithium ions accumulate unevenly on the surface under extreme conditions (high current density or low temperature), creating needle-like crystals growing perpendicular to the surface. A dendrite that grows too large can puncture the separator membrane, cause an internal short circuit, and lead to thermal runaway.

**Project goal:** Develop an automatic segmentation system for dendrites from SEM (Scanning Electron Microscope) images for failure prediction and prevention.

## Task Definition: Segmentation in a Noisy Environment

Pixel-wise semantic segmentation (Pixel-wise classification) separating dendrite structure from background.

**SEM image challenges:**
- Low signal-to-noise ratio (shot noise)
- Saturation artifacts ("burned" regions)
- Charging Effect blur
- Soft gradients at boundaries
- Treat intensity as topographic height map — do not rely on global thresholds alone

**Required outputs:**
1. Binary Mask separating dendrite from background
2. Pre-processing and cleaning: remove foreign elements (technical text, scale bars) and sensor noise
3. Skeletonization: single-pixel-width centerline extraction for geometric computations

## Methodology: Two Approaches (Must Implement and Compare Both)

### Approach A: Deep Learning (YOLO-Seg)

- Use SOTA models: YOLOv8/v11
- Transfer Learning for non-linear pattern recognition
- High data dependency: requires accurately labeled supervised dataset
- High robustness to complex texture changes
- **Labeling method:** must use Polygons only
- **Recommended labeling tools:**
  - Roboflow (auto-labeling capabilities, native YOLO Segmentation export)
  - CVAT (Computer Vision Annotation Tool) for on-premise projects

### Approach B: Classic Image Processing

- Deterministic pipeline based on mathematical morphology
- Object isolation based on geometric and statistical regularities
- Low data dependency: can be applied on minimal samples without training
- Sensitive to lighting changes; requires parameter fine-tuning
- Allowed libraries: cv2 and scikit-image

## Classic Pipeline Implementation Details

> **Note:** The detailed steps below combine the required pipeline (section 5 of the assignment) with the expanded appendix suggestion. The appendix is explicitly marked as "just a suggestion" — any other classic approach is valid.

### Stage A: Pre-processing

1. **Histogram Normalization:** Linear stretch of pixel values to full range. Ensures a common basis across all images so threshold values remain stable regardless of exposure differences.

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Divides image into tiles, computes separate histogram per region with clipping to prevent background noise from becoming "objects." Dramatically improves detection of thin and delicate branches, even in shadowed electrode regions.

3. **Bilateral Filter (Edge-Preserving Denoising):** Smooths surfaces but includes an intensity difference component — when it detects a sharp edge, it stops smoothing at that point. Preserves dendrite wall sharpness while removing internal and external noise. Avoid Gaussian Blur which blurs dendrite edges and loses critical branch thickness information.

### Stage B: Segmentation and Denoising

1. **Adaptive Thresholding (preferred for SEM):** Computes a dynamic threshold per pixel based on its neighborhood average. Works well with non-uniform illumination.

2. **Otsu's Binarization:** Only suitable if illumination is completely uniform (no shadows). Statistical-global method finding the optimal threshold minimizing within-class variance.

### Stage C: Morphological Post-Processing

1. **Morphological Reconstruction (Geodesic Dilation):**
   - **Mask:** Original binary image (contains full dendrite + noise)
   - **Marker:** Image after aggressive erosion — only the thickest, most certain branch cores remain (no background noise at all)
   - **Process:** Iterative dilation of the Marker within the Mask. Seeds grow until they meet the original dendrite boundaries in the mask.
   - This cleans noise without damaging dendrite structure (unlike regular Opening which can erase thin branches)

2. **Closing:** Morphological operation to fill holes and create full continuity.

3. **Connected Components Analysis:** Every object with area below a threshold (e.g., < 50 pixels) is classified as noise and removed, under the physical assumption that a dendrite is a continuous, large structure.

### Stage D: Separation Algorithms

- **Distance Transform** combined with **Watershed** algorithm to separate branches that touch each other.

## Evaluation Metrics

1. **Dice Score / IoU:** Overlap measurement between algorithm output and manual ground truth.
2. **Robustness:** Algorithm stability across different images.
3. **Failure Analysis:** In-depth discussion of failure cases with root cause characterization (e.g., borderline resolution, unique artifacts).

## Submission Requirements

### A. Summary Report (PDF)
Technical paper structure:
1. Abstract: problem definition, chosen solution, main results
2. Methodology: detailed explanation of both pipelines (classic and DL), including parameter choices
3. Results and Discussion: accuracy metrics (IoU/Precision/Recall), comparison tables, visual analysis of successes and failures
4. Conclusions: which method to use in which scenario

### B. Source Code
- Organized and documented code (Comments + Docstrings)
- `requirements.txt` for installation
- `README.md` with clear run instructions (how to run training, how to run inference)
- Allowed libraries: cv2 and scikit-image

### C. Results Artifacts
- At least 5 visual examples showing: source image -> mask (classic) -> mask (YOLO) -> skeleton
- Weights file: `.pt` of the best trained model (can submit Drive link)

### D. Presentation
5-minute presentation including a live demo with a recorded video of the algorithm running on a new dataset.
