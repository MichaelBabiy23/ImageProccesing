# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University image processing and deep learning course repository. Contains completed exercises (ex1-ex4) covering classical computer vision through CNNs, plus a final project. Student ID: 323073734.

## Running Exercises

Each exercise is a standalone Python script. Run from the repo root:

```bash
python ex1/Main_323073734.py          # Canny edge detection + Hough line transform
python ex2/code_starter_for_stud_ex1.py  # Orientation-consistent RANSAC
python ex2/code_starter_for_stud_ex2.py  # Scale-ratio RANSAC verification
python ex3/code_starter_for_stud.py      # CNN feature map explorer
python ex4/code_starter_for_stud.py      # Hyperparameter optimizer
```

ex2 scripts and ex3/ex4 have built-in test functions that run via `__main__`.

## Dependencies

- **numpy** - all exercises
- **opencv-python (cv2)** - ex1, ex2 (image I/O, SIFT, homography)
- **pandas** - ex1 (CSV output)
- Python 3.12 (based on cached `.pyc` files)

ex3 uses only numpy. ex4 uses only stdlib + its local `mock_trainer.py`.

## Architecture

### ex1 - Document Separator Detection (Main_323073734.py)
Full pipeline in a single file: load grayscale image, build blob mask (Gaussian blur + Canny + morphological ops), detect lines via Hough transform, snap/clip/merge lines, crop documents. All CV operations implemented from scratch (no cv2 filtering functions). Expects input images at `/kaggle/input/images`. Outputs annotated images and `lines_data.csv`.

### ex2 - RANSAC Variants (two files)
- `code_starter_for_stud_ex1.py`: RANSAC + SIFT orientation consistency verification for homography estimation
- `code_starter_for_stud_ex2.py`: RANSAC + SIFT scale-ratio (Jacobian determinant) verification

Both follow the same pattern: extract SIFT features, match with ratio test, run RANSAC with additional geometric verification, refine with all inliers.

### ex3 - CNN Forward Pass (code_starter_for_stud.py)
Numpy-only CNN simulator: convolution layer with multiple filters, activation functions (ReLU/leaky ReLU/sigmoid), max pooling, pipeline builder that tracks dimensions, and receptive field calculator. Data format: feature maps are (H, W, C), filters are (num_filters, kH, kW, C_in).

### ex4 - Hyperparameter Optimizer (code_starter_for_stud.py + mock_trainer.py)
Random search optimizer for Fashion-MNIST (simulated). `mock_trainer.py` provides `MockModelTrainer` that generates deterministic fake training curves. The main file implements search space definition, sampling, training with early stopping, experiment logging, and best config selection.

## Course Lecture Materials (PDFs)

- `im2023_lect_02_Point_operation_histEQ_v2.pdf` - Point operations, histogram equalization
- `im2025_lect_03_n_04_spatial_operations.pdf` - Spatial operations
- `im2023_lect_04_spataial_operations_second_deriv_v1.pdf` - Second derivative spatial ops
- `im2025_lect_05_morphological.pdf` - Morphological operations
- `im2025_lect_06_CANNY_HT_v1.pdf` - Canny edge detection, Hough Transform
- `im2025_lect_08_Harris_pyramid_SIFT_v1.pdf` - Harris corners, image pyramids, SIFT
- `im2025_lect_10_Stitching_RANSAC_v1.pdf` - Image stitching, RANSAC
- `DeepLearning_1day_tutorial_1.pdf` / `DeepLearning_1day_tutorial_2.pdf` - Deep learning tutorials

## Final Project: Morphological Segmentation of SEM Dendrites

The active focus of this repository. The project definition is in `final project/עיבוד תמונה - פרויקט סיום.pdf`.

### Goal

Automatic pixel-wise semantic segmentation of lithium dendrites in SEM (Scanning Electron Microscope) images. Dendrites are fractal/branch-like metallic microstructures that grow on battery anodes during charging.

### Two Required Approaches (must compare both)

**Approach A - Deep Learning (YOLO-Seg):** YOLOv8/v11 instance segmentation with transfer learning. Data must be labeled with polygons (Roboflow or CVAT recommended).

**Approach B - Classic CV Pipeline (no neural networks):** Allowed libs: cv2, scikit-image.
1. Pre-processing: Histogram normalization, CLAHE, Bilateral Filter (edge-preserving denoise)
2. Segmentation: Adaptive Thresholding (preferred for SEM) or Otsu's Binarization
3. Post-processing: Morphological Reconstruction (erosion seeds + geodesic dilation), Closing, Connected Components filtering (remove small noise < 50px)
4. Separation: Distance Transform + Watershed for touching branches

### Required Deliverables

1. Binary mask (dendrite vs background)
2. Pre-processing/cleaning (remove text, scale bars, sensor noise)
3. Skeleton (single-pixel centerline extraction)
4. Technical report PDF, `requirements.txt`, `README.md` with run instructions
5. At least 5 visual examples: source -> classic mask -> YOLO mask -> skeleton
6. Best YOLO weights (.pt file)
7. 5-minute presentation with live demo video on new data

### Evaluation Metrics

- Dice Score / IoU against manual ground truth
- Robustness across different images
- Failure analysis with root cause characterization

### SEM Image Challenges

Low SNR (shot noise), saturation artifacts, Charging Effect blur, soft gradients. Intensity should be treated as topographic height — avoid relying on global thresholds alone.

### Relevant Prior Work in This Repo

- **ex1** has from-scratch implementations of: Gaussian blur, Canny edge detection, morphological ops (dilate, erode, close), connected components, Hough transform — directly applicable to the classic pipeline
- **ex2** covers SIFT features and homography — relevant background for feature-based approaches
- **Lecture PDFs** on morphological operations (lect_05) and Canny/Hough (lect_06) are directly relevant

## Conventions

- Exercises use "starter code" pattern: function signatures and test functions are provided (marked "DO NOT MODIFY"), students fill in implementations
- ex1 deliberately avoids cv2 high-level functions (implements convolution, morphology, connected components, Bresenham's line drawing manually)
- ex2 uses cv2 for SIFT and `findHomography` but implements all verification logic manually
