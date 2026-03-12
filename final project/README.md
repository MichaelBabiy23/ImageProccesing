# SEM Dendrite Segmentation

Automatic pixel-wise semantic segmentation of lithium dendrites in Scanning Electron Microscope (SEM) images. Implements and compares two approaches: a classical computer vision pipeline and a deep learning (YOLO-Seg) pipeline.

Student ID: 323073734

## Setup

```bash
pip install -r requirements.txt
```

## Interactive GUI

Windows launcher:

```bash
run-gui.bat
```

macOS / Linux:

```bash
python3 pipeline_gui.py
```

On macOS you can also use the bundled launcher:

```bash
./run-gui.command
```

## Classic Pipeline

Stages: histogram normalization, CLAHE, bilateral filter, adaptive thresholding, morphological reconstruction, closing, small component removal, distance transform + watershed separation, and skeletonization.

Single image:

```bash
python classic_pipeline.py <path/to/image.png>
```

Single image with custom output directory and no intermediate stages:

```bash
python classic_pipeline.py <path/to/image.png> --output results/ --no-intermediates
```

Batch (all images in a directory):

```bash
python classic_pipeline.py --input <images_dir> --output <output_dir>
```

Running without arguments executes a synthetic self-test:

```bash
python classic_pipeline.py
```

Intermediate images (11 stages) are saved per image under `output/classic/<image_name>/`.

## YOLO Pipeline

Uses YOLOv11 instance segmentation with transfer learning from COCO pretrained weights.

### Dataset Preparation

Label images with polygon annotations using Roboflow or CVAT, then export in YOLOv8 Segmentation format. Expected structure:

```
dataset/
  data.yaml
  train/images/  train/labels/
  valid/images/  valid/labels/
```

### Training

```bash
python yolo_pipeline.py train --data <path/to/data.yaml> [--epochs 100] [--model yolo11n-seg.pt]
```

Best weights are saved to `output/yolo/train/dendrite_seg/weights/best.pt`.

### Inference

Single image:

```bash
python yolo_pipeline.py predict --model <weights.pt> --source <image.png>
```

Batch:

```bash
python yolo_pipeline.py predict --model <weights.pt> --source <images_dir> --output <output_dir>
```

## Full Pipeline (run_all.py)

The orchestrator runs both pipelines on the same images and produces all deliverables in one command.

Both pipelines with evaluation:

```bash
python run_all.py --images data/sem/ --gt data/ground_truth/ --yolo-model weights/best.pt --output output/
```

Classic pipeline only (no YOLO model):

```bash
python run_all.py --images data/sem/ --gt data/ground_truth/
```

Without ground truth (no metrics, just masks and comparison figures):

```bash
python run_all.py --images data/sem/
```

Self-test with synthetic images:

```bash
python run_all.py
```

The orchestrator runs 5 stages:
1. Classic pipeline on all images
2. YOLO pipeline (if model provided)
3. Skeletonization of classic masks
4. 4-panel comparison figures (Source, Classic, YOLO, Skeleton)
5. Evaluation with metrics and failure analysis (if ground truth provided)

## Evaluation

Computes Dice, IoU, Precision, and Recall against ground truth masks. Generates 4-panel comparison figures (Source, Classic Mask, YOLO Mask, Skeleton) and a text summary with failure analysis.

### Batch Evaluation

```python
from evaluate import evaluate_all

evaluate_all(
    classic_dir="output/classic",
    yolo_dir="output/yolo",
    gt_dir="data/ground_truth",
    image_dir="data/images",
    output_dir="output/evaluation"
)
```

### Failure Analysis

The evaluation module automatically performs failure analysis on images with Dice score below a configurable threshold (default: 0.5). For each failure, a root cause is characterized based on metric patterns:

- **Over-segmentation** (low precision, high recall) — noise or background included as dendrite
- **Under-segmentation** (high precision, low recall) — thin branches or low-contrast dendrites missed
- **Fundamental mismatch** (low precision, low recall) — wrong region detected or severe image artifacts
- Cross-method insights: if classic fails but YOLO succeeds, the likely cause is non-uniform illumination (classic threshold sensitivity); if YOLO fails but classic succeeds, the sample may be out-of-distribution for the trained model

The failure report is appended to the metrics summary text file.

### Self-Test

```bash
python evaluate.py
```

Runs synthetic tests verifying metric correctness (Dice=1.0 for perfect overlap, ~0.67 for half overlap, 0.0 for no overlap), generates a sample comparison figure, and validates the failure analysis logic.

## Output Structure

```
output/
  classic/
    <image_name>/
      01_original.png ... 11_skeleton.png
  yolo/
    train/dendrite_seg/weights/best.pt
    <image_name>_mask.png
  comparisons/
    <image_name>_comparison.png
  evaluation/
    <image_name>_comparison.png
    metrics_summary.txt
```
