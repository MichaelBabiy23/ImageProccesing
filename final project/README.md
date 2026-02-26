# SEM Dendrite Segmentation

Automatic pixel-wise semantic segmentation of lithium dendrites in Scanning Electron Microscope (SEM) images. Implements and compares two approaches: a classical computer vision pipeline and a deep learning (YOLO-Seg) pipeline.

Student ID: 323073734

## Setup

```bash
pip install -r requirements.txt
```

## Classic Pipeline

Stages: histogram normalization, CLAHE, bilateral filter, adaptive thresholding, morphological reconstruction, closing, small component removal, distance transform + watershed separation, and skeletonization.

Single image:

```bash
python classic_pipeline.py <path/to/image.png>
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

## Evaluation

Computes Dice, IoU, Precision, and Recall against ground truth masks. Generates 4-panel comparison figures (Source, Classic Mask, YOLO Mask, Skeleton) and a text summary.

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

### Self-Test

```bash
python evaluate.py
```

Runs synthetic tests verifying metric correctness (Dice=1.0 for perfect overlap, ~0.67 for half overlap, 0.0 for no overlap) and generates a sample comparison figure.

## Output Structure

```
output/
  classic/
    <image_name>/
      01_original.png ... 11_skeleton.png
  yolo/
    train/dendrite_seg/weights/best.pt
    <image_name>_mask.png
  evaluation/
    <image_name>_comparison.png
    metrics_summary.txt
```
