# V2 Training Guide

This guide explains how to train and evaluate the optimized V2 classifier.

## Purpose

V2 keeps the original V1 model untouched and trains a separate optimized model. The main goal is to improve classes that were weak in V1:

```text
cardboard, paper, plastic, trash, glass
```

## Optimization Methods

- ConvNeXt-Tiny backbone with ImageNet pretrained weights.
- Focal Loss for difficult samples.
- Hard class weighting for frequently confused classes.
- WeightedRandomSampler for class and hard-class balance.
- Stronger data augmentation.
- Optional supplement data through `--supplement-dir`.
- Early stopping and best validation checkpoint saving.

## Required Data

Split data:

```text
data/split_data/train
data/split_data/val
data/split_data/test
```

Optional supplement data:

```text
data/supplement_candidates
```

The supplement folder may contain only some classes. It does not need to match all 10 train classes.

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Main packages:

```text
torch
torchvision
numpy
pandas
scikit-learn
matplotlib
pillow
```

## Kaggle Training

Use Kaggle GPU when possible.

```bash
!python /kaggle/working/Garbage-Classification-System/model/v2/train_classifier_v2.py \
  --data-dir /kaggle/working/Garbage-Classification-System/data/split_data \
  --output-dir /kaggle/working/runs/convnext_tiny_v2 \
  --model convnext_tiny \
  --img-size 384 \
  --batch-size 16 \
  --epochs 25 \
  --loss focal \
  --hard-class-weight 1.8 \
  --supplement-dir /kaggle/working/Garbage-Classification-System/data/supplement_candidates
```

If memory is not enough:

```bash
--batch-size 8
```

If pretrained weight download fails:

```bash
--no-pretrained
```

Using pretrained weights is recommended when Internet is available.

## Colab Training

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install -r "/content/drive/MyDrive/Garbage-Classification-System/model/v2/requirements.txt"
```

```bash
!python "/content/drive/MyDrive/Garbage-Classification-System/model/v2/train_classifier_v2.py" \
  --data-dir "/content/drive/MyDrive/Garbage-Classification-System/data/split_data" \
  --output-dir "/content/drive/MyDrive/Garbage-Classification-System/runs/convnext_tiny_v2" \
  --model convnext_tiny \
  --img-size 384 \
  --batch-size 16 \
  --epochs 25 \
  --loss focal \
  --hard-class-weight 1.8 \
  --supplement-dir "/content/drive/MyDrive/Garbage-Classification-System/data/supplement_candidates"
```

## Local macOS Smoke Test

Full local training is not recommended without GPU. Use local runs only for syntax or path checking.

```bash
cd /Users/sylviachan/Desktop/機器學習/分組/Garbage-Classification-System
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r model/v2/requirements.txt
```

```bash
python3 model/v2/train_classifier_v2.py \
  --data-dir data/split_data \
  --output-dir runs/convnext_tiny_v2_smoke \
  --model convnext_tiny \
  --img-size 224 \
  --batch-size 4 \
  --epochs 1 \
  --loss focal \
  --hard-class-weight 1.8 \
  --supplement-dir data/supplement_candidates \
  --no-pretrained
```

## Output Files

```text
best_model_v2.pt
class_names.json
train_log_v2.csv
metrics_v2.json
test_metrics_v2.json
classification_report_v2.txt
confusion_matrix_v2.png
misclassified_samples_v2.csv
```

## Switching Rule

Do not replace V1 immediately. Before switching backend inference to V2, confirm:

- class order is unchanged;
- model architecture is still `convnext_tiny`;
- image size is still `384`;
- backend path points to the V2 weight file;
- teammates agree to use the new model.
