# V2 Model Result

This folder contains the optimized V2 garbage classification model script, training instructions, and Kaggle output files.

## Files

- `train_classifier_v2.py`: optimized training script for ConvNeXt-Tiny V2.
- `requirements.txt`: Python dependencies for local or Colab training.
- `V2_TRAINING.md`: detailed training notes and collaboration rules.
- `best_model.pt`: trained V2 model weights (gitignored by `*.pt`).
- Evaluation artifacts: `class_names.json`, `classification_report.txt`, `confusion_matrix.png`, `metrics.json`, `misclassified_samples.csv`, `test_metrics.json`, `train_log.csv`.

## V2 Result Summary

V2 was trained with:

- model: `convnext_tiny`
- image size: `384`
- loss: `focal`
- hard class weight: `1.8`
- supplement samples: `583`
- classes: `battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`

Main test results:

| Metric | V1 | V2 |
|---|---:|---:|
| Accuracy | 0.9084 | 0.9237 |
| Macro F1 | 0.9176 | 0.9299 |
| Weighted F1 | 0.9085 | 0.9235 |
| Macro ROC-AUC OvR | 0.9834 | 0.9897 |

The improvement is moderate but consistent. V2 especially improves the previously weaker classes such as `cardboard`, `paper`, `plastic`, and `trash`.

## Data Layout

The script expects this split data structure:

```text
data/split_data/
  train/
    battery/
    biological/
    ...
  val/
  test/
```

Optional supplement data can be stored separately:

```text
data/supplement_candidates/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Supplement data does not need to include all 10 classes. It only needs to use class names that already exist in `split_data/train`.

## Run On Kaggle

1. Upload the project or add it as a Kaggle Dataset.
2. Turn on GPU in Notebook settings.
3. Turn on Internet if pretrained weights need to be downloaded.
4. Copy the read-only input project to `/kaggle/working` before editing or training.

Example:

```bash
!cp -r /kaggle/input/datasets/sgwunc/garbage-classification-system/Garbage-Classification-System /kaggle/working/Garbage-Classification-System
```

Training command:

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

If the hard-example CSV is available, add:

```bash
--hard-example-csv /path/to/hard_examples_for_v2.csv
```

V2 can still run without this CSV because focal loss, hard-class weighting, sampler, augmentation, and supplement data are already enabled.

If GPU memory is not enough, change:

```bash
--batch-size 16
```

to:

```bash
--batch-size 8
```

If Kaggle cannot download pretrained weights, either turn on Internet or add:

```bash
--no-pretrained
```

## Run On Google Colab

1. Upload the project to Google Drive.
2. Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install dependencies:

```bash
!pip install -r "/content/drive/MyDrive/Garbage-Classification-System/model/v2/requirements.txt"
```

4. Run training:

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

## Run Locally On macOS

Local CPU training is not recommended because ConvNeXt-Tiny at 384px is slow. Local execution is mainly useful for checking syntax or running a short smoke test.

Create an environment:

```bash
cd /Users/sylviachan/Desktop/機器學習/分組/Garbage-Classification-System
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r model/v2/requirements.txt
```

Short smoke test:

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

Full local training should only be done on a machine with a suitable GPU.

## Collaboration Notes

- Do not overwrite the original V1 `best_model.pt` until the group agrees to switch.
- Confirm the class order is unchanged before backend integration.
- If GitHub rejects large files, do not push `.pt` model weights. Share them through Kaggle output, cloud drive, or Git LFS.
