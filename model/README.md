# 模型训练与评估说明

本目录存放垃圾图像分类任务的模型训练、测试评估代码及最终模型产物。本部分工作目标是基于已处理好的 `split_dataset` 数据集，训练一个 10 类垃圾图像分类模型，并在独立测试集上完成性能评估。

## 1. 任务设置

模型输入为垃圾图像，输出为 10 类物理材质/物品类别：

```text
battery
biological
cardboard
clothes
glass
metal
paper
plastic
shoes
trash
```

后续系统部署时，可以通过规则映射进一步转换为常见四分类结果：

```text
可回收物 / 有害垃圾 / 厨余垃圾 / 其他垃圾
```

## 2. 数据集

训练使用的数据集为数据处理阶段生成的：

```text
data/split_dataset/
  train/
  val/
  test/
  class_names.json
  split_manifest.csv
```

划分比例为：

```text
train: 80%
val:   10%
test:  10%
```

测试集共 917 张图片，仅用于最终评估，不参与训练和调参。

## 3. 文件说明

```text
model/
  train_classifier.py        # Kaggle/PyTorch 模型训练脚本
  evaluate_model.ipynb       # 模型测试与评估 Notebook
  README.md                  # 当前说明文件
  artifacts/                 # 训练与评估产物，不建议直接上传 GitHub
```

`model/artifacts/` 中的主要产物如下：

```text
best_model.pt                # 最优模型权重
class_names.json             # 类别顺序文件
train_log.csv                # 训练日志
metrics.json                 # 训练脚本生成的指标
test_metrics.json            # 测试集最终评估指标
classification_report.txt    # Precision / Recall / F1 报告
confusion_matrix.png         # 混淆矩阵
roc_curve_ovr.png            # 多分类 ROC 曲线
misclassified_samples.csv    # 错误分类样本明细
misclassified_examples.png   # 错误分类样本可视化
```

## 4. 模型方案

最终模型采用 `ConvNeXt-Tiny`，并使用 ImageNet 预训练权重进行迁移学习。

主要训练设置：

```text
Model: ConvNeXt-Tiny
Input size: 384 x 384
Number of classes: 10
Training platform: Kaggle GPU
Loss: CrossEntropyLoss
Optimizer: AdamW
Scheduler: CosineAnnealingLR
Epochs: 20
Best epoch: 15
```

训练过程中保存验证集表现最好的模型为：

```text
model/artifacts/best_model.pt
```

## 5. 测试集评估结果

最终模型在独立测试集上的结果如下：

```text
Test Accuracy:        0.9084
Macro Precision:      0.9209
Macro Recall:         0.9157
Macro F1:             0.9176
Weighted F1:          0.9085
ROC-AUC Macro OvR:    0.9834
ROC-AUC Weighted OvR: 0.9822
```

各类别 F1-score：

```text
battery:    0.9725
biological: 0.9800
cardboard:  0.8610
clothes:    0.9941
glass:      0.9304
metal:      0.9196
paper:      0.8619
plastic:    0.8288
shoes:      0.9881
trash:      0.8400
```

整体来看，模型在 `clothes`、`shoes`、`biological`、`battery` 等类别上表现较好；主要误差集中在 `paper`、`cardboard`、`plastic` 和 `trash` 等视觉特征相近的类别之间。

## 6. 运行方式

### 6.1 训练

在 Kaggle Notebook 中运行训练脚本：

```bash
python train_classifier.py \
  --data-dir /kaggle/input/datasets/chunloookkk/split-dataset/split_dataset \
  --output-dir /kaggle/working/runs/convnext_tiny \
  --model convnext_tiny \
  --img-size 384 \
  --batch-size 16 \
  --epochs 20
```

如果显存不足，可将 `--batch-size` 改为 `8`。

### 6.2 评估

使用 `evaluate_model.ipynb` 加载 `best_model.pt`，在 `test` 集上重新计算指标，并生成：

```text
test_metrics.json
classification_report.txt
confusion_matrix.png
roc_curve_ovr.png
misclassified_samples.csv
misclassified_examples.png
```

### 6.3 推理

推理脚本位于：

```text
src/inference.py
```

示例：

```bash
python src/inference.py --input path/to/image.jpg
```

该脚本会输出 10 类预测结果、中文类别、四分类结果和置信度，供后续前后端连接使用。

## 7. GitHub 与北大网盘上传说明

由于模型权重和数据集文件较大，不建议直接上传到 GitHub。尤其是：

```text
model/artifacts/best_model.pt
```

该文件约 111 MB，超过 GitHub 单文件推荐/限制范围，应上传至北大网盘。

建议上传到北大网盘的文件：

```text
model/artifacts/best_model.pt
model/artifacts/class_names.json
model/artifacts/test_metrics.json
model/artifacts/classification_report.txt
model/artifacts/confusion_matrix.png
model/artifacts/roc_curve_ovr.png
model/artifacts/misclassified_samples.csv
model/artifacts/misclassified_examples.png
model/artifacts/train_log.csv
data/split_dataset.zip
```

其中 `best_model.pt` 和 `data/split_dataset.zip` 最重要，其他组员后续进行前后端连接或复现实验时需要从网盘下载。

北大网盘地址：

```text
https://disk.pku.edu.cn/link/ARA7FA87114B884B2B864CB8C85FDA58E6
```

建议 GitHub 只保留代码和说明文件：

```text
model/train_classifier.py
model/evaluate_model.ipynb
model/README.md
src/inference.py
```

大文件统一通过北大网盘共享，避免占用 GitHub 仓库空间。
