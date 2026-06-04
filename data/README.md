# 数据收集与处理说明

本目录用于存放垃圾图像分类任务的数据处理代码、探索性数据分析 Notebook，以及本地数据集目录说明。

由于原始图片数据和处理后的训练数据体积较大，实际图片文件不建议上传至 GitHub。需要复现数据处理、模型训练或模型评估的同学，请从北大网盘下载对应压缩包。

北大网盘地址：

```text
https://disk.pku.edu.cn/link/ARA7FA87114B884B2B864CB8C85FDA58E6
```

## 1. 原始数据 raw_data

`raw_data` 用于存放从网盘下载的原始数据压缩包或解压后的原始数据目录。

当前涉及的数据源包括：

```text
GD.zip
TACO.zip
Trashnet.zip
Integrated_Dataset_384.zip
```

各数据集含义如下：

```text
GD.zip
  主数据集 The Garbage Dataset，包含多类真实生活场景垃圾图像，是本项目主要训练来源。

TACO.zip
  TACO 数据集，包含复杂背景、野外场景和标注信息，用于增强模型对复杂环境的鲁棒性。

Trashnet.zip
  TrashNet 数据集，经典垃圾分类数据集，图像背景较干净，常用于补充基础材质特征。

Integrated_Dataset_384.zip
  数据处理后得到的融合数据集，已统一类别并处理为 384x384 图像，是后续划分 train/val/test 的来源。
```

推荐本地目录结构：

```text
data/
  raw_data/
    GD.zip
    TACO.zip
    Trashnet.zip
    Integrated_Dataset_384.zip
```

如果只进行模型训练和评估，通常不需要重新处理 `GD.zip`、`TACO.zip` 和 `Trashnet.zip`，直接使用处理后的 `Integrated_Dataset_384.zip` 或 `split_dataset` 即可。

## 2. 融合数据 Integrated_Dataset_384

`Integrated_Dataset_384` 是由 GD、TACO 和 TrashNet 经过类别映射、去重、裁剪/缩放等处理后生成的 10 类融合数据集。

类别包括：

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

典型目录结构：

```text
Integrated_Dataset_384/
  battery/
  biological/
  cardboard/
  clothes/
  glass/
  metal/
  paper/
  plastic/
  shoes/
  trash/
```

该数据集是 `split_dataset` 的直接来源。

## 3. 训练划分 split_dataset

`split_dataset` 是由 `Integrated_Dataset_384` 按类别分层划分得到的训练、验证和测试数据集，用于后续模型训练与评估。

目录结构：

```text
split_dataset/
  train/
    battery/
    biological/
    ...
  val/
    battery/
    biological/
    ...
  test/
    battery/
    biological/
    ...
  class_names.json
  split_manifest.csv
```

划分比例：

```text
train: 80%
val:   10%
test:  10%
```

当前划分后的样本数量：

```text
battery:    train 423, val 52,  test 54
biological: train 400, val 50,  test 51
cardboard:  train 895, val 111, test 113
clothes:    train 671, val 83,  test 85
glass:      train 916, val 114, test 116
metal:      train 873, val 109, test 110
paper:      train 891, val 111, test 112
plastic:    train 904, val 113, test 113
shoes:      train 676, val 84,  test 85
trash:      train 615, val 76,  test 78
```

总样本数为 9084 张，其中测试集为 917 张。

## 4. 相关代码文件

```text
data_processing.ipynb
  多源数据融合与预处理流程，包括类别映射、pHash 去重、TACO 裁剪和 Letterbox 缩放。

EDA.ipynb
  数据探索性分析，包括类别分布、图像信息熵、深度特征可视化和异常样本检测。

create_train_val_test_split.py
  从 Integrated_Dataset_384 生成 split_dataset 的脚本。
```

重新生成 `split_dataset`：

```bash
python data/create_train_val_test_split.py --overwrite
```

## 5. 原始数据公开来源

部分数据也可以从以下公开链接获取：

```text
https://www.kaggle.com/datasets/kneroma/tacotrashdataset
https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
https://www.kaggle.com/datasets/feyzazkefe/trashnet
```
