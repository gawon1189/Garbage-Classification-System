# model — 模型文件

## 目录结构

```
model/
├── README.md               # 本文件
├── current/                # 后端 API 当前使用的模型（基于 V2）
│   ├── model.pt
│   └── class_names.json
├── v1/                     # V1 模型完整记录
│   ├── best_model.pt
│   ├── class_names.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── misclassified_examples.png
│   ├── misclassified_samples.csv
│   ├── roc_curve_ovr.png
│   ├── test_metrics.json
│   └── train_log.csv
└── v2/                     # V2 模型完整记录
    ├── best_model.pt
    ├── class_names.json
    ├── classification_report.txt
    ├── confusion_matrix.png
    ├── metrics.json
    ├── misclassified_samples.csv
    ├── test_metrics.json
    ├── train_log.csv
    ├── train_classifier_v2.py
    ├── requirements.txt
    ├── README.md            # V2 训练说明
    └── V2_TRAINING.md       # V2 训练笔记
```

## 版本对比

| 指标 | V1 | V2 |
|------|----|----|
| 模型 | convnext_tiny | convnext_tiny |
| 输入尺寸 | 384×384 | 384×384 |
| 类别数 | 10 | 10 |
| 测试准确率 | 90.84% | 92.37% |
| Macro F1 | 0.9176 | 0.9299 |
| Weighted F1 | 0.9085 | 0.9235 |
| ROC-AUC (macro) | 0.9834 | 0.9897 |
| 损失函数 | CrossEntropy | Focal Loss |
| 补充数据 | 无 | +583 张 |

## 说明

- `current/` 是后端 `server.py` 引用的目录，当前基于 V2 权重，由 `.gitignore` 忽略，不上传 GitHub
- 模型权重 (`*.pt`) 由 `.gitignore` 全局忽略，通过北大网盘分享
- V1/V2 的评估报告和图表为小文件，可直接在仓库中查看
