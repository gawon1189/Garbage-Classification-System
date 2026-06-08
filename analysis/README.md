# analysis — 模型分析与评估

## 作用

对训练好的模型进行错误样本分析、指标评估和版本对比，为模型优化提供依据。

## 内容

### error_sample_analysis/

V1 和 V2 的完整分析报告：

| 文件 | 说明 |
|------|------|
| `model_error_and_metrics_analysis.ipynb` | V1 模型指标分析和错误样本解读 |
| `v1_v2_comparison_analysis.ipynb` | V1 与 V2 对比分析报告 |

### figures/

分析报告生成的可视化图表（混淆矩阵、ROC-AUC、错误率、训练曲线等）。

### supporting_files/

- `optimization_plan.md` — V2 优化方案
- `supplement_candidates_report.md` — 补充数据来源报告
- `corresponding_data_report.md` — 对应数据查找报告
- `tables/` — 各类指标 CSV 表
- 对比用的中间数据文件

## 文件结构

```
analysis/
├── README.md
└── error_sample_analysis/
    ├── README.md
    ├── model_error_and_metrics_analysis.ipynb   # V1 主分析报告
    ├── v1_v2_comparison_analysis.ipynb           # V1/V2 对比报告
    ├── figures/                                   # 可视化图表
    ├── tables/                                    # 指标汇总表
    └── supporting_files/                          # 支撑材料
```
