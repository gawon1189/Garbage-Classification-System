# 基于分析的模型优化方案与协作影响控制

## 1. 是否会影响其他同学

如果只是补充分析报告、提出优化建议、生成图表，不会影响其他同学。

如果重新训练模型并替换现有文件，则可能影响负责后端推理、前端展示和前后端连接的同学。尤其不能直接覆盖以下文件：

- `best_model.pt`
- `class_names.json`
- `test_metrics.json`
- `classification_report.txt`
- `misclassified_samples.csv`

这些文件可能已经被后端或前端同学用于模型加载、类别映射和结果展示。

## 2. 安全优化原则

- 不覆盖原始模型产物，保留当前 v1 模型作为稳定基线。
- 新实验使用独立目录，例如 `model/artifacts/convnext_tiny_v2/`。
- 新模型使用清晰命名，例如 `best_model_v2.pt`。
- 类别顺序必须继续沿用原来的 10 类顺序，不能随意改 `class_names.json`。
- 输入尺寸继续保持 `384 x 384`，除非后端推理代码同步修改。
- 先在分析报告中比较 v1/v2 指标，再决定是否让全组切换到新模型。

## 3. 推荐优化方向

### 3.1 数据层面

根据错误分析，优先优化 `paper/cardboard/plastic/trash`：

- 增加 `paper/cardboard` 边界样本：厚纸、瓦楞纸、折痕纸箱、普通薄纸、揉皱纸。
- 增加 `plastic/trash` 边界样本：脏污塑料、透明塑料、塑料袋、塑料包装、被遮挡的塑料制品。
- 对高置信错分样本逐张复核，区分标签错误、背景干扰、局部裁剪和真实难例。
- 将确认无误的高置信错分样本加入 hard example set，用于下一轮训练。

### 3.2 训练层面

- 使用 class-balanced sampling 或 WeightedRandomSampler，提高困难类别出现频率。
- 尝试 focal loss，降低简单样本主导训练的情况。
- 保留 early stopping，避免训练集准确率继续升高但验证集不再提升。
- 针对透明、反光、遮挡和脏污场景增加数据增强。

### 3.3 部署层面

- 对低置信样本返回 top-k 结果，而不是只返回 top-1。
- 对高频混淆对进行二次确认，例如 `paper/cardboard`、`plastic/trash`、`plastic/glass`。
- 对 `trash` 类设置更谨慎的阈值，因为它是兜底类别，类内差异最大。

## 4. 建议实验流程

1. 保留当前模型为 v1，不移动、不覆盖。
2. 建立新实验目录：`model/artifacts/convnext_tiny_v2/`。
3. 使用相同类别顺序、相同输入尺寸训练 v2。
4. 在独立测试集上生成：
   - `test_metrics_v2.json`
   - `classification_report_v2.txt`
   - `confusion_matrix_v2.png`
   - `misclassified_samples_v2.csv`
5. 在 `/analysis` 中新增 v1/v2 对比表。
6. 如果 v2 在 `plastic/trash/cardboard/paper` 上明显提升，且整体 Accuracy/Macro-F1 不下降，再通知后端同学切换模型。

## 5. 切换模型前必须确认

- `class_names.json` 顺序没有变。
- 模型结构和 checkpoint 中的 `model_name` 能被推理代码识别。
- 输入尺寸仍为 `384`，或推理代码已同步更新。
- 后端同学确认新模型路径。
- 前端展示的中文类别和四分类映射没有变化。

## 6. 本阶段结论

当前阶段建议先提交分析与优化方案，不直接替换原模型。这样既体现了基于错误分析提出改进，也不会阻塞其他同学继续完成前端、后端和系统整合工作。
