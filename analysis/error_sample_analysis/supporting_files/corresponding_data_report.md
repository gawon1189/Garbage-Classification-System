# 对应数据查找结果

## 1. 需要重点补充/复核的类别

根据模型错误分析，应优先关注以下类别：`paper`、`cardboard`、`plastic`、`trash`，其次关注 `glass`、`metal`。

## 2. 当前 split_data 中的数量
v1 错分样本中有 `84` 个能在当前本机 `split_data` 中匹配到实际图片。


| class | train | val | test | total |
|---|---:|---:|---:|---:|
| cardboard | 895 | 111 | 113 | 1119 |
| glass | 916 | 114 | 116 | 1146 |
| metal | 873 | 109 | 110 | 1092 |
| paper | 891 | 111 | 112 | 1114 |
| plastic | 904 | 113 | 113 | 1130 |
| trash | 615 | 76 | 78 | 769 |

## 3. 已生成的数据清单

- `tables/target_classes_existing_data.csv`：目标类别全部现有图片路径。
- `tables/confusion_pair_reference_pool.csv`：按高频混淆方向整理的可参考图片池。
- `tables/misclassified_samples_local_matches.csv`：v1 错分样本和本机 `split_data` 的匹配结果。
- `tables/targeted_review_copied_files.csv`：已复制到复核文件夹的图片清单。
- `data/targeted_review_samples/`：便于人工打开查看的小样本文件夹。

## 4. 怎么使用这些数据

优先人工查看 `data/targeted_review_samples/v1_misclassified/`，这些是模型已经分错的样本，最适合作为 hard examples 或标签复核对象。

如果要继续补充数据，建议围绕以下方向找新图：

- `cardboard -> paper`：找厚纸、瓦楞纸、纸箱边缘、折痕明显的纸板。
- `paper -> cardboard`：找薄纸、打印纸、揉皱纸、包装纸。
- `plastic -> trash` / `trash -> plastic`：找脏污塑料、塑料袋、塑料包装、被遮挡塑料。
- `plastic -> glass` / `glass -> plastic`：找透明塑料瓶、透明玻璃瓶、强反光背景。
- `glass -> metal`：找带金属盖的玻璃瓶、反光玻璃和金属包装对比样本。
