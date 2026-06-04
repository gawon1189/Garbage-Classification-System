# 补充样本候选集说明

## 数据源

- TrashNet：Gary Thung 与 Mindy Yang 的 Stanford CS229 项目数据集，含 cardboard/glass/metal/paper/plastic/trash 六类。
- GD：本项目 raw_data 中已有的分类数据源，含与当前 10 类体系一致的垃圾图片。

本次没有直接修改 `train/val/test`，而是把候选图片放到 `data/supplement_candidates/`，建议人工复核后再加入训练集。

## 已选候选样本数

| class | selected |
|---|---:|
| cardboard | 120 |
| glass | 80 |
| metal | 80 |
| paper | 120 |
| plastic | 120 |
| trash | 63 |

## 建议

- `trash` 当前数量最少，候选样本应优先人工复核并加入训练集。
- `paper/cardboard` 要重点挑选边界样本，不要只增加很标准的白底图片。
- `plastic/glass/metal` 要重点看透明、反光、金属盖等混淆情况。
- 人工复核后，建议只把确认标签正确、视觉上有代表性的图片加入训练集。
