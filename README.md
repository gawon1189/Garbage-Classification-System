# ♻️ 生活垃圾自动分类系统 (Garbage-Classification-System)

---

## 🌿 1. GitHub 分支使用规范
目前所有组员均已成功加入项目 Collaborators。**请尽量不要直接提交（Push）代码到 `main` 分支！**

在 GitHub 网页端上传文件或修改代码前，请先在页面左上角将分支从 `main` 切换到**自己的专属分支**，并在各自的分支内进行开发。

### 👥 各组成员专属分支及分工：
* 📊 **`data`** ：**李家愿**（数据收集与处理）
* 🤖 **`model`** ：**黄俊乐**（模型选择与训练）
* 📈 **`analysis`** ：**陈湘媛**（分错样本分析 + 指标分析）
* 🎨 **`frontend`** ：**李悦慈**（前端开发）
* 🔗 **`src`** ：**张诚**（前后端连接）

---

## 📦 2. 大文件与模型存储规范（请勿上传至 GitHub）
由于原始图片数据集和大体积的模型权重文件（如 `.pt`, `.onnx` 等）体积过大，GitHub 无法直接承载，且项目已设置 `.gitignore` 自动过滤。**请大家不要将这些大文件直接上传到 GitHub！**

所有的数据集和训练好的模型文件，请统一上传、更新至以下北大网盘链接：
* 🌐 **校园网盘地址：** https://disk.pku.edu.cn/link/ARA7FA87114B884B2B864CB8C85FDA58E6
* 💡 **协作说明：**
  * **李家愿** 会将清洗后的标准数据集上传至该网盘。
  * **黄俊乐** 请将训练好的最终模型文件同步到该网盘中，以便 **张诚** 下载并进行前后端连接。

---

## 📂 3. 项目目录结构

```
Garbage-Classification-System/
├── README.md                    # 本文件
├── _docs/                       # 展示文稿与文档备份
│   ├── README.md
│   └── data_collection.md
├── analysis/                    # 模型分析与评估
│   ├── README.md
│   └── error_sample_analysis/   # V1/V2 分析报告与可视化
│       ├── model_error_and_metrics_analysis.ipynb
│       ├── v1_v2_comparison_analysis.ipynb
│       ├── figures/             # 可视化图表
│       ├── tables/              # 指标汇总表
│       └── supporting_files/    # 优化方案、补充数据报告等
├── data/                        # 数据集（gitignored，通过网盘获取）
├── frontend/                    # 前端页面
│   ├── README.md
│   └── index.html               # 单页应用（HTML/CSS/JS）
├── model/                       # 模型文件
│   ├── README.md                # 版本对比与说明
│   ├── current/                 # 后端当前使用的模型（gitignored）
│   │   ├── model.pt             # V2 权重
│   │   └── class_names.json
│   ├── v1/                      # V1 完整记录
│   │   ├── class_names.json
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── ...（评估图表）
│   └── v2/                      # V2 完整记录
│       ├── train_classifier_v2.py
│       ├── requirements.txt
│       ├── README.md
│       ├── V2_TRAINING.md
│       ├── class_names.json
│       └── ...（评估图表）
└── src/                         # 后端 API 服务
    ├── README.md
    ├── requirements.txt
    └── backend/
        ├── __init__.py
        └── server.py            # FastAPI 应用
```

> **注意：** `data/`、`model/current/`、`*.pt`、`*.onnx`、`.venv/` 等均由 `.gitignore` 忽略，不提交到 GitHub。模型权重通过北大网盘同步。

---

## 🚀 4. 本地启动

```bash
# 后端
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
uvicorn src.backend.server:app --host 0.0.0.0 --port 8000 --reload

# 前端（另一个终端）
python3 -m http.server 3000 --directory frontend
```

打开 `http://localhost:3000` 使用前端界面，后端 API 在 `http://localhost:8000`。

---

