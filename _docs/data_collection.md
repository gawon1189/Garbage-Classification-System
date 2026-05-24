# 基于物理材质原生态分类（Level 1）的多源数据集融合方案

## Multi-Source Dataset Fusion Scheme Based on Level 1 Physical Material Taxonomy

---

### 1. 选定数据集及官方来源 (Selected Datasets & Official Sources)

#### 1.1 主干训练集 (Main Spine): The Garbage Dataset (GD) [Kunwar, 2026]

* **数据占比：** 65% ~ 70% 


* **数据体量：** 12,259 张真实生活场景垃圾图像，包含 10 大物理材质类别 。


* **官方来源：** 通过 DWaste 智能垃圾分类移动端 App 采集与社区众包整合，其图像具备高背景香农熵（$6.2\pm1.5$ bits/pixel）与真实光照偏差，构成模型泛化的根本基石。

#### 1.2 野外鲁棒增强 (Wild Robustness): TACO

* **数据占比：** 20%
* **数据体量：** 1,500 张包含野外真实背景（森林、海滩、街头）的高精细分割标注图像。
* **官方来源：** 基于 Flickr 平台的垃圾众包标注项目，引入丰富的室外复杂光影噪声和物理遮挡，攻克模型在野外场景下的识别盲区。

#### 1.3 几何特征收敛 (Geometric Feature Convergence): TrashNet

* **数据占比：** 10% ~ 15%
* **数据体量：** 2,527 张标准单一废弃物图像。
* **官方来源：** 斯坦福大学经典学术数据集，在纯白背景与受控光源下拍摄，提供极其纯净、无背景噪声的特征，辅助骨干网络快速提取基础材质机理与物理边界。

---

### 2. 为什么选择物理材质原生态分类（Level 1）作为基准？ (Why Level 1 Physical Classification?)

* **超越地域法规的通用性 (Universal Adaptability)**
* 避免将深度学习模型直接绑定在特定城市的临时行政垃圾分类规章上（如北京 4 分类或首尔分拣机制），回归物体最本质的物理材质属性 。




* **高效的下游动态映射 (Dynamic Downstream Mapping)**
* 训练阶段以物理类别为基准输出（金属、纸张、玻璃、塑料等），在应用部署端通过轻量化 JSON 映射表，即可瞬间零成本适配全球任何城市（如北京的厨余/可回收/有害/其他 4 类体系） 。





---

### 3. 核心技术加工与对齐管道 (Core Technical Processing & Feeding Pipeline)

* **分类标签对齐 (Category Alignment)**
* 利用 TACO 官方提供的 `annotations.json`，将 60 个细分维度标签严格重映射并合并至 GD 数据集的 10 大核心物理材质系统，实现多源标签格式的无缝整合。


* **长宽比零失真预处理 (Letterbox Padding)**
* 废弃会导致物体几何畸变的常规拉伸手段，采用边缘等比填充（Letterbox Padding）将分辨率标准化至 384x384 或 512x512 像素，完美保留瓶罐等物体的本质几何轮廓特征。


* **严格数据治理与隔离 (Rigorous Data Isolation)**
* 引入感知哈希（pHash）剔除跨库冗余图像 ；采用分层随机抽样（Stratified Sampling）按 80% / 10% / 10% 严格划分 Train / Val / Test 集合，杜绝任何学术层面的“数据泄露”。




* **像素级多级鲁棒增强 (Data Augmentation)**
* 仅在训练集中集成 MixUp、CutOut 以及 HSV 色彩扰动，测试集保持绝对干净，以科学客观地评估模型的分类性能上限。
