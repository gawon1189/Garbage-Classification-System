# frontend — 前端页面

## 作用

用户交互界面，提供拍照/上传图片、查看分类结果、城市处理指南等功能。

## 功能

- 拖拽/点击上传垃圾图片
- 调用后端 `/predict` 接口进行识别
- 展示识别结果（类别、置信度、Top-3 候选）
- 城市垃圾分类指南（38 个中国城市）
- 批量上传与摘要报告
- 错误提示与 Loading 状态

## 文件结构

```
frontend/
├── README.md     # 本文件
└── index.html    # 单页应用（含完整 HTML/CSS/JS）
```

## 启动方式

```bash
python3 -m http.server 3000 --directory frontend
```

打开 `http://localhost:3000` 即可访问。

> 注意：前端需要后端 API 服务在 `http://localhost:8000` 运行，否则无法分类。
