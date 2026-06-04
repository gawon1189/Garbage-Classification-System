#### 前后端连接

## 项目核心定位

本项目是一个生活垃圾自动分类系统。用户在前端上传一张垃圾图片后，后端调用训练好的图像分类模型，识别其物理材质类别，并把结果返回给前端展示。

当前系统以“物理材质分类”为核心口径，不直接绑定北京四分类或其他城市法规。模型先输出 `battery`、`plastic`、`paper` 等通用材质类别；如果后续需要城市规则，可以在应用层增加映射表。

## `src` 模块职责

`src` 是前端和模型之间的连接层，负责把训练好的模型包装成前端可以调用的 HTTP API。

本模块负责：

- 接收前端上传的图片。
- 校验图片格式和大小。
- 按模型要求进行图片预处理。
- 加载本地 `convnext_tiny` 模型并执行推理。
- 返回统一 JSON，包含主分类、置信度、Top-K 候选类别和错误信息。

本模块不负责：

- 训练模型。
- 清洗数据集。
- 上传模型权重到 GitHub。
- 直接输出北京四分类等法规类别。

## 本地模型文件

模型文件来自北大网盘，当前默认放置位置为：

```text
model/artifacts/artifacts/best_model.pt
model/artifacts/artifacts/class_names.json
```

==注意==：`model/artifacts/` 已加入 `.gitignore`，模型权重和下载产物不要提交到 GitHub。

模型信息：

- 模型结构：`convnext_tiny`
- 输入尺寸：`384 x 384`
- 类别数量：`10`
- 测试准确率：约 `90.84%`
- 模型版本：`convnext-tiny-v1`

## 类别顺序

后端接口中的类别顺序以 `model/artifacts/artifacts/class_names.json` 为准：

```json
[
  "battery",
  "biological",
  "cardboard",
  "clothes",
  "glass",
  "metal",
  "paper",
  "plastic",
  "shoes",
  "trash"
]
```

前端可以展示中文名称，但接口字段统一使用英文类别名。请不要在前端假设类别顺序，建议通过 `GET /meta` 读取。

## 运行方式

建议在项目根目录下创建虚拟环境并安装依赖：

```bash
cd /Users/contramundum/Documents/pku/2026spring/机器学习/小组作业/Garbage-Classification-System
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

启动服务：

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务默认地址：

```text
http://localhost:8000
```

## 接口约定

### `GET /health`

检查 API 服务和模型加载状态。

成功响应示例：

```json
{
  "success": true,
  "status": "ok",
  "service": "garbage-classification-api",
  "model_version": "convnext-tiny-v1",
  "model": {
    "name": "convnext_tiny",
    "ready": true,
    "path": ".../model/artifacts/artifacts/best_model.pt",
    "error": null
  }
}
```

### `GET /meta`

获取类别列表、模型信息和上传限制。

成功响应示例：

```json
{
  "success": true,
  "model_version": "convnext-tiny-v1",
  "model": {
    "name": "convnext_tiny",
    "image_size": 384,
    "ready": true,
    "error": null
  },
  "classes": [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
  ],
  "upload": {
    "field_name": "file",
    "content_type": "multipart/form-data",
    "max_size_mb": 10,
    "supported_extensions": ["jpg", "jpeg", "png", "webp"],
    "supported_content_types": ["image/jpeg", "image/png", "image/webp"]
  },
  "prediction": {
    "top_k": 3,
    "classification_type": "physical_material"
  }
}
```

### `POST /predict`

上传单张图片并获取分类结果。

请求要求：

- `Content-Type`: `multipart/form-data`
- 表单字段名：`file`
- 单次只上传一张图片。
- 支持图片类型：`jpg`、`jpeg`、`png`、`webp`
- 最大文件大小：`10MB`

前端请求示例：

```javascript
const formData = new FormData();
formData.append("file", selectedFile);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

const result = await response.json();
```

成功响应示例：

```json
{
  "success": true,
  "request_id": "b4cf821df4ab",
  "predicted_class": "plastic",
  "confidence": 0.93,
  "top_k": [
    {"class": "plastic", "score": 0.93},
    {"class": "paper", "score": 0.04},
    {"class": "trash", "score": 0.01}
  ],
  "model_version": "convnext-tiny-v1",
  "processing_ms": 87
}
```

失败响应示例：

```json
{
  "success": false,
  "request_id": "b4cf821df4ab",
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Uploaded file is not a readable image"
  }
}
```

## 错误码约定

| code | HTTP 状态 | 场景 | 前端建议 |
| --- | --- | --- | --- |
| `MISSING_FILE` | `400` | 没有上传 `file` 字段 | 提示用户选择图片 |
| `UNSUPPORTED_FILE_TYPE` | `400` | 文件类型不是 `jpg`、`jpeg`、`png`、`webp` | 提示用户更换图片格式 |
| `FILE_TOO_LARGE` | `413` | 文件超过 `10MB` | 提示用户压缩或更换图片 |
| `INVALID_IMAGE` | `400` | 文件无法被识别为有效图片 | 提示用户重新选择图片 |
| `MODEL_NOT_READY` | `503` | 模型暂未加载完成或加载失败 | 提示稍后重试，或联系后端同学检查模型路径 |
| `INFERENCE_FAILED` | `500` | 推理过程失败 | 提示分类失败，可重试 |

## 前端对接流程

前端建议按以下顺序接入：

1. 页面初始化时请求 `GET /meta`，读取类别列表、上传限制和模型状态。
2. 用户选择图片后，先在前端做图片预览。
3. 上传前检查文件类型和大小，避免无效请求。
4. 请求 `POST /predict` 时展示 loading 状态并禁用重复提交。
5. `success: true` 时展示 `predicted_class`、`confidence` 和 `top_k`。
6. `success: false` 时展示 `error.message`。
7. 如果 `confidence` 较低，仍展示结果，但可以提示“结果仅供参考”。

## 联调测试场景

- 访问 `/health`，确认服务可用。
- 访问 `/meta`，确认类别顺序和 `class_names.json` 一致。
- 上传正常图片，展示分类结果。
- 上传非图片文件，展示错误提示。
- 上传超过 `10MB` 的文件，前端先拦截。
- 后端未启动时，前端展示连接失败。
- 模型未加载成功时，`/predict` 返回 `MODEL_NOT_READY`。
