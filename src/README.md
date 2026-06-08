# src — 后端 API 服务

## 作用

`src` 是前端与模型之间的连接层，将训练好的 ConvNeXt-Tiny 模型封装成 HTTP API，供前端调用。

## 功能

- 接收前端上传的图片（`multipart/form-data`）
- 校验图片格式（jpg/jpeg/png/webp）和大小（≤10MB）
- 按模型要求进行预处理（Resize 384×384、归一化等）
- 加载 `convnext_tiny` 模型并执行推理
- 返回统一 JSON（主分类、置信度、Top-3 候选类别）
- 统一的错误码体系（`MISSING_FILE`、`INVALID_IMAGE`、`MODEL_NOT_READY` 等）
- CORS 跨域支持

## 文件结构

```
src/
├── README.md              # 本文件
├── requirements.txt       # Python 依赖
└── backend/
    ├── __init__.py        # 包标识
    └── server.py          # FastAPI 应用（含 Settings、分类器、路由）
```

## 启动方式

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt

# 2. 启动后端
uvicorn src.backend.server:app --host 0.0.0.0 --port 8000 --reload

# 3. 启动前端（另一个终端）
python3 -m http.server 3000 --directory frontend
```

## 模型信息

| 项目 | 值 |
|------|-----|
| 模型结构 | `convnext_tiny` |
| 输入尺寸 | 384 × 384 |
| 类别数 | 10 |
| 测试准确率 | 92.37% |
| 模型版本 | `convnext-tiny-v2` |

类别顺序以 `model/current/class_names.json` 为准：
`battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash`

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 检查服务状态和模型是否就绪 |
| GET | `/meta` | 获取类别列表、上传限制、模型信息 |
| POST | `/predict` | 上传图片，返回分类结果 |

### GET /health

```json
{"success": true, "status": "ok", "service": "garbage-classification-api", "model_version": "convnext-tiny-v2", "model": {"name": "convnext_tiny", "ready": true, "path": "...", "error": null}}
```

### GET /meta

```json
{"success": true, "classes": ["battery", ...], "upload": {"max_size_mb": 10, "supported_extensions": ["jpg","jpeg","png","webp"]}, "prediction": {"top_k": 3, "classification_type": "physical_material"}}
```

### POST /predict

请求：`Content-Type: multipart/form-data`，字段名 `file`

成功响应：
```json
{"success": true, "predicted_class": "plastic", "confidence": 0.93, "top_k": [...], "processing_ms": 87}
```

失败响应：
```json
{"success": false, "error": {"code": "INVALID_IMAGE", "message": "Uploaded file is not a readable image"}}
```

## 错误码

| code | HTTP | 场景 |
|------|------|------|
| MISSING_FILE | 400 | 未上传 file 字段 |
| UNSUPPORTED_FILE_TYPE | 400 | 格式不支持 |
| FILE_TOO_LARGE | 413 | 超过 10MB |
| INVALID_IMAGE | 400 | 无法识别为图片 |
| MODEL_NOT_READY | 503 | 模型未加载 |
| INFERENCE_FAILED | 500 | 推理失败 |
