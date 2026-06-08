import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ── Settings ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    project_root: Path
    model_path: Path
    class_names_path: Path
    service_name: str = "garbage-classification-api"
    model_name: str = "convnext_tiny"
    model_version: str = "convnext-tiny-v2"
    image_size: int = 384
    top_k: int = 3
    max_file_size_mb: int = 10
    upload_field_name: str = "file"
    supported_extensions: tuple[str, ...] = ("jpg", "jpeg", "png", "webp")
    supported_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )
    default_class_names: tuple[str, ...] = (
        "battery",
        "biological",
        "cardboard",
        "clothes",
        "glass",
        "metal",
        "paper",
        "plastic",
        "shoes",
        "trash",
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    artifact_dir = project_root / "model" / "current"
    return Settings(
        project_root=project_root,
        model_path=artifact_dir / "model.pt",
        class_names_path=artifact_dir / "class_names.json",
    )


# ── Inference ──────────────────────────────────────────────────────────

class ModelLoadError(RuntimeError):
    pass


class InvalidImageError(ValueError):
    pass


class ImageClassifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.load_error: str | None = None
        try:
            self.class_names = self._load_class_names(settings.class_names_path)
        except ModelLoadError as exc:
            self.class_names = list(settings.default_class_names)
            self.load_error = str(exc)
        self.device = None
        self.model = None
        self.preprocess = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.preprocess is not None

    def load(self) -> None:
        if self.is_ready:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
        except Exception as exc:
            self.load_error = f"Missing model dependency: {exc}"
            raise ModelLoadError(self.load_error) from exc

        if not self.settings.model_path.exists():
            self.load_error = f"Model file not found: {self.settings.model_path}"
            raise ModelLoadError(self.load_error)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(self.class_names))

        checkpoint = torch.load(self.settings.model_path, map_location=self.device)
        if hasattr(checkpoint, "eval"):
            model = checkpoint
        else:
            state_dict = self._extract_state_dict(checkpoint)
            state_dict = self._strip_module_prefix(state_dict)
            model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.settings.image_size, self.settings.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.load_error = None

    def predict(self, image_bytes: bytes) -> list[dict[str, float | str]]:
        if not self.is_ready:
            self.load()

        try:
            import torch
            from PIL import Image, UnidentifiedImageError
        except Exception as exc:
            raise ModelLoadError(f"Missing inference dependency: {exc}") from exc

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("Uploaded file is not a readable image") from exc

        assert self.model is not None
        assert self.preprocess is not None
        assert self.device is not None

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            scores, indices = torch.topk(
                probabilities,
                k=min(self.settings.top_k, len(self.class_names)),
            )

        return [
            {
                "class": self.class_names[int(index.item())],
                "score": float(score.item()),
            }
            for score, index in zip(scores, indices, strict=False)
        ]

    @staticmethod
    def _load_class_names(path: Path) -> list[str]:
        if not path.exists():
            raise ModelLoadError(f"Class names file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            class_names = json.load(file)
        if not isinstance(class_names, list) or not all(isinstance(item, str) for item in class_names):
            raise ModelLoadError("class_names.json must be a JSON array of strings")
        return class_names

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
        if not isinstance(checkpoint, dict):
            raise ModelLoadError("Unsupported checkpoint format: expected a dict/state_dict")

        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        tensor_like_values = [value for value in checkpoint.values() if hasattr(value, "shape")]
        if tensor_like_values:
            return checkpoint

        raise ModelLoadError("Unsupported checkpoint format: no state_dict found")

    @staticmethod
    def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
        return {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }


# ── App ────────────────────────────────────────────────────────────────

settings = get_settings()
classifier = ImageClassifier(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        classifier.load()
    except ModelLoadError:
        pass
    yield


app = FastAPI(
    title="Garbage Classification API",
    version=settings.model_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "success": True,
        "status": "ok",
        "service": settings.service_name,
        "model_version": settings.model_version,
        "model": {
            "name": settings.model_name,
            "ready": classifier.is_ready,
            "path": str(settings.model_path),
            "error": classifier.load_error,
        },
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc) -> JSONResponse:
    request_id = _request_id()
    for error in exc.errors():
        if "file" in error.get("loc", ()):
            return _error_response(
                request_id,
                "MISSING_FILE",
                "Please upload an image using the file field",
                status_code=400,
            )
    return _error_response(request_id, "INVALID_REQUEST", "Invalid request", status_code=400)


@app.get("/meta")
def meta() -> dict[str, object]:
    return {
        "success": True,
        "model_version": settings.model_version,
        "model": {
            "name": settings.model_name,
            "image_size": settings.image_size,
            "ready": classifier.is_ready,
            "error": classifier.load_error,
        },
        "classes": classifier.class_names,
        "upload": {
            "field_name": settings.upload_field_name,
            "content_type": "multipart/form-data",
            "max_size_mb": settings.max_file_size_mb,
            "supported_extensions": list(settings.supported_extensions),
            "supported_content_types": list(settings.supported_content_types),
        },
        "prediction": {
            "top_k": settings.top_k,
            "classification_type": "physical_material",
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    request_id = _request_id()
    started_at = time.perf_counter()

    validation_error = await _validate_upload(file)
    if validation_error is not None:
        return _error_response(request_id, validation_error[0], validation_error[1], status_code=400)

    image_bytes = await file.read()
    if len(image_bytes) > settings.max_file_size_bytes:
        return _error_response(
            request_id,
            "FILE_TOO_LARGE",
            f"Uploaded file must be {settings.max_file_size_mb}MB or smaller",
            status_code=413,
        )

    try:
        top_k = classifier.predict(image_bytes)
    except InvalidImageError:
        return _error_response(
            request_id,
            "INVALID_IMAGE",
            "Uploaded file is not a readable image",
            status_code=400,
        )
    except ModelLoadError as exc:
        return _error_response(request_id, "MODEL_NOT_READY", str(exc), status_code=503)
    except Exception as exc:
        return _error_response(request_id, "INFERENCE_FAILED", str(exc), status_code=500)

    processing_ms = round((time.perf_counter() - started_at) * 1000)
    predicted = top_k[0]
    return JSONResponse(
        {
            "success": True,
            "request_id": request_id,
            "predicted_class": predicted["class"],
            "confidence": predicted["score"],
            "top_k": top_k,
            "model_version": settings.model_version,
            "processing_ms": processing_ms,
        }
    )


async def _validate_upload(file: UploadFile | None) -> tuple[str, str] | None:
    if file is None:
        return "MISSING_FILE", "Please upload an image using the file field"

    filename = file.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in settings.supported_extensions:
        return (
            "UNSUPPORTED_FILE_TYPE",
            "Only jpg, jpeg, png, and webp images are supported",
        )

    if file.content_type and file.content_type not in settings.supported_content_types:
        return (
            "UNSUPPORTED_FILE_TYPE",
            "Only jpg, jpeg, png, and webp images are supported",
        )

    return None


def _error_response(
    request_id: str,
    code: str,
    message: str,
    status_code: int = 400,
) -> JSONResponse:
    return JSONResponse(
        {
            "success": False,
            "request_id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        },
        status_code=status_code,
    )


def _request_id() -> str:
    return uuid.uuid4().hex[:12]
