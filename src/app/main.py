import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .inference import ImageClassifier, InvalidImageError, ModelLoadError


settings = get_settings()
classifier = ImageClassifier(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        classifier.load()
    except ModelLoadError:
        # The API should still start so /health can explain why prediction is unavailable.
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
