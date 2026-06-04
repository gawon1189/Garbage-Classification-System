import json
from io import BytesIO
from pathlib import Path
from typing import Any

from .config import Settings


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
        except Exception as exc:  # pragma: no cover - depends on local env
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
        except Exception as exc:  # pragma: no cover - depends on local env
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
