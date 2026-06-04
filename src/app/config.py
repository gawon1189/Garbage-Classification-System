from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    model_path: Path
    class_names_path: Path
    service_name: str = "garbage-classification-api"
    model_name: str = "convnext_tiny"
    model_version: str = "convnext-tiny-v1"
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
    artifact_dir = project_root / "model" / "artifacts" / "artifacts"
    return Settings(
        project_root=project_root,
        model_path=artifact_dir / "best_model.pt",
        class_names_path=artifact_dir / "class_names.json",
    )
