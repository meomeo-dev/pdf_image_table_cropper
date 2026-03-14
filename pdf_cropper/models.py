from __future__ import annotations

from dataclasses import dataclass, field

from .constants import (
    DEFAULT_DAEMON_IDLE_SECONDS,
    DEFAULT_DAEMON_RUN_TIMEOUT,
    DEFAULT_DAEMON_START_TIMEOUT,
    DEFAULT_OD_MODEL_RELATIVE_PATH,
    DEFAULT_OD_REPO,
    HERON_DEFAULT_MODEL,
)


@dataclass
class CropRecord:
    content_type: str
    page_index: int
    page_number: int
    page_size_pdf: list[float]
    bbox_pdf: list[float]
    bbox_pixels: list[int]
    score: float
    image_path: str
    merged_from: list[list[int]] = field(default_factory=list)


@dataclass
class CropJobConfig:
    """SDK job configuration for one PDF crop task."""

    input_pdf: str
    output_dir: str
    detect_type: str = "both"
    pages: str = "all"
    dpi: int = 200
    imgsz: int = 1280
    conf: float = 0.10
    iou: float = 0.45
    device: str | None = None
    enable_opendatalab: bool = False
    model_repo: str = DEFAULT_OD_REPO
    model_file: str = DEFAULT_OD_MODEL_RELATIVE_PATH
    hf_cache_dir: str | None = None
    hf_token: str | None = None
    no_merge: bool = False
    heron_model: str = HERON_DEFAULT_MODEL
    heron_conf: float = 0.5
    metadata_file: str = "metadata.json"
    daemon_mode: str = "off"
    daemon_socket: str | None = None
    daemon_idle_seconds: int = DEFAULT_DAEMON_IDLE_SECONDS
    daemon_start_timeout: float = DEFAULT_DAEMON_START_TIMEOUT
    daemon_run_timeout: float = DEFAULT_DAEMON_RUN_TIMEOUT
