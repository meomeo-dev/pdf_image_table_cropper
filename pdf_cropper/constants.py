IMAGE_BODY_CLASS_ID = 3
IMAGE_CAPTION_CLASS_ID = 4
TABLE_BODY_CLASS_ID = 5
TABLE_CAPTION_CLASS_ID = 6
CODE_BODY_CLASS_ID = 10
CODE_CAPTION_CLASS_ID = 11
ALGORITHM_CLASS_ID = 12

DEFAULT_OD_REPO = "opendatalab/PDF-Extract-Kit-1.0"
DEFAULT_OD_MODEL_RELATIVE_PATH = (
    "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
)

HERON_DEFAULT_MODEL = "docling-project/docling-layout-heron"
HERON_CODE_ID = 12
HERON_ADJACENT_IDS = frozenset({0, 7})
HERON_PICTURE_ID = 6
HERON_TABLE_ID = 8

DEFAULT_DAEMON_IDLE_SECONDS = 300
DEFAULT_DAEMON_START_TIMEOUT = 12.0
DEFAULT_DAEMON_RUN_TIMEOUT = 0.0

# caption/body family mapping used by CC merge.
CAPTION_FAMILY: dict[str, str] = {
    "image_caption": "image",
    "table_caption": "table",
    "code_caption": "code",
}
