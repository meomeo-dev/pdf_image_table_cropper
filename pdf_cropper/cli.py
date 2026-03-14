from __future__ import annotations

import argparse
import json
from typing import Any

from .constants import (
    DEFAULT_DAEMON_IDLE_SECONDS,
    DEFAULT_DAEMON_RUN_TIMEOUT,
    DEFAULT_DAEMON_START_TIMEOUT,
    DEFAULT_OD_MODEL_RELATIVE_PATH,
    DEFAULT_OD_REPO,
    HERON_DEFAULT_MODEL,
)
from .core import positive_int, resolve_device
from .daemon import default_daemon_socket, run_daemon_worker
from .models import CropJobConfig
from .sdk import crop_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-image-table-cropper",
        description=(
            "Crop table/image regions from a PDF and export crop files + "
            "position metadata. No OCR or text extraction is performed."
        ),
    )
    parser.add_argument("-i", "--input-pdf", required=True, help="input PDF file path")
    parser.add_argument(
        "-o",
        "--output-dir",
        "--storage-root",
        dest="storage_root",
        required=True,
        help="storage root; crops are written to <output-dir>/<pdf-stem>/",
    )
    parser.add_argument(
        "--type",
        choices=["both", "image", "table", "code", "algorithm"],
        default="both",
        help="which region types to export (default: both)",
    )
    parser.add_argument(
        "--pages",
        default="all",
        help="page selection, e.g. 'all', '1', '1-3', or '1-3,7,10'",
    )
    parser.add_argument(
        "--dpi",
        type=positive_int,
        default=200,
        help="render DPI (default: 200)",
    )
    parser.add_argument(
        "--imgsz",
        type=positive_int,
        default=1280,
        help="OpenDataLab supplementary model input size (default: 1280)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="OpenDataLab supplementary confidence threshold (default: 0.10)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="OpenDataLab supplementary IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=("inference device: cpu | mps | cuda | cuda:0 ... " "(default: auto)"),
    )
    parser.add_argument(
        "--enable-opendatalab",
        action="store_true",
        default=False,
        help="enable OpenDataLab YOLO as supplementary detector",
    )
    parser.add_argument(
        "--model-repo",
        default=DEFAULT_OD_REPO,
        help=f"OpenDataLab supplementary model repo (default: {DEFAULT_OD_REPO})",
    )
    parser.add_argument(
        "--model-file",
        default=DEFAULT_OD_MODEL_RELATIVE_PATH,
        help="OpenDataLab supplementary model file inside snapshot",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="optional HuggingFace cache directory",
    )
    parser.add_argument("--hf-token", default=None, help="optional HuggingFace token")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        default=False,
        help="disable Connected-Components region merging",
    )
    parser.add_argument(
        "--heron-model",
        default=HERON_DEFAULT_MODEL,
        metavar="MODEL_ID",
        help=(
            "docling Heron RT-DETR model id/path used as primary detector "
            f"(default: {HERON_DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--heron-conf",
        type=float,
        default=0.5,
        help="confidence threshold for the heron model (default: 0.5)",
    )
    parser.add_argument(
        "--metadata-file",
        default="metadata.json",
        help="metadata file name under output dir (default: metadata.json)",
    )
    parser.add_argument(
        "--daemon-mode",
        choices=["off", "auto", "on"],
        default="off",
        help=(
            "model daemon mode: off=direct run, auto=try daemon then fallback, "
            "on=daemon required"
        ),
    )
    parser.add_argument(
        "--daemon-socket",
        default=default_daemon_socket(),
        help="unix socket path for local daemon",
    )
    parser.add_argument(
        "--daemon-idle-seconds",
        type=positive_int,
        default=DEFAULT_DAEMON_IDLE_SECONDS,
        help=(
            "daemon idle timeout in seconds before auto-exit "
            f"(default: {DEFAULT_DAEMON_IDLE_SECONDS})"
        ),
    )
    parser.add_argument(
        "--daemon-start-timeout",
        type=float,
        default=DEFAULT_DAEMON_START_TIMEOUT,
        help=(
            "max seconds waiting for daemon startup "
            f"(default: {DEFAULT_DAEMON_START_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--daemon-run-timeout",
        type=float,
        default=DEFAULT_DAEMON_RUN_TIMEOUT,
        help="max seconds waiting for daemon job response; 0 means no timeout",
    )
    parser.add_argument(
        "--_daemon-worker",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser


def _build_config(args: argparse.Namespace) -> CropJobConfig:
    return CropJobConfig(
        input_pdf=str(args.input_pdf),
        output_dir=str(args.storage_root),
        detect_type=args.type,
        pages=args.pages,
        dpi=args.dpi,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        enable_opendatalab=args.enable_opendatalab,
        model_repo=args.model_repo,
        model_file=args.model_file,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=args.hf_token,
        no_merge=args.no_merge,
        heron_model=args.heron_model,
        heron_conf=args.heron_conf,
        metadata_file=args.metadata_file,
        daemon_mode=args.daemon_mode,
        daemon_socket=args.daemon_socket,
        daemon_idle_seconds=args.daemon_idle_seconds,
        daemon_start_timeout=args.daemon_start_timeout,
        daemon_run_timeout=args.daemon_run_timeout,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    if args._daemon_worker:
        return run_daemon_worker(args)

    if not args.input_pdf:
        parser.error("-i/--input-pdf is required")
    if not args.storage_root:
        parser.error("-o/--output-dir is required")

    config = _build_config(args)
    result: dict[str, Any] | None = None
    try:
        result = crop_pdf(config)
    except Exception as err:
        parser.error(str(err))

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
