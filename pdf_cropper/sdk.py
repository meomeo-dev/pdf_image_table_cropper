from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from .core import ModelRuntime, resolve_device, run_single_job
from .daemon import daemon_rpc, start_daemon_process, wait_for_daemon
from .models import CropJobConfig


def crop_pdf(config: CropJobConfig) -> dict[str, Any]:
    """Run one crop task via SDK and return metadata summary."""
    payload: dict[str, Any] = {
        "input_pdf": config.input_pdf,
        "storage_root": config.output_dir,
        "type": config.detect_type,
        "pages": config.pages,
        "dpi": config.dpi,
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "device": config.device,
        "enable_opendatalab": config.enable_opendatalab,
        "model_repo": config.model_repo,
        "model_file": config.model_file,
        "hf_cache_dir": config.hf_cache_dir,
        "hf_token": config.hf_token,
        "no_merge": config.no_merge,
        "heron_model": config.heron_model,
        "heron_conf": config.heron_conf,
        "metadata_file": config.metadata_file,
    }

    run_timeout: float | None = (
        None if config.daemon_run_timeout <= 0 else config.daemon_run_timeout
    )

    # SDK mode shares CLI behavior: daemon(on/auto/off) + direct fallback.
    if config.daemon_mode in ("auto", "on"):
        socket_path = config.daemon_socket
        if socket_path is None:
            raise ValueError("daemon_socket must be provided when daemon mode is set")
        try:
            reply = daemon_rpc(
                socket_path,
                payload={"action": "run", "job": payload},
                timeout=run_timeout,
            )
            if not reply.get("ok"):
                raise RuntimeError(reply.get("error", "daemon run failed"))
            return reply["result"]
        except Exception:
            # Keep daemon-mode semantics compatible with CLI.
            daemon_args = SimpleNamespace(
                daemon_socket=socket_path,
                daemon_idle_seconds=config.daemon_idle_seconds,
                device=resolve_device(config.device),
            )

            start_daemon_process(daemon_args)
            ready = wait_for_daemon(
                socket_path=socket_path,
                timeout=config.daemon_start_timeout,
            )
            if ready:
                reply = daemon_rpc(
                    socket_path,
                    payload={"action": "run", "job": payload},
                    timeout=run_timeout,
                )
                if reply.get("ok"):
                    return reply["result"]
                if config.daemon_mode == "on":
                    raise RuntimeError(reply.get("error", "daemon run failed"))
            elif config.daemon_mode == "on":
                raise RuntimeError("daemon mode is on, but daemon is not available")

    runtime = ModelRuntime(device=resolve_device(config.device))
    try:
        return run_single_job(payload, runtime)
    finally:
        runtime.cleanup()


def crop_pdf_simple(
    input_pdf: str,
    output_dir: str,
    detect_type: str = "both",
    pages: str = "all",
    dpi: int = 200,
    enable_opendatalab: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """A convenient SDK wrapper with the most common options."""
    config = CropJobConfig(
        input_pdf=input_pdf,
        output_dir=output_dir,
        detect_type=detect_type,
        pages=pages,
        dpi=dpi,
        enable_opendatalab=enable_opendatalab,
        device=device,
    )
    return crop_pdf(config)
