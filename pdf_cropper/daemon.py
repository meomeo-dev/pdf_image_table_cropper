from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any

from .core import ModelRuntime, run_single_job


def default_daemon_socket() -> str:
    uid = str(os.getuid()) if hasattr(os, "getuid") else "0"
    return f"/tmp/pdf_cropper_modeld_{uid}.sock"


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _send_json_line(conn: socket.socket, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(data)


def _recv_json_line(conn: socket.socket, max_bytes: int = 8 * 1024 * 1024):
    buffer = bytearray()
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise RuntimeError("daemon message too large")
        if b"\n" in chunk:
            break
    raw = bytes(buffer).split(b"\n", 1)[0].strip()
    if not raw:
        raise RuntimeError("empty daemon response")
    return json.loads(raw.decode("utf-8"))


def daemon_rpc(
    socket_path: str,
    payload: dict[str, Any],
    timeout: float | None,
) -> dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        if timeout is not None and timeout > 0:
            conn.settimeout(timeout)
        conn.connect(socket_path)
        _send_json_line(conn, payload)
        return _recv_json_line(conn)


def start_daemon_process(args) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pdf_cropper.cli",
        "--_daemon-worker",
        "--daemon-socket",
        args.daemon_socket,
        "--daemon-idle-seconds",
        str(args.daemon_idle_seconds),
        "--device",
        args.device,
    ]
    subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_for_daemon(socket_path: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(socket_path):
            try:
                daemon_rpc(
                    socket_path,
                    payload={"action": "ping"},
                    timeout=1.5,
                )
                return True
            except OSError:
                pass
        time.sleep(0.1)
    return False


def run_daemon_worker(args) -> int:
    _safe_unlink(args.daemon_socket)
    runtime = ModelRuntime(device=args.device)
    atexit.register(runtime.cleanup)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(args.daemon_socket)
        server.listen(8)
        server.settimeout(1.0)
        os.chmod(args.daemon_socket, 0o600)
        idle_deadline = time.time() + float(args.daemon_idle_seconds)

        while True:
            if time.time() >= idle_deadline:
                break
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            with conn:
                try:
                    request = _recv_json_line(conn)
                    action = request.get("action")
                    if action == "ping":
                        _send_json_line(conn, {"ok": True, "state": "ready"})
                        idle_deadline = time.time() + float(args.daemon_idle_seconds)
                        continue
                    if action == "shutdown":
                        _send_json_line(conn, {"ok": True, "state": "bye"})
                        break
                    if action != "run":
                        raise ValueError(f"unsupported daemon action: {action}")

                    result = run_single_job(request["job"], runtime)
                    result["path"] = "daemon"
                    _send_json_line(conn, {"ok": True, "result": result})
                    idle_deadline = time.time() + float(args.daemon_idle_seconds)
                except Exception as err:
                    _send_json_line(conn, {"ok": False, "error": str(err)})

    runtime.cleanup()
    _safe_unlink(args.daemon_socket)
    return 0
