# -*- coding: utf-8 -*-
"""
Network server for OpenCL diaphragm simulation.
Controls diaphragm_opencl module, accepts connections from UI.
Run standalone: python simulation_server.py --server [--host 0.0.0.0] [--port 8765]
"""

from __future__ import annotations

import base64
import io
import json
import logging
import pickle
import socket
import struct
import sys
import threading
import zlib
from datetime import datetime
from pathlib import Path

from simulation_io import (
    argv_from_ui_params,
    pack_simulation_results,
    prepare_material_library_rows,
    results_dict_to_wire_b64,
    save_results_pickle,
)

RESULTS_DIR = Path("results")

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
SOCKET_TIMEOUT_S = 120  # Match client - long simulations, heartbeat keeps alive
MAX_PAYLOAD_BYTES = 50 * 1024 * 1024  # 50 MB — struct ">I" limit is 4 GB, but keep payloads manageable


def _decompress_run_data_b64(run_data_b64: str) -> tuple[dict, list | None, dict | None]:
    """Decode base64, decompress zlib, unpickle to (params, material_library, topology)."""
    raw_b64 = base64.b64decode(run_data_b64)
    decompressed = zlib.decompress(raw_b64)
    packed = pickle.loads(decompressed)
    params = packed.get("params", {})
    material_library = packed.get("material_library")
    topology = params.pop("topology", None)
    return params, material_library, topology


def _compress_results_b64(results: dict) -> str:
    """Serialize results to binary, compress with zlib, encode as base64 (network)."""
    return results_dict_to_wire_b64(pack_simulation_results(results))


def _save_results_to_file(results: dict, send_fn) -> Path | None:
    """Save full results to results/sim_results_YYYYMMDD_HHMMSS.pkl. Returns path or None."""
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RESULTS_DIR / f"sim_results_{ts}.pkl"
        save_results_pickle(path, results)
        send_fn({"type": "log", "text": f"[Server] Full results saved to {path.absolute()}\n"})
        return path
    except Exception as e:
        send_fn({"type": "log", "text": f"[Server] Failed to save results to file: {e}\n"})
        return None


def _send_message(sock: socket.socket, obj: dict) -> None:
    """Send JSON message: 4-byte length (big-endian) + UTF-8 payload."""
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    n = len(payload)
    if n > 4294967295:
        raise ValueError(f"Payload too large: {n} bytes (max 4 GB for protocol)")
    if n > MAX_PAYLOAD_BYTES:
        logger.warning("Payload %d bytes exceeds recommended %d MB; consider shorter simulation.",
                       n, MAX_PAYLOAD_BYTES // (1024 * 1024))
    sock.sendall(struct.pack(">I", n) + payload)


MAX_RECV_BYTES = 100 * 1024 * 1024  # 100 MB — run command with topology can be large


def _recv_message(sock: socket.socket) -> dict | None:
    """Receive JSON message. Returns None on EOF or error."""
    try:
        hdr = sock.recv(4)
        if len(hdr) < 4:
            return None
        length = struct.unpack(">I", hdr)[0]
        if length > MAX_RECV_BYTES:
            logger.warning("Message size %d MB exceeds limit %d MB; rejecting.", length // (1024 * 1024), MAX_RECV_BYTES // (1024 * 1024))
            return None
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(length - len(data), 65536))
            if not chunk:
                return None
            data += chunk
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, OSError, struct.error) as e:
        logger.debug("recv_message error: %s", e)
        return None


class _LogStream(io.TextIOBase):
    """Stream that forwards writes to send_fn for real-time log transmission."""

    def __init__(self, send_fn) -> None:
        self._send = send_fn
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer or "\r" in self._buffer:
            idx = min(
                self._buffer.find("\n") if "\n" in self._buffer else len(self._buffer),
                self._buffer.find("\r") if "\r" in self._buffer else len(self._buffer),
            )
            if idx == len(self._buffer):
                break
            line = self._buffer[: idx + 1]
            self._buffer = self._buffer[idx + 1 :]
            self._send({"type": "log", "text": line})
        return len(s)

    def flush(self) -> None:
        if self._buffer:
            self._send({"type": "log", "text": self._buffer})
            self._buffer = ""


def _run_simulation_worker(
    params: dict,
    material_library: list | None,
    topology: dict | None,
    send_fn,
    stop_flag: threading.Event,
) -> None:
    """Run simulation in worker, send log/results via send_fn."""
    import contextlib
    import traceback

    try:
        import diaphragm_opencl as cl_model
    except ImportError as e:
        send_fn({"type": "log", "text": f"[Server] Import error: {e}\n"})
        send_fn({"type": "status", "state": "error", "message": str(e)})
        return

    log_stream = _LogStream(send_fn)
    try:
        argv = argv_from_ui_params(params, no_plot=True)

        material_rows = prepare_material_library_rows(material_library, topology)

        send_fn({"type": "log", "text": "[Server] Parsing args and starting simulation...\n"})

        with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
            parsed = cl_model._parse_cli_args(argv)
            model, hist = cl_model.run_cli_simulation(
                parsed,
                stop_check=lambda: stop_flag.is_set(),
                topology=topology,
                material_library_rows=material_rows,
            )

        log_stream.flush()

        hist_center = list(hist) if hist is not None and len(hist) > 0 else (
            list(model.history_disp_center) if model.history_disp_center else []
        )
        hist_disp_all = list(model.history_disp_all) if getattr(model, "history_disp_all", None) else []

        def _decimate(frames: list, max_frames: int) -> list:
            """Decimate to max_frames by sampling evenly to keep payload under protocol limit."""
            if len(frames) <= max_frames:
                return frames
            step = max(1, len(frames) // max_frames)
            out = [frames[i] for i in range(0, len(frames), step)][:max_frames]
            send_fn({"type": "log", "text": f"[Server] Decimated {len(frames)} -> {len(out)} frames for transfer.\n"})
            return out

        def _decimate_1d(arr: list, max_points: int) -> list:
            """Decimate 1D array (e.g. history_disp_center) to max_points."""
            if len(arr) <= max_points:
                return arr
            step = max(1, len(arr) // max_points)
            out = [arr[i] for i in range(0, len(arr), step)][:max_points]
            send_fn({"type": "log", "text": f"[Server] Decimated center history {len(arr)} -> {len(out)} points.\n"})
            return out

        MAX_CENTER_POINTS = 500_000   # Enough for time/spectrum; relaxed from 200k
        MAX_DISP_FRAMES = 800     # Displacement maps; relaxed from 200
        hist_center_dec = _decimate_1d(hist_center, MAX_CENTER_POINTS)
        hist_disp_all_dec = _decimate(hist_disp_all, MAX_DISP_FRAMES)

        dt_val = float(parsed.dt)
        width_mm = model.width * 1e3 if hasattr(model, "width") else 0.0
        height_mm = model.height * 1e3 if hasattr(model, "height") else 0.0

        results_full = {
            "history_disp_center": hist_center,
            "history_disp_all": hist_disp_all,
            "dt": dt_val,
            "width_mm": width_mm,
            "height_mm": height_mm,
        }
        results = {
            "history_disp_center": hist_center_dec,
            "history_disp_all": hist_disp_all_dec,
            "dt": dt_val,
            "width_mm": width_mm,
            "height_mm": height_mm,
        }
        results_small = {
            "history_disp_center": hist_center_dec,
            "history_disp_all": [],
            "dt": dt_val,
            "width_mm": width_mm,
            "height_mm": height_mm,
        }
        try:
            data_b64 = _compress_results_b64(results)
            msg = {"type": "results", "data_b64": data_b64}
            payload_size = len(json.dumps(msg, ensure_ascii=False).encode("utf-8"))
            if payload_size > MAX_PAYLOAD_BYTES:
                send_fn({"type": "log", "text": f"[Server] Results {payload_size // (1024*1024)} MB exceed {MAX_PAYLOAD_BYTES // (1024*1024)} MB limit; saving full results to file, sending center-only.\n"})
                _save_results_to_file(results_full, send_fn)
                data_b64 = _compress_results_b64(results_small)
                msg = {"type": "results", "data_b64": data_b64}
            send_fn(msg)
        except (struct.error, ValueError) as e:
            send_fn({"type": "log", "text": f"[Server] Results too large, saving to file and sending center-only: {e}\n"})
            _save_results_to_file(results_full, send_fn)
            data_b64_small = _compress_results_b64(results_small)
            send_fn({"type": "results", "data_b64": data_b64_small})
        stopped = stop_flag.is_set()
        send_fn({
            "type": "status",
            "state": "stopped" if stopped else "finished",
            "message": "Stopped by user" if stopped else "OK",
        })

    except Exception as e:
        send_fn({"type": "log", "text": "\n[Server] Exception:\n" + traceback.format_exc()})
        send_fn({"type": "status", "state": "error", "message": str(e)})


def _handle_client(conn: socket.socket, addr) -> None:
    """Handle one client connection. One client at a time per server."""
    conn.settimeout(SOCKET_TIMEOUT_S)
    logger.info("Client connected from %s", addr)
    sim_thread: threading.Thread | None = None
    stop_flag = threading.Event()

    def send(obj: dict) -> None:
        try:
            _send_message(conn, obj)
        except OSError:
            pass

    try:
        while True:
            msg = _recv_message(conn)
            if msg is None:
                break
            msg_type = msg.get("type", "")
            if msg_type == "run":
                if sim_thread and sim_thread.is_alive():
                    send({"type": "log", "text": "[Server] Simulation already running.\n"})
                    continue
                send({"type": "log", "text": "[Server] Received run command, parsing data...\n"})
                if "run_data_b64" in msg:
                    try:
                        params, material_library, topology = _decompress_run_data_b64(msg["run_data_b64"])
                    except Exception as e:
                        logger.warning("Failed to decompress run data: %s", e)
                        send({"type": "log", "text": f"[Server] Decompress failed: {e}\n"})
                        send({"type": "status", "state": "error", "message": f"Decompress failed: {e}"})
                        continue
                else:
                    params = msg.get("params", {})
                    material_library = msg.get("material_library")
                    topology = params.pop("topology", None)
                stop_flag.clear()
                sim_thread = threading.Thread(
                    target=_run_simulation_worker,
                    args=(params, material_library, topology, send, stop_flag),
                    daemon=True,
                )
                sim_thread.start()
                send({"type": "status", "state": "running", "message": "Started"})
            elif msg_type == "stop":
                stop_flag.set()
                send({"type": "log", "text": "[Server] Stop requested (current run will complete).\n"})
                send({"type": "status", "state": "stopping", "message": "Stop requested"})
            elif msg_type == "ping":
                send({"type": "pong"})
            else:
                send({"type": "log", "text": f"[Server] Unknown message type: {msg_type}\n"})
    except OSError:
        pass
    finally:
        conn.close()
        logger.info("Client disconnected from %s", addr)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the simulation server. Blocks until interrupted."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        sock.listen(1)
        print(f"Simulation server listening on {host}:{port}")
        print("Connect UI with host={} port={}".format(host if host != "0.0.0.0" else "<this machine IP>", port))
        while True:
            conn, addr = sock.accept()
            _handle_client(conn, addr)
    finally:
        sock.close()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Diaphragm simulation server (OpenCL backend)")
    parser.add_argument("--server", action="store_true", help="Run as network server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default: {DEFAULT_HOST}, use 0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    args = parser.parse_args()
    if not args.server:
        parser.print_help()
        return 1
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
