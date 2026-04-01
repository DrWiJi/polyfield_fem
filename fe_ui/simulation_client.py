# -*- coding: utf-8 -*-
"""
Network client for simulation server.
Connects to simulation_server, sends run/stop, receives log/results.
"""

from __future__ import annotations

import base64
import json
import logging
import pickle
import socket
import struct
import threading
import uuid
import zlib
from typing import Callable

logger = logging.getLogger(__name__)

# Timeouts and heartbeat
SOCKET_TIMEOUT_S = 120  # 2 min - long simulations may not send data for a while
HEARTBEAT_INTERVAL_S = 15  # Send ping every 15s to keep connection alive
MAX_MSG_BYTES = 512 * 1024 * 1024  # 512 MB — allow larger project/results payloads
RUN_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MB chunks for large run-case transfer

try:
    from PySide6.QtCore import QObject, Signal
    HAS_QT = True
except ImportError:
    HAS_QT = False


def _send_message(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_MSG_BYTES:
        raise ValueError(
            f"Outgoing message too large: {len(payload)} bytes "
            f"(limit {MAX_MSG_BYTES} bytes)"
        )
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _compress_run_data_b64(params: dict, material_library: list | None) -> str:
    """Serialize run input to binary, compress with zlib, encode as base64.

    material_library: rows of 8 floats (…, acoustic_impedance, acoustic_inject)
    embedded in the payload (not a file path). Server passes them to run_cli_simulation directly.
    """
    packed = {"params": params, "material_library": material_library or []}
    raw = pickle.dumps(packed, protocol=4)
    compressed = zlib.compress(raw, level=6)
    return base64.b64encode(compressed).decode("ascii")


def _decompress_results_b64(data_b64: str) -> dict:
    """Decode base64, decompress zlib, unpickle to results dict (same as simulation_io)."""
    try:
        from simulation_io import results_dict_from_wire_b64

        return results_dict_from_wire_b64(data_b64)
    except ImportError:
        raw_b64 = base64.b64decode(data_b64)
        decompressed = zlib.decompress(raw_b64)
        return pickle.loads(decompressed)


def _recv_message(sock: socket.socket) -> dict | None:
    try:
        hdr = sock.recv(4)
        if len(hdr) < 4:
            return None
        length = struct.unpack(">I", hdr)[0]
        if length > MAX_MSG_BYTES:
            logger.warning("Message size %d bytes exceeds client limit %d MB; results discarded.", length, MAX_MSG_BYTES // (1024 * 1024))
            return None
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(length - len(data), 65536))
            if not chunk:
                return None
            data += chunk
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, OSError, struct.error):
        return None


class SimulationClient:
    """Client for simulation server. Thread-safe for connect/disconnect/send."""

    def __init__(
        self,
        on_log: Callable[[str], None] | None = None,
        on_status: Callable[[str, str], None] | None = None,
        on_results: Callable[[dict], None] | None = None,
        on_connected: Callable[[], None] | None = None,
        on_disconnected: Callable[[], None] | None = None,
    ) -> None:
        self._sock: socket.socket | None = None
        self._recv_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._on_log = on_log
        self._on_status = on_status
        self._on_results = on_results
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._stop_recv = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def connect(self, host: str, port: int) -> bool:
        """Connect to server. Returns True on success."""
        with self._lock:
            if self._sock is not None:
                self.disconnect()
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(10.0)
                self._sock.connect((host, port))
                self._sock.settimeout(SOCKET_TIMEOUT_S)
                self._stop_recv.clear()
                self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
                self._recv_thread.start()
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    daemon=True,
                )
                self._heartbeat_thread.start()
                if self._on_connected:
                    self._on_connected()
                return True
            except OSError as e:
                logger.warning("Connect failed: %s", e)
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                return False

    def disconnect(self) -> None:
        """Disconnect from server."""
        with self._lock:
            self._stop_recv.set()
            sock = self._sock
            self._sock = None
            self._recv_thread = None
            self._heartbeat_thread = None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
            except Exception:
                pass
            if self._on_disconnected:
                self._on_disconnected()

    def is_connected(self) -> bool:
        with self._lock:
            return self._sock is not None

    def run_simulation(self, params: dict, material_library: list | None = None) -> bool:
        """Send run command. Returns True if sent successfully."""
        with self._lock:
            if self._sock is None:
                return False
            try:
                run_data_b64 = _compress_run_data_b64(params, material_library)
                msg = {"type": "run", "run_data_b64": run_data_b64}
                payload_size = len(json.dumps(msg, ensure_ascii=False).encode("utf-8"))
                if payload_size <= RUN_CHUNK_BYTES:
                    _send_message(self._sock, msg)
                    return True
                transfer_id = uuid.uuid4().hex
                total = (len(run_data_b64) + RUN_CHUNK_BYTES - 1) // RUN_CHUNK_BYTES
                _send_message(
                    self._sock,
                    {
                        "type": "run_begin",
                        "transfer_id": transfer_id,
                        "total_chunks": total,
                        "total_b64_len": len(run_data_b64),
                    },
                )
                for idx in range(total):
                    beg = idx * RUN_CHUNK_BYTES
                    end = min((idx + 1) * RUN_CHUNK_BYTES, len(run_data_b64))
                    _send_message(
                        self._sock,
                        {
                            "type": "run_chunk",
                            "transfer_id": transfer_id,
                            "index": idx,
                            "chunk_b64": run_data_b64[beg:end],
                        },
                    )
                _send_message(self._sock, {"type": "run_commit", "transfer_id": transfer_id})
                return True
            except (OSError, ValueError) as e:
                logger.warning("Failed to send run command: %s", e)
                return False

    def stop_simulation(self) -> bool:
        """Send stop command."""
        with self._lock:
            if self._sock is None:
                return False
            try:
                _send_message(self._sock, {"type": "stop"})
                return True
            except (OSError, ValueError):
                return False

    def _heartbeat_loop(self) -> None:
        """Send ping periodically to keep connection alive."""
        import time
        while not self._stop_recv.wait(timeout=HEARTBEAT_INTERVAL_S):
            with self._lock:
                sock = self._sock
            if sock is None:
                break
            try:
                _send_message(sock, {"type": "ping"})
            except (OSError, ValueError):
                break

    def _recv_loop(self) -> None:
        sock = self._sock
        while sock and not self._stop_recv.is_set():
            msg = _recv_message(sock)
            if msg is None:
                break
            msg_type = msg.get("type", "")
            if msg_type == "log" and self._on_log:
                self._on_log(msg.get("text", ""))
            elif msg_type == "status" and self._on_status:
                self._on_status(msg.get("state", ""), msg.get("message", ""))
            elif msg_type == "results" and self._on_results:
                if "data_b64" in msg:
                    try:
                        data = _decompress_results_b64(msg["data_b64"])
                    except Exception as e:
                        logger.warning("Failed to decompress results: %s", e)
                        data = {}
                else:
                    data = msg.get("data", {})
                self._on_results(data)
            elif msg_type == "pong":
                pass  # Heartbeat response, keep connection alive
        self.disconnect()


if HAS_QT:

    class SimulationClientBridge(QObject):
        """Qt-friendly wrapper: emits signals for main-thread handling."""

        log_received = Signal(str)
        status_received = Signal(str, str)  # state, message
        results_received = Signal(object)   # dict
        connected_changed = Signal(bool)

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self._client = SimulationClient(
                on_log=lambda t: self.log_received.emit(t),
                on_status=lambda s, m: self.status_received.emit(s, m),
                on_results=lambda d: self.results_received.emit(d),
                on_connected=lambda: self.connected_changed.emit(True),
                on_disconnected=lambda: self.connected_changed.emit(False),
            )

        def connect_to_server(self, host: str, port: int) -> bool:
            return self._client.connect(host, port)

        def disconnect_from_server(self) -> None:
            self._client.disconnect()

        def is_connected(self) -> bool:
            return self._client.is_connected()

        def run_simulation(self, params: dict, material_library: list | None = None) -> bool:
            return self._client.run_simulation(params, material_library)

        def stop_simulation(self) -> bool:
            return self._client.stop_simulation()
