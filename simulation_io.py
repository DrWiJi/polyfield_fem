# -*- coding: utf-8 -*-
"""
Shared simulation results I/O: same packed format as the network (zlib + pickle + base64 wire),
plain pickle (server disk save), and run-case pickle for CLI replay (--sim-file).
"""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

RESULTS_SCHEMA = "fe_sim_results_v1"
RUN_CASE_SCHEMA = "fe_sim_run_v1"


def prepare_material_library_rows(material_library: list | None, topology: dict | None) -> list | None:
    """
    Pad material rows so max(material_index) is covered (same as simulation server).
    """
    if not material_library or len(material_library) == 0:
        return None
    lib_rows = list(material_library)
    if topology and "material_index" in topology:
        try:
            import numpy as np

            mi = np.asarray(topology["material_index"], dtype=np.uint8)
            max_idx = int(mi.max()) if mi.size > 0 else -1
            default_row = [1380.0, 5.0e9, 3.5e9, 0.30, 1.0, 0.8, 2.626785107e6, 1.0]
            while len(lib_rows) <= max_idx:
                lib_rows.append(default_row)
        except Exception:
            pass
    return lib_rows


def pack_simulation_results(results: dict) -> dict:
    """Normalize results dict to storable / wire payload (numpy -> lists/floats)."""
    import numpy as np

    def _to_array(x):
        if x is None:
            return None
        a = np.asarray(x, dtype=np.float64)
        if a.ndim == 0:
            return float(a)
        return a

    return {
        "history_disp_center": _to_array(results.get("history_disp_center")),
        "history_disp_all": [_to_array(f) for f in (results.get("history_disp_all") or [])],
        "history_air_pressure_xy_center_z": [_to_array(f) for f in (results.get("history_air_pressure_xy_center_z") or [])],
        "history_air_pressure_step": int(results.get("history_air_pressure_step", 1)),
        "dt": float(results.get("dt", 1e-6)),
        "width_mm": float(results.get("width_mm", 0)),
        "height_mm": float(results.get("height_mm", 0)),
    }


def results_dict_to_wire_b64(packed: dict) -> str:
    """Same encoding as JSON message field data_b64 over the socket."""
    raw = pickle.dumps(packed, protocol=4)
    compressed = zlib.compress(raw, level=6)
    return base64.b64encode(compressed).decode("ascii")


def results_dict_from_wire_b64(data_b64: str) -> dict:
    """Decode network / .json file payload."""
    raw_b64 = base64.b64decode(data_b64)
    decompressed = zlib.decompress(raw_b64)
    return pickle.loads(decompressed)


def save_results_pickle(path: str | Path, results: dict) -> None:
    """Write packed results (same as server results/*.pkl)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = pack_simulation_results(results)
    with open(path, "wb") as f:
        pickle.dump(packed, f, protocol=4)


def save_results_wire_json(path: str | Path, results: dict) -> None:
    """Write JSON {schema, data_b64} — identical wire form to network."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = pack_simulation_results(results)
    payload = {
        "schema": RESULTS_SCHEMA,
        "data_b64": results_dict_to_wire_b64(packed),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_simulation_results_file(path: str | Path) -> dict:
    """
    Load results dict for UI / CLI. Accepts:
    - Pickle of packed dict (server sim_results_*.pkl)
    - JSON with schema + data_b64 (network export)
    - Legacy: JSON with top-level results keys (unlikely)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".json" or path.name.lower().endswith(".simresults.json"):
        text = path.read_text(encoding="utf-8")
        obj = json.loads(text)
        if isinstance(obj, dict) and "data_b64" in obj:
            return results_dict_from_wire_b64(obj["data_b64"])
        # Fallback: flat keys
        if isinstance(obj, dict) and "history_disp_center" in obj:
            return pack_simulation_results(obj)
        raise ValueError("JSON results file must contain data_b64 or history_disp_center")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "data_b64" in obj and obj.get("schema") == RESULTS_SCHEMA:
        return results_dict_from_wire_b64(obj["data_b64"])
    if isinstance(obj, dict) and "history_disp_center" in obj:
        return obj
    raise ValueError(f"Unrecognized pickle contents in {path}")


def pack_run_case(params: dict, material_library: list | None, topology: dict | None) -> dict:
    """Pickle-friendly run payload for CLI (--sim-file run mode)."""
    return {
        "schema": RUN_CASE_SCHEMA,
        "params": dict(params),
        "material_library": list(material_library or []),
        "topology": topology,
    }


def save_run_case_pickle(path: str | Path, params: dict, material_library: list | None, topology: dict | None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pack_run_case(params, material_library, topology), f, protocol=4)


def argv_from_ui_params(params: dict, *, no_plot: bool = True) -> list[str]:
    """
    Build argv list for diaphragm_opencl._parse_cli_args (same keys as UI / simulation_server).
    """
    argv = ["diaphragm_opencl.py"]
    if no_plot:
        argv.append("--no-plot")
    argv.extend(["--dt", str(params.get("dt", 1e-6))])
    argv.extend(["--duration", str(params.get("duration", 0.05))])
    argv.extend(["--force-shape", str(params.get("force_shape", "impulse"))])
    argv.extend(["--excitation-mode", str(params.get("excitation_mode", "external"))])
    argv.extend(["--force-amplitude", str(params.get("force_amplitude_pa", 10.0))])
    argv.extend(["--force-freq", str(params.get("force_freq_hz", 1000.0))])
    argv.extend(["--force-freq-end", str(params.get("force_freq_end", 5000.0))])
    if params.get("air_grid_step_mm") is not None:
        argv.extend(["--air-grid-step-mm", str(params["air_grid_step_mm"])])
    if params.get("air_pressure_history_every_steps") is not None:
        argv.extend(["--air-pressure-history-every-steps", str(params["air_pressure_history_every_steps"])])
    return argv


def load_run_case_pickle(path: str | Path) -> tuple[dict, list | None, dict | None]:
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or obj.get("schema") != RUN_CASE_SCHEMA:
        raise ValueError(f"Not a run-case file (expected schema {RUN_CASE_SCHEMA!r}): {path}")
    params = dict(obj.get("params") or {})
    topology = obj.get("topology")
    material_library = obj.get("material_library")
    if material_library is not None:
        material_library = list(material_library)
    return params, material_library, topology
