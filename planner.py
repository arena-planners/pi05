"""pi0.5 base as a zero-shot navigation baseline, driven through its mobile-base velocity outputs."""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import pathlib
import socket
import subprocess
import time

import msgpack
import numpy as np
from arena_planners.sdk import Step, load_manifest, main_loop
from websockets.sync.client import connect

_DIR = pathlib.Path(__file__).parent
_MANIFEST = load_manifest(_DIR / "planner.yaml")
_SERVER = _DIR / "server" / "serve.py"
_FETCH_HINT = "run `arena feature planners add pi05`, then `server/setup.sh` in the planner dir"
_HOST_PYTHON_ENV = frozenset({"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"})

MODEL_DT = 0.02
ACTION_DIMS = 16
BASE_DIMS = slice(14, 16)
STATE_DIMS = 14
EMBODIMENTS = ("trossen_mobile", "arx_mobile")
BASE_CAMERA = "base_0_rgb"
WRIST_CAMERAS = ("left_wrist_0_rgb", "right_wrist_0_rgb")
_WRIST_SHAPE = (224, 224, 3)
_STOP_GRACE_S = 15.0

_log = logging.getLogger("pi05")


def to_wire(obj: object) -> object:
    """`obj` with ndarrays and numpy scalars as openpi-client's tagged maps, before msgpack_numpy's patch sees them."""
    if isinstance(obj, dict):
        return {key: to_wire(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_wire(value) for value in obj]
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": list(obj.shape)}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def from_wire(obj: dict) -> object:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj: object) -> bytes:
    return msgpack.packb(to_wire(obj), use_bin_type=True)


def unpackb(data: bytes) -> object:
    return msgpack.unpackb(data, object_hook=from_wire)


class PolicyClient:
    """Client of openpi's websocket policy server, which greets each connection with its metadata."""

    def __init__(self, uri: str, open_timeout_s: float = 10.0) -> None:
        self._stack = contextlib.ExitStack()
        self._ws = self._stack.enter_context(connect(uri, compression=None, max_size=None, open_timeout=open_timeout_s))
        self.metadata = unpackb(self._ws.recv())

    def infer(self, obs: dict) -> dict:
        self._ws.send(packb(obs))
        reply = self._ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(f"pi05 server failed:\n{reply}")
        return unpackb(reply)

    def close(self) -> None:
        self._stack.close()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_server(uri: str, proc: subprocess.Popen | None, timeout_s: float) -> PolicyClient:
    """Connect once the server listens, raising if `proc` exits or `timeout_s` passes first."""
    deadline = time.monotonic() + timeout_s
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"pi05 server exited with code {proc.returncode} before serving {uri}")
        try:
            return PolicyClient(uri)
        except OSError as exc:
            if time.monotonic() > deadline:
                raise TimeoutError(f"pi05 server not serving {uri} after {timeout_s:g} s") from exc
        time.sleep(1.0)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def state_vector(checkpoint: pathlib.Path, embodiment: str) -> np.ndarray:
    """The embodiment's mean proprioceptive state, from the checkpoint's norm stats."""
    path = checkpoint / "assets" / embodiment / "norm_stats.json"
    with open(path) as fh:
        mean = np.asarray(json.load(fh)["norm_stats"]["state"]["mean"], dtype=np.float32)
    if mean.shape != (STATE_DIMS,):
        raise ValueError(f"{path}: state mean has shape {mean.shape}, expected ({STATE_DIMS},)")
    return mean


def build_observation(image: np.ndarray, state: np.ndarray, prompt: str) -> dict:
    """pi0.5 model input: the robot camera as the base view, both wrist views blank and masked out."""
    rgb = np.ascontiguousarray(np.asarray(image)[..., :3], dtype=np.uint8)
    if rgb.ndim != 3:
        raise ValueError(f"pi05: camera frame has shape {rgb.shape}, expected (H, W, 3)")
    blank = np.zeros(_WRIST_SHAPE, dtype=np.uint8)
    return {
        "image": {BASE_CAMERA: rgb, **dict.fromkeys(WRIST_CAMERAS, blank)},
        "image_mask": {BASE_CAMERA: np.True_, **dict.fromkeys(WRIST_CAMERAS, np.False_)},
        "state": np.asarray(state, dtype=np.float32),
        "prompt": prompt,
    }


def base_velocities(actions: np.ndarray) -> np.ndarray:
    """`(horizon, 2)` base `[v, omega]` of an unnormalized `(horizon, 16)` action chunk."""
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIMS:
        raise ValueError(f"pi05: action chunk has shape {arr.shape}, expected (horizon, {ACTION_DIMS})")
    return arr[:, BASE_DIMS]


def resample_factor(chunk_dt: float) -> int:
    factor = round(chunk_dt / MODEL_DT)
    if factor < 1 or abs(factor * MODEL_DT - chunk_dt) > 1e-9:
        raise ValueError(f"chunk.dt {chunk_dt} is not a whole multiple of the model step {MODEL_DT}")
    return factor


def resample(base: np.ndarray, factor: int) -> np.ndarray:
    """Average each `factor` consecutive 50 Hz twists into one entry, dropping an incomplete tail."""
    entries = len(base) // factor
    return base[: entries * factor].reshape(entries, factor, base.shape[1]).mean(axis=1)


class Pi05:
    def __init__(self, client: PolicyClient, state: np.ndarray, factor: int) -> None:
        self.client = client
        self.state = state
        self.factor = factor
        self.instruction = ""
        self.last_infer_ms = 0.0

    def reset(self, initial_state: dict | None) -> None:
        self.instruction = str((initial_state or {}).get("instruction", "")).strip()

    def chunk(self, image: np.ndarray) -> np.ndarray:
        """`(entries, 2)` twists at the chunk period for one camera frame."""
        reply = self.client.infer(build_observation(image, self.state, self.instruction))
        self.last_infer_ms = float(reply["policy_timing"]["infer_ms"])
        return resample(base_velocities(reply["actions"]), self.factor)

    def step(self, features: dict) -> Step:
        image = features.get("camera")
        if not self.instruction or image is None or np.asarray(image).ndim != 3:
            return Step()
        return Step(chunk=self.chunk(np.asarray(image)).tolist())


def _resolve(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else _DIR / path


def server_paths(config: dict) -> dict[str, pathlib.Path]:
    """Checkpoint, server interpreter and tokenizer named by `config`, raising if one is missing."""
    paths = {
        "checkpoint": _resolve(str(config.get("checkpoint", "model/pi05_base_pytorch"))),
        "server_python": _resolve(str(config.get("server_python", "server/.venv/bin/python"))),
        "tokenizer": _resolve(str(config.get("tokenizer", "model/paligemma_tokenizer.model"))),
    }
    for what, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"pi05 {what} missing at {path}, {_FETCH_HINT}")
    return paths


def server_command(config: dict, paths: dict[str, pathlib.Path], port: int) -> tuple[list[str], dict[str, str]]:
    """Command line and environment of `server/serve.py` for `config`."""
    env = {key: value for key, value in os.environ.items() if key not in _HOST_PYTHON_ENV}
    env["OPENPI_DATA_HOME"] = str(_resolve(str(config.get("openpi_data", "model/openpi-data"))))
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if (paths["checkpoint"] / "model.safetensors").is_file():
        env["JAX_PLATFORMS"] = "cpu"
    cmd = [
        str(paths["server_python"]),
        str(_SERVER),
        f"--checkpoint={paths['checkpoint']}",
        f"--embodiment={config.get('embodiment', 'trossen_mobile')}",
        f"--tokenizer={paths['tokenizer']}",
        f"--port={port}",
        f"--device={config.get('device', 'auto')}",
        f"--flow-steps={int(config.get('flow_steps', 10))}",
        f"--compile-mode={config.get('compile_mode', '') or 'none'}",
    ]
    return cmd, env


def spawn_server(config: dict, paths: dict[str, pathlib.Path]) -> tuple[subprocess.Popen, PolicyClient]:
    """Start `server/serve.py` in the server venv, stopped at exit, and connect to it."""
    port = int(config.get("port", 0)) or free_port()
    cmd, env = server_command(config, paths, port)
    proc = subprocess.Popen(cmd, env=env)
    atexit.register(stop_server, proc)
    try:
        client = wait_for_server(f"ws://127.0.0.1:{port}", proc, float(config.get("startup_timeout_s", 900.0)))
    except BaseException:
        stop_server(proc)
        raise
    return proc, client


_planner: Pi05 | None = None
_server: subprocess.Popen | None = None


def on_init(config: dict) -> None:
    global _planner, _server
    embodiment = str(config.get("embodiment", "trossen_mobile"))
    if embodiment not in EMBODIMENTS:
        raise ValueError(f"pi05: embodiment {embodiment!r} not in {list(EMBODIMENTS)}")
    paths = server_paths(config)
    state = state_vector(paths["checkpoint"], embodiment)
    factor = resample_factor(float(_MANIFEST["chunk"]["dt"]))
    started = time.monotonic()
    _server, client = spawn_server(config, paths)
    _log.warning("pi05: server up after %.1f s, metadata %s", time.monotonic() - started, client.metadata)
    _planner = Pi05(client, state, factor)


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    assert _planner is not None
    _planner.reset(initial_state)


def step(features: dict) -> Step:
    assert _planner is not None
    return _planner.step(features)


if __name__ == "__main__":
    logging.basicConfig()
    main_loop(step, manifest=_MANIFEST, on_reset=on_reset, on_init=on_init)
