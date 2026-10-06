from __future__ import annotations

import subprocess
import time

import numpy as np
import pytest

import planner

_INSTRUCTIONS = (
    "go to the door",
    "move forward",
    "turn left",
    "walk down the hallway and stop at the end",
)
_NEAR_ZERO = 0.01


def _corridor(door_x: int, seed: int) -> np.ndarray:
    """A 640x480 corridor view: floor, side walls, ceiling and a door at the far end."""
    h, w, horizon = 480, 640, 230
    rows, cols = np.mgrid[0:h, 0:w]
    depth = np.clip((rows - horizon) / (h - horizon), 0.0, 1.0)
    image = np.full((h, w, 3), (205, 200, 190), dtype=np.float32)
    image[(rows > horizon) & (np.abs(cols - door_x) < 40 + depth * 400)] = (120, 100, 80)
    image[rows < horizon - 150] = (235, 235, 235)
    image[(rows > horizon - 90) & (rows < horizon + 20) & (np.abs(cols - door_x) < 22)] = (90, 55, 30)
    image += np.random.default_rng(seed).normal(0.0, 6.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _gpu_mib(pid: int) -> int | None:
    query = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(query, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        app_pid, used = (field.strip() for field in line.split(","))
        if int(app_pid) == pid:
            return int(used)
    return None


@pytest.fixture(scope="module")
def pi05() -> planner.Pi05:
    config = dict(planner._MANIFEST["config"])
    try:
        planner.server_paths(config)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    started = time.monotonic()
    planner.on_init(config)
    print(f"\npi05: on_init (server spawn and checkpoint load) {time.monotonic() - started:.1f} s")
    assert planner._planner is not None and planner._server is not None
    yield planner._planner
    planner.stop_server(planner._server)


def test_instruction_chunks_on_real_weights(pi05: planner.Pi05):
    frames = [_corridor(320, 0), _corridor(180, 1), _corridor(470, 2)]
    planner.on_reset("warmup", {"instruction": _INSTRUCTIONS[0]})
    started = time.monotonic()
    planner.step({"camera": frames[0], "robot_pose": np.zeros(3)})
    print(f"pi05: first step {time.monotonic() - started:.2f} s")

    chunks, raw, wall_ms, infer_ms = [], [], [], []
    for instruction in _INSTRUCTIONS:
        planner.on_reset(instruction, {"instruction": instruction})
        for frame in frames:
            started = time.monotonic()
            step = planner.step({"camera": frame, "robot_pose": np.zeros(3)})
            wall_ms.append(1000.0 * (time.monotonic() - started))
            infer_ms.append(pi05.last_infer_ms)
            chunk = np.asarray(step.chunk)
            assert chunk.shape == (10, 2)
            assert np.isfinite(chunk).all()
            chunks.append(chunk)
            obs = planner.build_observation(frame, pi05.state, instruction)
            raw.append(planner.base_velocities(pi05.client.infer(obs)["actions"]))

    base = np.concatenate(raw)
    assert base.shape == (len(_INSTRUCTIONS) * len(frames) * 50, 2)
    v, w = base[:, 0], base[:, 1]
    print(f"pi05: 50 Hz base actions over {len(raw)} chunks")
    print(f"  v range [{v.min():+.4f}, {v.max():+.4f}] mean {v.mean():+.4f} mean|v| {np.abs(v).mean():.4f} m/s")
    print(f"  w range [{w.min():+.4f}, {w.max():+.4f}] mean {w.mean():+.4f} mean|w| {np.abs(w).mean():.4f} rad/s")
    still_v, still_w = np.mean(np.abs(v) < _NEAR_ZERO), np.mean(np.abs(w) < _NEAR_ZERO)
    print(f"  |v| < {_NEAR_ZERO}: {still_v:.1%}, |w| < {_NEAR_ZERO}: {still_w:.1%}")
    for instruction, chunk in zip(np.repeat(_INSTRUCTIONS, len(frames)), chunks, strict=True):
        print(f"  {instruction!r}: first 0.5 s mean v {chunk[:5, 0].mean():+.4f} w {chunk[:5, 1].mean():+.4f}")
    print(f"pi05: step wall {np.median(wall_ms):.0f} ms median ({min(wall_ms):.0f}-{max(wall_ms):.0f})")
    print(f"pi05: server infer {np.median(infer_ms):.0f} ms median")
    print(f"pi05: server GPU memory {_gpu_mib(planner._server.pid)} MiB")
