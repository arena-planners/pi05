from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from arena_planners.sdk import Step

import planner

_HORIZON = 50

_SERVER = """
import functools, json, sys

import msgpack
import numpy as np
from websockets.sync.server import serve


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def describe(obj):
    if isinstance(obj, dict):
        return {str(k): describe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        values = obj.ravel().tolist()[:16]
        total = float(obj.astype(np.float64).sum())
        return {"type": "ndarray", "dtype": obj.dtype.str, "shape": list(obj.shape), "values": values, "sum": total}
    if isinstance(obj, np.generic):
        return {"type": type(obj).__name__, "value": obj.item()}
    return {"type": type(obj).__name__, "value": obj}


actions = np.zeros((50, 16), dtype=np.float32)
actions[:, :14] = 7.0
actions[:, 14] = np.arange(50, dtype=np.float32) / 100.0
actions[:, 15] = -np.arange(50, dtype=np.float32) / 200.0
port, log = int(sys.argv[1]), sys.argv[2]


def handler(ws):
    packer = Packer()
    ws.send(packer.pack({"embodiment": "trossen_mobile", "action_dims": 16}))
    for message in ws:
        obs = unpackb(message)
        with open(log, "a") as fh:
            fh.write(json.dumps(describe(obs)) + "\\n")
        if obs.get("prompt") == "raise":
            ws.send("Traceback (most recent call last):\\nValueError: boom")
            ws.close()
            return
        reply = {"actions": actions, "state": np.zeros(32, np.float32), "policy_timing": {"infer_ms": 12.5}}
        ws.send(packer.pack(reply))


with serve(handler, "127.0.0.1", port, compression=None, max_size=None) as server:
    server.serve_forever()
"""


def _actions() -> np.ndarray:
    actions = np.zeros((_HORIZON, planner.ACTION_DIMS), dtype=np.float32)
    actions[:, :14] = 7.0
    actions[:, 14] = np.arange(_HORIZON, dtype=np.float32) / 100.0
    actions[:, 15] = -np.arange(_HORIZON, dtype=np.float32) / 200.0
    return actions


class OpenpiServer:
    """openpi's wire format served by a separate interpreter that never imports arena_planners or msgpack_numpy."""

    def __init__(self, workdir: pathlib.Path) -> None:
        script = workdir / "openpi_server.py"
        script.write_text(_SERVER)
        self.log = workdir / "received.jsonl"
        self.log.touch()
        port = planner.free_port()
        self.uri = f"ws://127.0.0.1:{port}"
        self.proc = subprocess.Popen([sys.executable, str(script), str(port), str(self.log)])
        planner.wait_for_server(self.uri, self.proc, 30.0).close()

    def received(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def clear(self) -> None:
        self.log.write_text("")


@pytest.fixture(scope="module")
def openpi_server(tmp_path_factory: pytest.TempPathFactory) -> OpenpiServer:
    srv = OpenpiServer(tmp_path_factory.mktemp("openpi"))
    yield srv
    planner.stop_server(srv.proc)


@pytest.fixture
def server(openpi_server: OpenpiServer) -> OpenpiServer:
    openpi_server.clear()
    return openpi_server


@pytest.fixture
def client(server: OpenpiServer) -> planner.PolicyClient:
    cl = planner.PolicyClient(server.uri)
    yield cl
    cl.close()


@pytest.fixture
def frame() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)


def test_wire_maps_are_openpi_tags():
    wire = planner.to_wire({"a": np.arange(6, dtype=np.uint8).reshape(2, 3), "m": np.True_, "p": "go", "l": (1.0,)})
    assert wire["a"] == {b"__ndarray__": True, b"data": bytes(range(6)), b"dtype": "|u1", b"shape": [2, 3]}
    assert wire["m"] == {b"__npgeneric__": True, b"data": True, b"dtype": "|b1"}
    assert wire["p"] == "go"
    assert wire["l"] == [1.0]


def test_unpack_restores_arrays_and_scalars():
    obs = {"state": np.linspace(-1.0, 1.0, 14, dtype=np.float32), "mask": np.False_, "prompt": "go"}
    out = planner.unpackb(planner.packb(obs))
    np.testing.assert_array_equal(out["state"], obs["state"])
    assert out["state"].dtype == np.float32
    assert out["mask"] is np.False_
    assert out["prompt"] == "go"


def test_object_arrays_are_refused():
    with pytest.raises(ValueError, match="unsupported dtype"):
        planner.packb({"x": np.array([None])})


def test_client_reads_metadata_then_infers(server: OpenpiServer, client: planner.PolicyClient):
    assert client.metadata == {"embodiment": "trossen_mobile", "action_dims": 16}
    reply = client.infer({"prompt": "go", "state": np.arange(14, dtype=np.float32), "flag": np.True_})
    np.testing.assert_array_equal(reply["actions"], _actions())
    assert reply["actions"].dtype == np.float32
    assert reply["policy_timing"]["infer_ms"] == 12.5
    (sent,) = server.received()
    assert sent["state"] == {"type": "ndarray", "dtype": "<f4", "shape": [14], "values": list(range(14)), "sum": 91.0}
    assert sent["flag"] == {"type": "bool", "value": True}
    assert sent["prompt"] == {"type": "str", "value": "go"}


def test_server_traceback_raises(client: planner.PolicyClient):
    with pytest.raises(RuntimeError, match="ValueError: boom"):
        client.infer({"prompt": "raise"})


def test_wait_for_server_connects(server: OpenpiServer):
    planner.wait_for_server(server.uri, None, 5.0).close()


def test_wait_for_server_times_out_on_a_closed_port():
    with pytest.raises(TimeoutError, match="not serving"):
        planner.wait_for_server(f"ws://127.0.0.1:{planner.free_port()}", None, 0.0)


def test_wait_for_server_raises_when_the_server_exits():
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    proc.wait()
    with pytest.raises(RuntimeError, match="exited with code 3"):
        planner.wait_for_server(f"ws://127.0.0.1:{planner.free_port()}", proc, 60.0)


def test_stop_server_terminates_a_running_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    planner.stop_server(proc)
    assert proc.returncode is not None


def test_observation_has_pi05_camera_layout(frame: np.ndarray):
    state = np.arange(14, dtype=np.float64)
    obs = planner.build_observation(frame, state, "go to the door")
    assert set(obs["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert obs["image"]["base_0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(obs["image"]["base_0_rgb"], frame)
    for wrist in planner.WRIST_CAMERAS:
        assert obs["image"][wrist].shape == (224, 224, 3)
        assert obs["image"][wrist].dtype == np.uint8
        assert not obs["image"][wrist].any()
        assert obs["image_mask"][wrist] is np.False_
    assert obs["image_mask"]["base_0_rgb"] is np.True_
    assert obs["state"].dtype == np.float32
    np.testing.assert_array_equal(obs["state"], state)
    assert obs["prompt"] == "go to the door"


def test_observation_drops_alpha(frame: np.ndarray):
    rgba = np.concatenate([frame, np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
    np.testing.assert_array_equal(planner.build_observation(rgba, np.zeros(14), "x")["image"]["base_0_rgb"], frame)


def test_observation_refuses_a_grayscale_frame():
    with pytest.raises(ValueError, match="expected"):
        planner.build_observation(np.zeros((4, 4), dtype=np.uint8), np.zeros(14), "x")


def test_base_velocities_are_dims_14_and_15():
    base = planner.base_velocities(_actions())
    assert base.shape == (_HORIZON, 2)
    np.testing.assert_allclose(base[:, 0], np.arange(_HORIZON) / 100.0, rtol=1e-6)
    np.testing.assert_allclose(base[:, 1], -np.arange(_HORIZON) / 200.0, rtol=1e-6)


@pytest.mark.parametrize("shape", [(50, 14), (50, 32), (16,)])
def test_base_velocities_refuse_other_action_widths(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match="action chunk"):
        planner.base_velocities(np.zeros(shape))


def test_resample_averages_consecutive_twists():
    base = np.stack([np.arange(12.0), -np.arange(12.0)], axis=1)
    np.testing.assert_allclose(planner.resample(base, 5), [[2.0, -2.0], [7.0, -7.0]])
    np.testing.assert_array_equal(planner.resample(base, 1), base)


@pytest.mark.parametrize(("dt", "factor"), [(0.02, 1), (0.1, 5), (0.5, 25)])
def test_resample_factor_from_chunk_period(dt: float, factor: int):
    assert planner.resample_factor(dt) == factor


@pytest.mark.parametrize("dt", [0.03, 0.01, 0.0])
def test_resample_factor_refuses_fractional_periods(dt: float):
    with pytest.raises(ValueError, match="whole multiple"):
        planner.resample_factor(dt)


def test_manifest_chunk_plays_whole_entries_per_tick():
    chunk = planner._MANIFEST["chunk"]
    ticks_per_entry = chunk["dt"] * planner._MANIFEST["rate_hz"]
    assert ticks_per_entry == pytest.approx(round(ticks_per_entry))
    assert planner.resample_factor(chunk["dt"]) * planner.MODEL_DT * _HORIZON >= chunk["execute_s"]


def test_state_vector_is_the_embodiment_state_mean(tmp_path: pathlib.Path):
    mean = np.linspace(-0.5, 0.5, 14).tolist()
    stats = {"norm_stats": {"state": {"mean": mean, "std": [1.0] * 14}, "actions": {"mean": [0.0] * 16}}}
    (tmp_path / "assets" / "trossen_mobile").mkdir(parents=True)
    (tmp_path / "assets" / "trossen_mobile" / "norm_stats.json").write_text(json.dumps(stats))
    state = planner.state_vector(tmp_path, "trossen_mobile")
    assert state.dtype == np.float32
    np.testing.assert_allclose(state, mean, rtol=1e-6)


def test_state_vector_refuses_another_state_width(tmp_path: pathlib.Path):
    (tmp_path / "assets" / "fibocom_mobile").mkdir(parents=True)
    stats = {"norm_stats": {"state": {"mean": [0.0] * 15}}}
    (tmp_path / "assets" / "fibocom_mobile" / "norm_stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="expected \\(14,\\)"):
        planner.state_vector(tmp_path, "fibocom_mobile")


def test_step_without_instruction_holds_still(server: OpenpiServer, client: planner.PolicyClient, frame: np.ndarray):
    pi05 = planner.Pi05(client, np.zeros(14, dtype=np.float32), 5)
    for state in (None, {}, {"instruction": "   "}):
        pi05.reset(state)
        assert pi05.step({"camera": frame}) == Step()
    assert not server.received()


def test_step_without_camera_holds_still(server: OpenpiServer, client: planner.PolicyClient):
    pi05 = planner.Pi05(client, np.zeros(14, dtype=np.float32), 5)
    pi05.reset({"instruction": "go to the door"})
    assert pi05.step({"camera": None}) == Step()
    assert not server.received()


def test_step_sends_the_instruction_and_returns_the_resampled_base_chunk(
    server: OpenpiServer, client: planner.PolicyClient, frame: np.ndarray
):
    state = np.linspace(0.0, 1.0, 14, dtype=np.float32)
    pi05 = planner.Pi05(client, state, 5)
    pi05.reset({"instruction": "  walk to the red chair\n", "signal": ""})
    step = pi05.step({"camera": frame, "robot_pose": np.zeros(3)})
    assert len(step.chunk) == 10
    assert all(len(entry) == 2 for entry in step.chunk)
    np.testing.assert_allclose(step.chunk, planner.resample(planner.base_velocities(_actions()), 5))
    assert step.action == [] and step.signal == "" and step.command == ""
    assert pi05.last_infer_ms == 12.5
    (sent,) = server.received()
    assert sent["prompt"] == {"type": "str", "value": "walk to the red chair"}
    assert sent["state"]["dtype"] == "<f4"
    np.testing.assert_allclose(sent["state"]["values"], state, rtol=1e-6)
    base = sent["image"]["base_0_rgb"]
    assert (base["dtype"], base["shape"], base["sum"]) == ("|u1", [480, 640, 3], float(frame.sum()))
    for wrist in planner.WRIST_CAMERAS:
        assert sent["image"][wrist]["shape"] == [224, 224, 3]
        assert sent["image"][wrist]["sum"] == 0.0
        assert sent["image_mask"][wrist] == {"type": "bool", "value": False}
    assert sent["image_mask"]["base_0_rgb"] == {"type": "bool", "value": True}


def test_server_paths_name_the_fetch_command(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="arena feature planners add pi05"):
        planner.server_paths({"checkpoint": str(tmp_path / "absent")})


def test_server_command_pins_jax_to_cpu_for_a_pytorch_checkpoint(tmp_path: pathlib.Path):
    paths = {"checkpoint": tmp_path, "server_python": tmp_path / "python", "tokenizer": tmp_path / "tok"}
    config = {"embodiment": "arx_mobile", "flow_steps": 4, "compile_mode": "", "openpi_data": str(tmp_path / "d")}
    cmd, env = planner.server_command(config, paths, 4321)
    assert cmd[:2] == [str(tmp_path / "python"), str(planner._SERVER)]
    assert "--embodiment=arx_mobile" in cmd
    assert "--port=4321" in cmd
    assert "--flow-steps=4" in cmd
    assert "--compile-mode=none" in cmd
    assert env["OPENPI_DATA_HOME"] == str(tmp_path / "d")
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env.get("JAX_PLATFORMS") == os.environ.get("JAX_PLATFORMS")
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert planner.server_command(config, paths, 4321)[1]["JAX_PLATFORMS"] == "cpu"


def test_unknown_embodiment_is_refused_before_anything_starts():
    with pytest.raises(ValueError, match="trossen_mobile"):
        planner.on_init({"embodiment": "fibocom_mobile"})
