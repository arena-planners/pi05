# pi05

Physical Intelligence's [pi0.5](https://www.physicalintelligence.company/blog/pi05) base checkpoint (`pi05_base`,
[openpi](https://github.com/Physical-Intelligence/openpi)) behind the arena_planners bridge, as a zero-shot navigation
baseline. It drives the robot through the two action dims that command a mobile manipulator's base, `[v, omega]`,
from the front camera and a text instruction, the way VEGA (arXiv 2606.18426, sec. 5.1) evaluates it. No public
navigation fine-tune of pi0.5 exists, so this is the base model as released.

```sh
arena feature planners add pi05
planners/pi05/server/setup.sh
arena launch sim:=gazebo robot:=jackal[camera=vln_navid] robot.planner:=pi05
```

`robot.planner:=pi05` selects `robot.mobile:=vla`. The planner accepts the instruction only (`goal_inputs:
[instruction]`), the goal pose never reaches it and decides success alone. It has no stop head and sends no signals,
Arena judges arrival.

## Model inputs and outputs

openpi's `pi05_aloha` inference config (openpi's own default `serve_policy.py --env ALOHA`, which also serves
`pi05_base`) with two changes in `server/serve.py`: the Mobile ALOHA norm stats (`trossen_mobile`) instead of the
static ALOHA ones, and an output transform that keeps the 16 action dims of that embodiment instead of the 14 arm dims
(`AlohaOutputs` drops the base).

| Input | Value |
|---|---|
| `base_0_rgb` | the robot camera frame, uint8 RGB at its native size, `resize_with_pad` to 224x224 on the server |
| `left_wrist_0_rgb`, `right_wrist_0_rgb` | black, image mask false |
| state (14 dims) | the embodiment's mean state from its `norm_stats.json`, constant: the robot has no arms |
| prompt | the goal instruction, whitespace-stripped |

The model samples 50 actions of 32 dims with 10 flow-matching steps, unnormalized with quantile stats (`[-1, 1]` maps
to `[q01, q99]`). Dims 14 and 15 are the base's linear and angular velocity, Mobile ALOHA's `base_action`. For
`trossen_mobile`, `q01`/`q99` are -0.219/0.398 m/s and -0.314/0.252 rad/s.

The chunk is 50 Hz (1 s). The planner averages it into 0.1 s twists (10 entries) and returns a `velocity_chunk` at
`rate_hz: 10`, so every entry plays for one bridge tick. The bridge runs 0.5 s of it (`chunk.execute_s`, pi0.5's
replanning cadence of 25 steps), clamps each twist to the robot's velocity limits, and sends the next frame. There is no
rescaling between Mobile ALOHA's base and the robot's.

`robot.mobile.instruction:=<text>` stands in for goals that carry none. Without an instruction or a camera frame the
planner stands still without calling the model.

## Server

The model runs in a separate process: openpi pins Python 3.11, numpy<2, torch 2.7.1 and transformers 4.53.2, which
cannot share the planner venv. `server/` is a uv project for that venv, `server/serve.py` builds the policy and serves
it with openpi's websocket server on `127.0.0.1`. The planner spawns it at init, waits for its metadata frame, talks
openpi's msgpack wire format to it, and stops it at exit.

`server/setup.sh` (idempotent) syncs that venv (torch from the cu128 index, which carries Blackwell `sm_120` kernels),
applies openpi's `transformers_replace` patches into the venv's own copy of transformers (`UV_LINK_MODE=copy`, so the
uv cache stays clean), and converts the JAX checkpoint to PyTorch with openpi's
`examples/convert_jax_model_to_pytorch.py` at the pinned commit, copying the norm-stat assets the converter misses.

## Config

Set in `planner.yaml > config`, override per launch with `robot.mobile.config.<key>:=<value>`.

| Key | Default | Effect |
|---|---|---|
| `checkpoint` | `model/pi05_base_pytorch` | Checkpoint dir. A dir with `model.safetensors` loads in PyTorch (JAX pinned to CPU), `model/pi05_base` loads the original JAX checkpoint. |
| `embodiment` | `trossen_mobile` | Norm stats and state: `trossen_mobile` (Mobile ALOHA) or `arx_mobile`, both with the base at dims 14 and 15. |
| `device` | `auto` | PyTorch device, `auto` picks CUDA when available. |
| `flow_steps` | `10` | Flow-matching integration steps. |
| `compile_mode` | empty | `torch.compile` mode for `sample_actions` (openpi's default is `max-autotune`), empty compiles nothing. |
| `server_python` | `server/.venv/bin/python` | Interpreter of the server venv. |
| `tokenizer` | `model/paligemma_tokenizer.model` | PaliGemma tokenizer, copied into `openpi_data` so the server never downloads it. |
| `openpi_data` | `model/openpi-data` | `OPENPI_DATA_HOME` of the server. |
| `port` | `0` | Server port on 127.0.0.1, 0 picks a free one. |
| `startup_timeout_s` | `900` | Time allowed for the server to load the checkpoint. |

## Weights and license

`weights.yaml` fetches the 12.4 GB `gs://openpi-assets/checkpoints/pi05_base` (params plus the `trossen_mobile` and
`arx_mobile` norm stats) and the PaliGemma tokenizer from `gs://big_vision`. The pi0.5 weights derive from PaliGemma
and are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), see openpi's `LICENSE_GEMMA.txt`.
The code is Apache-2.0, see LICENSE and ATTRIBUTION.md.

```bibtex
@article{intelligence2025pi05,
  title={$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization},
  author={{Physical Intelligence} and others},
  journal={arXiv preprint arXiv:2504.16054},
  year={2025}
}
```
