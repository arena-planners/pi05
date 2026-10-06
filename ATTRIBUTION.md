# Attribution

## Code

Built on openpi (Physical Intelligence), commit 215abfb (2025), https://github.com/Physical-Intelligence/openpi,
Apache-2.0 (see LICENSE). openpi is installed unmodified into the server venv, only its `transformers_replace` patches
are copied into that venv's transformers as openpi's README prescribes.

- `server/serve.py` follows `scripts/serve_policy.py` (policy from `policy_config.create_trained_policy`, served by
  `serving.websocket_policy_server.WebsocketPolicyServer`) with the `pi05_aloha` model config of
  `training/config.py`. `MobileInputs` and `MobileOutputs` replace `policies/aloha_policy.py`'s `AlohaInputs` and
  `AlohaOutputs`: the camera layout of `AlohaInputs` (base view plus masked blank wrist views) comes from the planner,
  and the outputs keep dims 0 to 15 instead of 0 to 13.
- `planner.py`'s `to_wire`, `from_wire` and `PolicyClient` reimplement openpi-client's `msgpack_numpy.py` and
  `websocket_client_policy.py` wire format, so the planner venv needs neither openpi-client (numpy<2) nor
  msgpack-numpy's format, which `arena_planners` patches into msgpack process-wide.
- `server/setup.sh` runs `examples/convert_jax_model_to_pytorch.py` from the same commit.

## Weights

`gs://openpi-assets/checkpoints/pi05_base` (params and norm stats) and `gs://big_vision/paligemma_tokenizer.model`.
pi0.5 is built on PaliGemma, and its weights are subject to the Gemma Terms of Use
(https://ai.google.dev/gemma/terms, openpi's `LICENSE_GEMMA.txt`).
