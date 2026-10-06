"""openpi websocket server for pi0.5 base with a mobile-base embodiment's norm stats, run in the server venv."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import shutil

import numpy as np
from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import download
from openpi.training import config as _config

EMBODIMENTS = ("trossen_mobile", "arx_mobile")
ACTION_DIMS = 16
MODEL_DT = 0.02
_TOKENIZER = pathlib.PurePosixPath("big_vision", "paligemma_tokenizer.model")


@dataclasses.dataclass(frozen=True)
class MobileInputs(transforms.DataTransformFn):
    """Model inputs exactly as the planner sends them, the camera keys already in pi0.5's layout."""

    def __call__(self, data: dict) -> dict:
        return {
            "image": {name: np.asarray(img) for name, img in data["image"].items()},
            "image_mask": {name: np.bool_(mask) for name, mask in data["image_mask"].items()},
            "state": np.asarray(data["state"], dtype=np.float32),
            "prompt": str(data["prompt"]),
        }


@dataclasses.dataclass(frozen=True)
class MobileOutputs(transforms.DataTransformFn):
    """The embodiment's 16 action dims (two 7-dof arms, then the base), dropping the padding."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :ACTION_DIMS])}


def train_config(checkpoint: pathlib.Path, embodiment: str, compile_mode: str | None) -> _config.TrainConfig:
    return _config.TrainConfig(
        name="pi05_arena_mobile",
        model=pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=compile_mode),
        data=_config.SimpleDataConfig(
            assets=_config.AssetsConfig(assets_dir=str(checkpoint / "assets"), asset_id=embodiment),
            data_transforms=lambda model: transforms.Group(inputs=[MobileInputs()], outputs=[MobileOutputs()]),
        ),
        policy_metadata={"embodiment": embodiment, "action_dims": ACTION_DIMS, "model_dt": MODEL_DT},
    )


def stage_tokenizer(tokenizer: pathlib.Path) -> None:
    """Copy the PaliGemma tokenizer where openpi looks for it, so the server never downloads it."""
    target = download.get_cache_dir() / _TOKENIZER
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tokenizer, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--embodiment", choices=EMBODIMENTS, default="trossen_mobile")
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="none",
    )
    args = parser.parse_args()

    stage_tokenizer(args.tokenizer)
    compile_mode = None if args.compile_mode == "none" else args.compile_mode
    policy = policy_config.create_trained_policy(
        train_config(args.checkpoint, args.embodiment, compile_mode),
        args.checkpoint,
        sample_kwargs={"num_steps": args.flow_steps},
        pytorch_device=None if args.device == "auto" else args.device,
    )
    logging.info("pi05 server: %s on %s:%d", args.checkpoint, args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
