#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

server=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
planner=$(dirname "$server")
rev=215abfb217dbac7d5f1273282331b9b1866c0479
jax_ckpt=$planner/model/pi05_base
torch_ckpt=$planner/model/pi05_base_pytorch
python=$server/.venv/bin/python

if [ ! -d "$jax_ckpt/params" ]; then
    echo "pi05: $jax_ckpt/params missing, run \`arena feature planners add pi05\` first" >&2
    exit 1
fi

export UV_LINK_MODE=copy
(cd "$server" && uv sync --python 3.11)

transformers=$("$python" -c "import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)")
openpi=$("$python" -c "import pathlib, openpi; print(pathlib.Path(openpi.__file__).parent)")
cp -r "$openpi/models_pytorch/transformers_replace/." "$transformers/"

if [ ! -f "$torch_ckpt/model.safetensors" ]; then
    scratch=$(mktemp -d "$planner/model/.convert.XXXXXX")
    trap 'rm -rf "$scratch"' EXIT
    curl -fsSL -o "$scratch/convert.py" \
        "https://raw.githubusercontent.com/Physical-Intelligence/openpi/$rev/examples/convert_jax_model_to_pytorch.py"
    JAX_PLATFORMS=cpu "$python" "$scratch/convert.py" \
        --checkpoint_dir "$jax_ckpt" --config_name pi05_aloha --output_path "$scratch/pi05_base_pytorch"
    rm -rf "$scratch/pi05_base_pytorch/assets"
    cp -rL "$jax_ckpt/assets" "$scratch/pi05_base_pytorch/assets"
    rm -rf "$torch_ckpt"
    mv "$scratch/pi05_base_pytorch" "$torch_ckpt"
fi
echo "pi05: server venv at $server/.venv, PyTorch checkpoint at $torch_ckpt"
