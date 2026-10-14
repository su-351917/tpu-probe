#!/bin/bash
# executed by kaggle_launcher.py on every new commit. Edit freely; the kernel itself never changes.
set -x
echo "== diagnostics (lesson from CIT TPU probes a1-a6: JAX saw only TFRT_CPU_0; check allocation before anything)"
env | grep -iE "tpu|pjrt|xla|accelerator|kaggle" | sort
ls -la /dev/accel* /dev/vfio 2>/dev/null; lspci 2>/dev/null | grep -i -E "google|tpu" ; nproc; free -g | head -2
python -c "import torch; print('torch', torch.__version__)"
python -c "import jax; print('jax', jax.__version__, jax.devices())" 2>&1 | tail -2
python -c "import torch_xla; print('torch_xla', torch_xla.__version__)" 2>&1 | tail -1
pip list 2>/dev/null | grep -iE "^(torch|torch-xla|torch_xla|jax|jaxlib|libtpu|libtpu-nightly|transformers) "
export PJRT_DEVICE=TPU
[ -z "$TPU_ACCELERATOR_TYPE" ] && export TPU_ACCELERATOR_TYPE=v5litepod-8
STAGE=${STAGE:-1} python tpu_probe.py
# --- report back: push this run's log + result files to the `results` branch (launcher watches main only, so no re-trigger)
set +x
R=$(git config --get remote.origin.url); H=${RUN_HASH:-$(git rev-parse --short HEAD)}; T=$(date +%H%M%S)
D=$(mktemp -d) && git clone -q --depth 1 -b results "$R" "$D" 2>/dev/null || { git init -q "$D"; git -C "$D" checkout -q -b results; git -C "$D" remote add origin "$R"; }
mkdir -p "$D/$H" && cp "$RUN_LOG" "$D/$H/run_$T.log" 2>/dev/null; cp result*.json* "$D/$H/" 2>/dev/null
git -C "$D" -c user.name=kaggle -c user.email=kaggle@local add -A && git -C "$D" -c user.name=kaggle -c user.email=kaggle@local commit -qm "results $H $T" && git -C "$D" push -q origin results && echo "results pushed: $H/run_$T.log"
