#!/bin/bash
# executed by kaggle_launcher.py on every new commit. MODE=par: 8 independent experiments, one per chip (par.sh). MODE=probe: tpu_probe.py stage test.
set -x
env | grep -iE "tpu|pjrt" | sort | head -12; nproc; free -g | head -2
python -c "import torch, torch_xla; print('torch', torch.__version__, 'torch_xla', torch_xla.__version__)" 2>&1 | tail -1
# repair the base env if a previous serve run (vllm-tpu into the system site-packages) broke torch/torch_xla (seen 2026-10-17: torchaudio ABI, _XLAC undefined symbol)
python -c "import torch_xla" 2>/dev/null || { pip uninstall -y -q vllm vllm-tpu tpu-inference torchaudio torchvision 2>&1 | tail -1; pip install -q "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1; pip install -q "torch_xla[tpu]==2.8.0" 2>&1 | tail -1; python -c "import torch, torch_xla; print('repaired torch', torch.__version__, 'torch_xla', torch_xla.__version__)" 2>&1 | tail -1; }
export PJRT_DEVICE=TPU
[ -z "$TPU_ACCELERATOR_TYPE" ] && export TPU_ACCELERATOR_TYPE=v5litepod-8
MODE=${MODE:-$(cat MODE 2>/dev/null || echo par)}
case "$MODE" in
  par) STEPS=${STEPS:-30} bash par.sh ;;
  serve) bash serve.sh ;;
  report) echo "== SERVE lines from earlier runs"; grep -h "^SERVE_" /kaggle/working/logs/run0*.log 2>/dev/null | cut -c1-700; echo "== result_serve_bench.jsonl"; cat /kaggle/working/logs/result_serve_bench.jsonl 2>/dev/null | cut -c1-700; df -h /kaggle/working | tail -1; du -sh /kaggle/working/hf 2>/dev/null; rm -rf /kaggle/working/hf/hub/models--* ; df -h /kaggle/working | tail -1 ;;   # print what the launcher's tail cut off, then free the model cache
  *) STAGE=${STAGE:-2} python tpu_probe.py ;;
esac
# --- report back (needs GITHUB_TOKEN secret; harmless if absent)
set +x
R=$(git config --get remote.origin.url); [ -n "$GITHUB_TOKEN" ] && R="https://x-access-token:${GITHUB_TOKEN}@github.com/su-351917/tpu-probe.git"; H=${RUN_HASH:-$(git rev-parse --short HEAD)}; T=$(date +%H%M%S)
D=$(mktemp -d) && git clone -q --depth 1 -b results "$R" "$D" 2>/dev/null || { git init -q "$D"; git -C "$D" checkout -q -b results; git -C "$D" remote add origin "$R"; }
mkdir -p "$D/$H" && cp "$RUN_LOG" "$D/$H/run_$T.log" 2>/dev/null; cp result*.json* "$D/$H/" 2>/dev/null; cp /kaggle/working/logs/par_*.log "$D/$H/" 2>/dev/null
git -C "$D" -c user.name=kaggle -c user.email=kaggle@local add -A && git -C "$D" -c user.name=kaggle -c user.email=kaggle@local commit -qm "results $H $T" && git -C "$D" push -q origin results && echo "results pushed: $H/run_$T.log"
