#!/bin/bash
# serve.sh : install vLLM for TPU and run serve_bench.py. Logs everything; falls back model by model.
set -x
df -h /kaggle/working /tmp 2>/dev/null | tail -2; free -g | head -2
export HF_HOME=/kaggle/working/hf; mkdir -p $HF_HOME
V=/kaggle/working/venv_vllm   # own venv: vllm-tpu replaces torch, which broke torch_xla for the launcher's warm/par modes when installed system-wide
[ -x $V/bin/python ] || python -m venv $V
$V/bin/pip install -q "vllm-tpu==0.31.0" 2>&1 | tail -3
$V/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" 2>&1 | tail -1
export PJRT_DEVICE=TPU
$V/bin/python serve_bench.py 2>&1 | grep -vE "^\s*$" | grep -E "SERVE_|Error|error|Traceback|vllm|INFO.*(tok/s|throughput|Loading|compil)" | cut -c1-400 | tail -200
cp result_serve_bench.jsonl /kaggle/working/logs/ 2>/dev/null
