#!/bin/bash
# serve.sh : install vLLM for TPU and run serve_bench.py. Logs everything; falls back model by model.
set -x
df -h /kaggle/working /tmp 2>/dev/null | tail -2; free -g | head -2
export HF_HOME=/kaggle/working/hf; mkdir -p $HF_HOME
pip install -q "vllm-tpu==0.31.0" 2>&1 | tail -3
python -c "import vllm, sys; print('vllm', vllm.__version__)" 2>&1 | tail -1
export PJRT_DEVICE=TPU
python serve_bench.py 2>&1 | grep -vE "^\s*$" | grep -E "SERVE_|Error|error|Traceback|vllm|INFO.*(tok/s|throughput|Loading|compil)" | cut -c1-400 | tail -200
cp result_serve_bench.jsonl /kaggle/working/logs/ 2>/dev/null
