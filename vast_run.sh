#!/bin/bash
# vast_run.sh : onstart for a rented single-GPU box. Runs the jobs in $JOBS (';'-separated "NAME|ENV..." entries) with tiny_hashkey.py on CUDA.
# Data: public token bins from the HF dataset sasoriiii/cpt-tok-bins. Outputs: /workspace/out/<NAME>.log + jsonl. STOP file halts.
set -x
cd /workspace || cd /root
export DEBIAN_FRONTEND=noninteractive
python -c "import torch,sys; sys.exit(0 if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])>=(2,6) else 1)" || pip install -q "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1   # transformers 5.19 needs torch.accelerator (>=2.6); image ships 2.5.1
pip install -q "transformers==5.19.0" "tokenizers==0.23.2" sentencepiece numpy huggingface_hub 2>&1 | tail -1
[ -d src ] || git clone -q --depth 5 https://github.com/su-351917/tpu-probe.git src
mkdir -p data/tok out
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("sasoriiii/cpt-tok-bins", repo_type="dataset", local_dir="/workspace/data")
PY
ls -la data/tok
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd src
STEPS=${STEPS:-4000}
IFS=';' read -ra JL <<< "$JOBS"
for j in "${JL[@]}"; do
  name=${j%%|*}; envs=${j#*|}
  [ -e /workspace/STOP ] && break
  echo "=== job $name [$envs] $(date +%H:%M:%S)"
  ( export DATA=/workspace/data/tok OUT=/workspace/out STEPS=$STEPS; export $envs; python tiny_hashkey.py > /workspace/out/$name.log 2>&1; echo "EXIT $? $name $(date +%H:%M:%S)" >> /workspace/out/status.log )
  echo "--- results $name"; grep -E "^(ARM|EVAL|ADDR|RESULT|DIST|WRITE|GROW|Traceback|.*Error)" /workspace/out/$name.log | cut -c1-1200   # results go to stdout so the instance log (API) carries them; no secrets needed on the box
done
echo ALL_JOBS_DONE >> /workspace/out/status.log
