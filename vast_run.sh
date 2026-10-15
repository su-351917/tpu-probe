#!/bin/bash
# vast_run.sh : onstart for a rented single-GPU box. Runs the jobs in $JOBS (';'-separated "NAME|ENV..." entries) with tiny_hashkey.py on CUDA.
# Data: public token bins from the HF dataset sasoriiii/cpt-tok-bins. Outputs: /workspace/out/<NAME>.log + jsonl. STOP file halts.
set -x
cd /workspace || cd /root
export DEBIAN_FRONTEND=noninteractive
pip install -q transformers==4.57.1 sentencepiece numpy huggingface_hub 2>&1 | tail -1
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
done
echo ALL_JOBS_DONE >> /workspace/out/status.log
