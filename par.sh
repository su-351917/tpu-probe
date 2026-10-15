#!/bin/bash
# par.sh : run 8 INDEPENDENT 20M experiments, one per TPU chip (no collectives). Each process is pinned to a chip via TPU_VISIBLE_CHIPS.
# Jobs are "NAME|ENV..." lines in jobs.txt (edit + push to change). Data: public token bins from the HF dataset below.
set -x
W=/kaggle/working; DATA=$W/data; mkdir -p $DATA/tok $W/logs $W/out_par
pip install -q -U huggingface_hub 2>/dev/null | tail -1
for i in $(seq 1 40); do   # the upload may still be in progress: retry until all files are present
  python - <<'PY' && break
from huggingface_hub import snapshot_download
import os
p = snapshot_download("sasoriiii/cpt-tok-bins", repo_type="dataset", local_dir="/kaggle/working/data")
need = ["tok/fineweb_edu_en.bin", "tok/fineweb2_ja.bin", "tok/wiki_ja.bin"]
missing = [f for f in need if not os.path.exists(os.path.join(p, f))]
print("missing", missing); raise SystemExit(1 if missing else 0)
PY
  sleep 60
done
ls -la $DATA/tok
STEPS=${STEPS:-30}
i=0
while IFS='|' read -r name envs; do
  [ -z "$name" ] && continue
  echo "chip $i -> $name [$envs]"
  ( export PJRT_DEVICE=TPU TPU_VISIBLE_CHIPS=$i TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_PORT=$((8476+i)) CLOUD_TPU_TASK_ID=0 \
      TPU=1 DATA=$DATA/tok OUT=$W/out_par STEPS=$STEPS; export $envs; python tiny_hashkey.py > $W/logs/par_${name}.log 2>&1; echo "EXIT $? $name" >> $W/logs/par_status.log ) &
  i=$((i+1)); [ $i -ge 8 ] && break
done < jobs.txt
wait
for f in $W/logs/par_*.log; do echo "== $f"; grep -E "^(ARM|EVAL|ADDR|RESULT|Traceback|.*Error)" $f | tail -6 | cut -c1-300; done
cp $W/out_par/*.jsonl $W/logs/ 2>/dev/null; cat $W/logs/par_status.log
