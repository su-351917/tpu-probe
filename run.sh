#!/bin/bash
# executed by kaggle_launcher.py on every new commit. Edit freely; the kernel itself never changes.
set -x
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch_xla; print('torch_xla', torch_xla.__version__)" 2>&1 | tail -1
pip list 2>/dev/null | grep -iE "^(torch|torch-xla|torch_xla|jax|libtpu|transformers) " 
STAGE=${STAGE:-1} python tpu_probe.py
