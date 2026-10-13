# tpu_probe
Kaggle TPU (v5e-8) smoke test for the fixed-key memory CPT (LFM2.5 + product-key tables) under torch_xla.
`kaggle_launcher.py` is the Kaggle kernel: it clones this repo and runs `run.sh` on every new commit for ~8.4 h, so code fixes never need a new kernel push (and a new queue wait). Put a file named `STOP` in the repo to end the session.
Public models and random tokens only; no private data.
