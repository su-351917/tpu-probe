# kaggle_launcher.py : Kaggle TPU kernel that NEVER needs re-pushing. It clones a GitHub repo, runs `bash run.sh`
# whenever the repo's HEAD changes, and keeps polling for ~8.4 h (Kaggle TPU session cap is 9 h). Fix code -> push ->
# the running session picks it up within ~90 s. A file named STOP in the repo ends the session early (so /kaggle/working is saved).
import os, subprocess, time, sys, shutil
REPO = os.environ.get("REPO", "https://github.com/su-351917/tpu-probe.git"); BRANCH = os.environ.get("BRANCH", "main")
POLL, BUDGET = 90, 8.4 * 3600
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    try: REPO = _s.get_secret("TPU_REPO") or REPO
    except Exception: pass
    try:
        tok = _s.get_secret("GITHUB_TOKEN")
        if tok and REPO.startswith("https://github.com/"): REPO = REPO.replace("https://github.com/", f"https://x-access-token:{tok}@github.com/"); os.environ["GITHUB_TOKEN"] = tok
    except Exception: pass
except Exception: pass
W = "/kaggle/working"; SRC = f"{W}/src"; LOGS = f"{W}/logs"; os.makedirs(LOGS, exist_ok=True)
def warm():  # touch the TPU so Kaggle's "TPU idle > 2 h" watchdog (exit 137, seen 2026-10-14) does not kill the session
    try:
        import torch, torch_xla.core.xla_model as xm
        d = xm.xla_device(); a = torch.randn(1024, 1024, device=d); b = (a @ a).sum(); xm.mark_step(); return float(b)
    except Exception as e: return f"warm failed: {e}"
def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
def head():
    r = sh(f"cd {SRC} && git rev-parse --short HEAD"); return r.stdout.strip() if r.returncode == 0 else None
t0 = time.time(); last = None; n = 0
print("launcher start", "token" if os.environ.get("GITHUB_TOKEN") else "NO GITHUB_TOKEN secret (push-back will fail)", time.strftime("%H:%M:%S"), "repo", REPO.split("@")[-1], flush=True)
while time.time() - t0 < BUDGET:
    if not os.path.isdir(SRC):
        r = sh(f"git clone -q --depth 50 -b {BRANCH} '{REPO}' {SRC}")
        if r.returncode != 0: print("clone failed:", r.stderr[-300:], flush=True); time.sleep(POLL); continue
    else:
        r = sh(f"cd {SRC} && git fetch -q origin {BRANCH} && git reset -q --hard origin/{BRANCH}")
        if r.returncode != 0: print("fetch failed:", r.stderr[-300:], flush=True)
    h = head()
    if h and h != last:
        n += 1; last = h; log = f"{LOGS}/run{n:02d}_{h}.log"; print(f"=== run {n} @ {h} {time.strftime('%H:%M:%S')} -> {log}", flush=True)
        with open(log, "w") as f:
            p = subprocess.Popen(["bash", "run.sh"], cwd=SRC, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, RUN_LOG=log, RUN_HASH=h))
            try: p.wait(timeout=BUDGET - (time.time() - t0) - 120)
            except subprocess.TimeoutExpired: p.kill(); print("run timed out", flush=True)
        tail = open(log, errors="ignore").read()[-3000:]; print(f"--- run {n} exit {p.returncode}; tail:\n{tail}", flush=True)
        for f in os.listdir(SRC):
            if f.startswith("result") and (f.endswith(".json") or f.endswith(".jsonl")): shutil.copy(f"{SRC}/{f}", f"{LOGS}/{f}")
        if os.path.exists(f"{SRC}/STOP"): print("STOP file present -> ending session", flush=True); break
    else:
        print(f"idle {int((time.time() - t0) / 60)} min, head {h}, warm {warm()}", flush=True)
    time.sleep(POLL)
print("launcher end", time.strftime("%H:%M:%S"), "runs", n, flush=True)
