# tpu_probe.py : can the fixed-key memory CPT run on Kaggle TPU v5e-8 under torch_xla?
# STAGE 1: single chip, LFM2.5-350M + Mem (static shapes), random tokens, 20 train steps -> compile time, step time.
# STAGE 2: 8 chips data-parallel (xmp.spawn), same measurement. Writes result_tpu_probe.jsonl (copied to /kaggle/working/logs by the launcher).
import os, sys, time, json, math, subprocess
E = os.environ.get
def ensure_xla():
    try:
        import torch_xla; return
    except Exception as e:
        print("torch_xla import failed:", repr(e)[:200], "-> pip install", flush=True)
        import torch; v = torch.__version__.split("+")[0]
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"torch_xla[tpu]=={v}", "-f", "https://storage.googleapis.com/libtpu-releases/index.html", "-f", "https://storage.googleapis.com/libtpu-wheels/index.html"], capture_output=True, text=True)
        print("pip rc", r.returncode, r.stderr[-400:], flush=True)
        import torch_xla
ensure_xla()
import torch, torch.nn as nn, torch.nn.functional as F
import torch_xla, torch_xla.core.xla_model as xm
MODEL = E("MODEL", "LiquidAI/LFM2.5-350M-Base"); T, MB, STEPS = int(E("T", "1024")), int(E("MB", "4")), int(E("STEPS", "20"))
MEM_LAYERS = [int(x) for x in E("MEM_LAYERS", "3,7,11,15").split(",")]; NSUB, HALF, KTOP = int(E("NSUB", "192")), 128, int(E("KTOP", "128"))
FREEZE_CORE = E("FREEZE_CORE", "0") == "1"; STAGE = int(E("STAGE", "1"))
G = torch.Generator().manual_seed(1234)
class Mem(nn.Module):
    def __init__(self, d):
        super().__init__(); self.q = nn.Linear(d, 2 * HALF, bias=False); self.ln = nn.LayerNorm(2 * HALF)
        self.register_buffer("keys", torch.randn(2, NSUB, HALF, generator=G) / math.sqrt(HALF)); self.values = nn.Embedding(NSUB * NSUB, d); nn.init.zeros_(self.values.weight)
    def forward(self, x):
        s = x.shape[:-1]; q = self.ln(self.q(x.reshape(-1, x.shape[-1]).float())).view(-1, 2, HALF)
        s1 = q[:, 0] @ self.keys[0].t(); s2 = q[:, 1] @ self.keys[1].t(); v1, i1 = s1.topk(KTOP, -1); v2, i2 = s2.topk(KTOP, -1)
        cand = (v1[:, :, None] + v2[:, None, :]).view(-1, KTOP * KTOP); ci = (i1[:, :, None] * NSUB + i2[:, None, :]).view(-1, KTOP * KTOP)
        sc, j = cand.topk(KTOP, -1); idx = ci.gather(1, j); w = F.relu(sc) / KTOP
        v = F.embedding(idx, self.values.weight)                      # (n, k, d) static-shape gather (embedding_bag avoided for XLA)
        return (v * w[..., None].to(v.dtype)).sum(1).view(*s, -1).to(x.dtype)
class FFNPlusMem(nn.Module):
    def __init__(self, ffn, d): super().__init__(); self.ffn = ffn; self.mem = Mem(d)
    def forward(self, x): return self.ffn(x) + self.mem(x)
def build(dev):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16); D = model.config.hidden_size
    for i in MEM_LAYERS: model.model.layers[i].feed_forward = FFNPlusMem(model.model.layers[i].feed_forward, D)
    model = model.to(dev); model.train()
    vals = [model.model.layers[i].feed_forward.mem.values.weight for i in MEM_LAYERS]; vid = {id(p) for p in vals}
    core = [p for n, p in model.named_parameters() if id(p) not in vid and "embed" not in n]
    if FREEZE_CORE:
        for p in core: p.requires_grad_(False)
    for n, p in model.named_parameters():
        if "embed" in n: p.requires_grad_(False)
    groups = [{"params": vals, "lr": 3e-4}] + ([] if FREEZE_CORE else [{"params": core, "lr": 3e-5}])
    return model, torch.optim.AdamW(groups, weight_decay=0.0)
def run(rank=0, nproc=1):
    dev = xm.xla_device(); torch.manual_seed(rank)
    model, opt = build(dev); V = model.config.vocab_size; res = []
    for step in range(STEPS):
        t = time.time(); x = torch.randint(0, V, (MB, T + 1), device=dev)
        loss = F.cross_entropy(model(input_ids=x[:, :-1]).logits.float().view(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        if nproc > 1: xm.optimizer_step(opt)
        else: opt.step()
        xm.mark_step(); l = loss.item(); dt = time.time() - t   # .item() syncs -> real step time
        res.append(dt)
        if rank == 0: print(f"stage{STAGE} rank{rank} step {step} loss {l:.3f} {dt:.2f}s", flush=True)
    if rank == 0:
        steady = sorted(res[5:])[:max(1, (len(res) - 5) // 2)] if len(res) > 6 else res
        out = {"stage": STAGE, "nproc": nproc, "T": T, "MB": MB, "steps": STEPS, "first_step_s": round(res[0], 1), "second_step_s": round(res[1], 1) if len(res) > 1 else None,
               "steady_step_s": round(sum(steady) / len(steady), 3), "tok_per_s_per_chip": round(MB * T / (sum(steady) / len(steady))), "tok_per_s_total": round(nproc * MB * T / (sum(steady) / len(steady))),
               "freeze_core": FREEZE_CORE, "model": MODEL, "torch": torch.__version__, "torch_xla": torch_xla.__version__}
        print("RESULT_JSON " + json.dumps(out), flush=True); open("result_tpu_probe.jsonl", "a").write(json.dumps(out) + "\n")
def _mp(index, *a): run(index, xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else int(E("NPROC", "8")))
if __name__ == "__main__":
    print("devices:", torch_xla.device_count() if hasattr(torch_xla, "device_count") else "?", flush=True)
    if STAGE == 1: run(0, 1)
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_mp, args=())
