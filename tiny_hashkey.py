# tiny_hashkey.py : 10M-class from-scratch LM + memory tables with different KEY designs (PROTOCOL_hashkey_tiny.md).
# ARM=ctx|hash|hashctx|none. Trains ~3k steps on the CPT bins, then runs the addressing diagnostic and a values-only write test.
import os, math, json, time, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
E = os.environ.get
ARM = E("ARM", "ctx"); LOOP = int(E("LOOP", "1")); NBLK = int(E("NBLK", "4")); STEPS = int(E("STEPS", "3000")); BS, T = int(E("BS", "16")), 512; ACC = int(E("ACC", "1")); D, L, H = int(E("D", "256")), 4, 4
DATA = E("DATA", "/mnt/work2/cpt_lfm/data2/tok"); OUT = E("OUT", "/mnt/work2/cpt_lfm/out_tiny"); os.makedirs(OUT, exist_ok=True)
MIX = "fineweb2_ja:0.5,fineweb_edu_en:0.3,wiki_ja:0.2"; MEM_LAYERS = [int(x) for x in E("MEM_LAYERS", "1,3").split(",")]; ROWS = int(E("ROWS", "16384"))
TPU = E("TPU", "0") == "1"
if TPU:
    import torch_xla, torch_xla.core.xla_model as xm; dev = xm.xla_device(); AC, ACD = "xla", torch.bfloat16
    def step_mark(): xm.mark_step()
else:
    dev = "cuda"; AC, ACD = "cuda", torch.float16
    def step_mark(): pass
_SEED = int(E("SEED", "0")); torch.manual_seed(_SEED); random.seed(_SEED); np.random.seed(_SEED)
VOCAB = 65536
bins, probs = {}, []
for it in MIX.split(","):
    n, p = it.split(":"); bins[n] = np.memmap(f"{DATA}/{n}.bin", dtype=np.uint16, mode="r"); probs.append(float(p))
names = list(bins); HOLD = 1_000_000
def batch(n):
    out = []
    for _ in range(n):
        b = bins[names[np.random.choice(len(names), p=probs)]]; o = np.random.randint(0, len(b) - HOLD - T - 1)
        out.append(np.asarray(b[o:o + T + 1]).astype(np.int64))
    return torch.from_numpy(np.stack(out)).to(dev)

# ---------------- memory tables ----------------
IDF = E("IDF", "0") == "1"   # IDF=1: each unigram in the hashbag is weighted by log(N/(count+1)) (mean 1), so rare tokens dominate the address (BM25 idea)
if IDF:
    _cnt = np.zeros(VOCAB, dtype=np.int64)
    for _b in bins.values(): _cnt += np.bincount(np.asarray(_b[:20_000_000]).astype(np.int64), minlength=VOCAB)
    _idf = np.log(_cnt.sum() / (_cnt + 1.0)); _idf = _idf / _idf[_cnt > 0].mean()
    IDF_W = torch.tensor(_idf, dtype=torch.float32, device=dev)
def ihash(ids, mults, mod):   # ids: (B,T) int64; n-gram hash of the last len(mults) tokens (left-padded with 0) -> (B,T) in [0,mod)
    h = torch.zeros_like(ids)
    for k, m in enumerate(mults):
        sh = F.pad(ids, (k, 0))[:, :ids.shape[1]] if k else ids
        h = (h + sh * m) % 1_000_000_007
    return h % mod
class CtxMem(nn.Module):      # current main-line design: fixed random product keys, query from hidden state
    NSUB, HALF, KTOP = 128, 64, 16
    def __init__(s):
        super().__init__(); g = torch.Generator().manual_seed(1234)
        s.q = nn.Linear(D, 2 * s.HALF, bias=False); s.ln = nn.LayerNorm(2 * s.HALF)
        s.register_buffer("keys", torch.randn(2, s.NSUB, s.HALF, generator=g) / math.sqrt(s.HALF)); s.values = nn.Embedding(s.NSUB ** 2, D); nn.init.zeros_(s.values.weight)
    def forward(s, x, ids):
        sh = x.shape[:-1]; q = s.ln(s.q(x.reshape(-1, D))).view(-1, 2, s.HALF)
        s1 = q[:, 0] @ s.keys[0].t(); s2 = q[:, 1] @ s.keys[1].t(); v1, i1 = s1.topk(s.KTOP, -1); v2, i2 = s2.topk(s.KTOP, -1)
        cand = (v1[:, :, None] + v2[:, None, :]).view(-1, s.KTOP ** 2); ci = (i1[:, :, None] * s.NSUB + i2[:, None, :]).view(-1, s.KTOP ** 2)
        sc, j = cand.topk(s.KTOP, -1); idx = ci.gather(1, j); w = F.relu(sc) / s.KTOP; s.last_idx = idx.view(*sh, -1)
        return (s.values(idx) * w[..., None]).sum(1).view(*sh, D)
class HashMem(nn.Module):     # Engram-type: n-gram hash (n=1,2,3) -> row, gated by hidden state
    def __init__(s):
        super().__init__(); s.vals = nn.ModuleList([nn.Embedding(ROWS, D) for _ in range(3)]); [nn.init.zeros_(v.weight) for v in s.vals]
        s.gate = nn.Linear(D, 3); s.mults = [[7919], [7919, 104729], [7919, 104729, 1299709]]
    def forward(s, x, ids):
        g = torch.sigmoid(s.gate(x)); out = 0; sel = []
        for n in range(3):
            h = ihash(ids, s.mults[n], ROWS); out = out + g[..., n:n + 1] * s.vals[n](h); sel.append(h + n * ROWS)
        s.last_idx = torch.stack(sel, -1); return out
class HashCtxMem(nn.Module):  # bigram hash picks a bucket of 64 rows; context query picks top-8 inside the bucket (softmax)
    NB, BW, HALF, K = 256, 64, 64, 8
    def __init__(s):
        super().__init__(); g = torch.Generator().manual_seed(1234)
        s.q = nn.Linear(D, s.HALF, bias=False); s.ln = nn.LayerNorm(s.HALF)
        s.register_buffer("keys", torch.randn(s.NB, s.BW, s.HALF, generator=g) / math.sqrt(s.HALF)); s.values = nn.Embedding(s.NB * s.BW, D); nn.init.zeros_(s.values.weight)
    def forward(s, x, ids):
        sh = x.shape[:-1]; b = ihash(ids, [7919, 104729], s.NB).reshape(-1); q = s.ln(s.q(x.reshape(-1, D)))
        sc = torch.einsum("nh,nbh->nb", q, s.keys[b]); v, j = sc.topk(s.K, -1); w = torch.softmax(v, -1); idx = b[:, None] * s.BW + j
        s.last_idx = idx.view(*sh, -1); return (s.values(idx) * w[..., None]).sum(1).view(*sh, D)
class HashBagMem(nn.Module):  # bag of the last 8 tokens: one unigram-hash table, each offset gated by the hidden state (position-invariant -> item token reachable from any template)
    W = int(E("HB_W", "8"))
    def __init__(s):
        super().__init__(); s.vals = nn.Embedding(ROWS, D); nn.init.zeros_(s.vals.weight); s.gate = nn.Linear(D, s.W)
    def forward(s, x, ids):
        g = torch.sigmoid(s.gate(x)); out = 0; sel = []
        for k in range(s.W):
            sh = F.pad(ids, (k, 0))[:, :ids.shape[1]] if k else ids; h = (sh * 7919) % ROWS
            v = s.vals(h); out = out + g[..., k:k + 1] * (v * IDF_W[sh][..., None].to(v.dtype) if IDF else v); sel.append(h)
        s.last_idx = torch.stack(sel, -1); return out
class ShiftMem(nn.Module):   # CONTROL for hashbag: same 8-offset gated bag, but values come from a tiny 32-d token embedding + per-offset linear (0.2M params) instead of an 8.4M table.
    W = 8                     # if it matches hashbag on reverse/sort, the gain is a routing primitive (token-at-offset path), not table capacity
    def __init__(s):
        super().__init__(); s.e = nn.Embedding(VOCAB, 32); nn.init.normal_(s.e.weight, std=0.02); s.p = nn.ModuleList([nn.Linear(32, D, bias=False) for _ in range(s.W)]); s.gate = nn.Linear(D, s.W)
    def forward(s, x, ids):
        g = torch.sigmoid(s.gate(x)); out = 0
        for k in range(s.W):
            sh = F.pad(ids, (k, 0))[:, :ids.shape[1]] if k else ids; out = out + g[..., k:k + 1] * s.p[k](s.e(sh))
        s.last_idx = ids[..., None]; return out
class HashBagWide(HashBagMem):    # hashbag whose rows are 4D wide (for MEMMODE=win: one gate per FFN hidden unit)
    def __init__(s):
        nn.Module.__init__(s); s.vals = nn.Embedding(ROWS, 4 * D); nn.init.zeros_(s.vals.weight); s.gate = nn.Linear(D, s.W)
class HashOrdMem(HashBagMem):   # ablation: offset-specific rows (row = hash(token, offset)) -> position-SPECIFIC addressing, same table size
    def forward(s, x, ids):
        g = torch.sigmoid(s.gate(x)); out = 0; sel = []
        for k in range(s.W):
            sh = F.pad(ids, (k, 0))[:, :ids.shape[1]] if k else ids; h = (sh * 7919 + k * 104729) % ROWS
            out = out + g[..., k:k + 1] * s.vals(h); sel.append(h)
        s.last_idx = torch.stack(sel, -1); return out
class HashBagBiMem(HashBagMem):  # ablation: bag of 8 unigrams + the last bigram, same table (rows shared), gate has 9 outputs
    def __init__(s): super().__init__(); s.gate = nn.Linear(D, s.W + 1)
    def forward(s, x, ids):
        g = torch.sigmoid(s.gate(x)); out = 0; sel = []
        for k in range(s.W):
            sh = F.pad(ids, (k, 0))[:, :ids.shape[1]] if k else ids; h = (sh * 7919) % ROWS; v = s.vals(h); out = out + g[..., k:k + 1] * (v * IDF_W[sh][..., None].to(v.dtype) if IDF else v); sel.append(h)
        hb = ihash(ids, [7919, 104729], ROWS); out = out + g[..., s.W:s.W + 1] * s.vals(hb); sel.append(hb)
        s.last_idx = torch.stack(sel, -1); return out
class HashMoEMem(nn.Module):   # hash-routed MoE (Hash Layers, Roller 2021) = "table whose rows are tiny FFNs": expert e = hash(token at offset k), gated per offset; parameter-matched to hashbag (64 experts x 131k)
    W, NE, DH = 8, int(E("NE", "16")), int(E("DH", "1024"))
    def __init__(s):
        super().__init__(); s.vals = nn.ModuleList([nn.Sequential(nn.Linear(D, s.DH), nn.GELU(), nn.Linear(s.DH, D)) for _ in range(s.NE)]); s.gate = nn.Linear(D, s.W)
        for e in s.vals: nn.init.zeros_(e[2].weight); nn.init.zeros_(e[2].bias)   # start as a no-op like zero-init table values
    def forward(s, x, ids):
        sh = x.shape[:-1]; xf = x.reshape(-1, D); g = torch.sigmoid(s.gate(xf)); out = torch.zeros_like(xf); sel = []
        for k in range(s.W):
            tk = (F.pad(ids, (k, 0))[:, :ids.shape[1]] if k else ids).reshape(-1); e_id = (tk * 7919) % s.NE; sel.append(e_id)
            for e in range(s.NE):
                m = (e_id == e).nonzero(as_tuple=True)[0]
                if m.numel(): out.index_add_(0, m, g[m, k:k + 1] * s.vals[e](xf[m]))
        s.last_idx = torch.stack(sel, -1).view(*sh, -1); return out.view(*sh, D)
class MixMem(nn.Module):   # mixed key: hashbag rows (isolation, local reach) + ctx rows (hidden-state addressing, long reach); both tables written in MODE=write
    def __init__(s): super().__init__(); s.hb = HashBagMem(); s.cx = CtxMem()
    def forward(s, x, ids):
        o = s.hb(x, ids) + s.cx(x, ids); s.last_idx = torch.cat([s.hb.last_idx, s.cx.last_idx + 10 ** 6], -1); return o
class HashWordMem(HashBagMem):   # tokenizer-aware keys: the address of a position is the hash of its WORD so far (tokens merged until the next 'Ġ'-initial token; run capped at 4), then bag over the last W positions
    def __init__(s):
        super().__init__()
        from transformers import AutoTokenizer as _AT2
        tk = _AT2.from_pretrained("LiquidAI/LFM2.5-350M-Base"); toks = tk.convert_ids_to_tokens(list(range(VOCAB)))
        st = torch.tensor([1 if (t is None or t.startswith("Ġ") or not t.isascii()) else 0 for t in toks], dtype=torch.int64)   # non-ASCII (e.g. Japanese byte pieces) -> every token starts a unit
        s.register_buffer("is_start", st)
    def word_ids(s, ids):   # rolling hash within a word, reset at word starts or after 4 tokens
        B, T_ = ids.shape; out = torch.empty_like(ids); h = torch.zeros(B, dtype=torch.int64, device=ids.device); run = torch.zeros(B, dtype=torch.int64, device=ids.device)
        for t in range(T_):
            tok = ids[:, t]; start = (s.is_start[tok] == 1) | (run >= 4)
            h = torch.where(start, tok * 7919 % ROWS, (h * 31 + tok * 7919) % ROWS); run = torch.where(start, torch.ones_like(run), run + 1); out[:, t] = h
        return out
    def forward(s, x, ids): return super().forward(x, s.word_ids(ids))
class HashExtMem(HashBagMem):   # keys from an EXTERNAL segmenter: decode the core's tokens, re-segment with the sarashina2.2 tokenizer (word-level Japanese), and give every core token the hash of the sarashina unit that covers it
    def __init__(s):
        super().__init__()
        from transformers import AutoTokenizer as _AT3
        s.core_tok = _AT3.from_pretrained("LiquidAI/LFM2.5-350M-Base"); s.ext_tok = _AT3.from_pretrained("sbintuitions/sarashina2.2-0.5b"); s.cache = {}
    def ext_ids(s, ids):
        out = torch.empty_like(ids); idl = ids.tolist()
        for b, row in enumerate(idl):
            key = hash(tuple(row))
            if key in s.cache: out[b] = s.cache[key].to(ids.device); continue
            text = s.core_tok.decode(row, skip_special_tokens=False)
            enc = s.core_tok(text, add_special_tokens=False, return_offsets_mapping=True); offs = enc["offset_mapping"]
            ext = s.ext_tok(text, add_special_tokens=False, return_offsets_mapping=True); eoff = ext["offset_mapping"]; eid = ext["input_ids"]
            # char -> ext token id
            cmap = [0] * (len(text) + 1)
            for tid, (a, z) in zip(eid, eoff):
                for c in range(a, max(z, a + 1)): 
                    if c < len(cmap): cmap[c] = tid
            vals = []
            n = min(len(offs), len(row))
            for i in range(len(row)):
                if i < n and offs[i][1] > offs[i][0]: vals.append(cmap[min(offs[i][0], len(cmap) - 1)])
                else: vals.append(row[i])   # fallback: the core token itself
            v = torch.tensor([(x * 7919) % ROWS for x in vals], dtype=torch.long); s.cache[key] = v; out[b] = v.to(ids.device)
            if len(s.cache) > 4096: s.cache.clear()
        return out
    def forward(s, x, ids): return super().forward(x, s.ext_ids(ids))
MEMS = {"ctx": CtxMem, "hashbag": HashBagMem, "hashword": HashWordMem, "hashext": HashExtMem, "mix": MixMem, "hashbagwide": HashBagWide, "hashmoe": HashMoEMem, "shift": ShiftMem, "hashord": HashOrdMem, "hashbagbi": HashBagBiMem, "hash": HashMem, "hashctx": HashCtxMem, "none": None}

# ---------------- tiny LM ----------------
MEMMODE = E("MEMMODE", "add")   # add | mul | win
class Block(nn.Module):
    def __init__(s, mem, smem=None):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D)
        s.f1, s.f2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D); s.mem = mem; s.smem = smem   # smem: a table shared across depth, added on top of the per-layer table
    def forward(s, x, ids):
        B, T_, _ = x.shape; q, k, v = s.qkv(s.ln1(x)).view(B, T_, 3, H, D // H).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2).reshape(B, T_, D)
        x = x + s.o(a); y = s.ln2(x); h = F.gelu(s.f1(y))
        if s.mem is not None and MEMMODE == "win":
            with torch.autocast(AC, enabled=False): h = h * torch.sigmoid(2.0 * s.mem(y.float(), ids)).to(h.dtype)   # memory picks the active FFN units (parameter window)
        f = s.f2(h)
        if s.mem is not None and MEMMODE == "add":
            with torch.autocast(AC, enabled=False): f = f + s.mem(y.float(), ids).to(f.dtype)
        if s.smem is not None:
            with torch.autocast(AC, enabled=False): f = f + s.smem(y.float(), ids).to(f.dtype)
        if s.mem is not None and MEMMODE == "mul":
            with torch.autocast(AC, enabled=False): f = f * (1.0 + torch.tanh(s.mem(y.float(), ids))).to(f.dtype)   # memory gates the FFN output
        return x + f
class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(VOCAB, D); s.pos = nn.Embedding(T, D); nn.init.normal_(s.emb.weight, std=0.02); nn.init.normal_(s.pos.weight, std=0.02)
        if E("SHARE_MEM", "0") == "1" and MEMS[ARM]:   # "vertical" table: ONE table read at every memory layer (values and gate shared across depth)
            shared = MEMS[ARM](); s.blocks = nn.ModuleList([Block(shared if i in MEM_LAYERS else None) for i in range(NBLK)])
        elif E("SHARE_MEM", "0") == "2" and MEMS[ARM]:   # per-layer tables PLUS one depth-shared table read at every memory layer
            shared = MEMS[ARM](); s.shared_mem = shared; s.blocks = nn.ModuleList([Block(MEMS[ARM]() if i in MEM_LAYERS else None, shared if i in MEM_LAYERS else None) for i in range(NBLK)])
        else:
            s.blocks = nn.ModuleList([Block(MEMS[ARM]() if (MEMS[ARM] and i in MEM_LAYERS) else None) for i in range(NBLK)]) if NBLK > 0 else nn.ModuleList()
        s.tmem = nn.ModuleList([MEMS[ARM]() for _ in range(2)]) if NBLK == 0 and MEMS[ARM] else None; s.lnf = nn.LayerNorm(D)
    def forward(s, ids):   # LOOP>1: the whole 4-block stack is applied LOOP times with shared weights (looped LM, Ouro/CIT-style); tables are re-read every loop
        x = s.emb(ids) + s.pos(torch.arange(ids.shape[1], device=ids.device))
        for _ in range(LOOP):
            for b in s.blocks: x = b(x, ids)
        if s.tmem is not None:
            for m in s.tmem: x = x + m(x.float(), ids).to(x.dtype)
        return s.lnf(x) @ s.emb.weight.t()
    def mems(s):
        base = [b.mem for b in s.blocks if b.mem is not None] + (list(s.tmem) if s.tmem is not None else [])
        if E("SHARE_MEM", "0") == "1": base = base[:1]
        if E("SHARE_MEM", "0") == "2": base = base + [s.shared_mem]
        return base
model = LM().to(dev)
n_core = sum(p.numel() for n, p in model.named_parameters() if ".mem." not in n); n_mem = sum(p.numel() for n, p in model.named_parameters() if ".mem." in n)
print(f"ARM {ARM} LOOP {LOOP} core {n_core/1e6:.1f}M mem {n_mem/1e6:.1f}M", flush=True)

# ---------------- synthetic in-context tasks (no world knowledge needed) ----------------
TASKS = E("TASKS", "0") == "1"; TASK_P = float(E("TASK_P", "0.25"))
if TASKS or True:
    from transformers import AutoTokenizer as _AT
    _tok = _AT.from_pretrained("LiquidAI/LFM2.5-350M-Base")
def gen_task(kind, rng):
    if kind == "kv":      # in-context key-value lookup
        ks = rng.sample([a + b for a in "abcdefghij" for b in "klmnopqrst"], rng.randint(4, 8)); vs = [rng.randint(0, 99) for _ in ks]; i = rng.randrange(len(ks))
        return " ".join(f"{k}={v}" for k, v in zip(ks, vs)) + f" ; {ks[i]}?", f" {vs[i]}"
    if kind == "chain":   # multi-step arithmetic chain mod 100, answer = final value
        x = rng.randint(0, 99); s_ = f"start {x}"; 
        for _ in range(rng.randint(2, 4)):
            op, d = rng.choice("+-"), rng.randint(1, 20); x = (x + d if op == "+" else x - d) % 100; s_ += f" {op}{d}"
        return s_ + " =", f" {x}"
    if kind == "chaincot":  # same arithmetic chain, but the answer span is the whole trace of intermediate values (chain-of-thought supervision)
        x = rng.randint(0, 99); ops = []; vals = []
        for _ in range(rng.randint(2, 4)):
            op, d = rng.choice("+-"), rng.randint(1, 20); x = (x + d if op == "+" else x - d) % 100; ops.append(f"{op}{d}"); vals.append(str(x))
        return f"start {x if False else (x - 0)} ; " if False else f"start {_start(ops, vals)} ; {' '.join(ops)} :", " " + " ".join(vals)
    if kind == "reverse": # reverse a digit list
        ds = [str(rng.randint(0, 9)) for _ in range(rng.randint(4, 8))]
        return "reverse " + " ".join(ds) + " ->", " " + " ".join(reversed(ds))
    if kind == "sort":    # sort small numbers
        ds = [rng.randint(0, 20) for _ in range(rng.randint(4, 7))]
        return "sort " + " ".join(map(str, ds)) + " ->", " " + " ".join(map(str, sorted(ds)))
def _start(ops, vals):
    d = int(ops[0][1:]); v0 = int(vals[0]); return (v0 - d) % 100 if ops[0][0] == "+" else (v0 + d) % 100
KINDS = ["kv", "chain", "reverse", "sort"] + (["chaincot"] if E("COT", "0") == "1" else []); _trng = random.Random(1)
def task_batch(n):   # pack examples into (n, T+1) streams
    out = []
    for _ in range(n):
        ids = []
        while len(ids) < T + 1:
            q, a = gen_task(_trng.choice(KINDS), _trng); ids += _tok(q + a + "\n", add_special_tokens=False).input_ids
        out.append(torch.tensor(ids[:T + 1]))
    return torch.stack(out).to(dev)
@torch.no_grad()
def task_eval(n=200, seed=12345):
    model.eval(); rng = random.Random(seed); r = {}
    for kind in KINDS:
        ok = 0
        for _ in range(n):
            q, a = gen_task(kind, rng); qi = _tok(q, add_special_tokens=False).input_ids; ai = _tok(q + a, add_special_tokens=False).input_ids[len(qi):]
            x = torch.tensor(qi + ai, device=dev)[None]; lg = model(x)[0]; pred = lg[len(qi) - 1:len(qi) - 1 + len(ai)].argmax(-1).tolist(); ok += int(pred == ai)
        r[kind] = round(ok / n, 3)
    model.train(); return r

# ---------------- eval helpers ----------------
@torch.no_grad()
def bpt(name, ntok=300_000):
    model.eval(); b = bins[name]; a = torch.from_numpy(np.asarray(b[len(b) - HOLD:len(b) - HOLD + ntok]).astype(np.int64)); n = (len(a) - 1) // T
    a = a[:n * T + 1]; tot = 0.0
    EB = int(E("EVAL_BS", "16"))   # smaller on memory-tight boxes; same result
    for i in range(0, n, EB):
        x = torch.stack([a[j * T:(j + 1) * T + 1] for j in range(i, min(n, i + EB))]).to(dev)
        with torch.autocast(AC, dtype=ACD): lg = model(x[:, :-1])
        tot += F.cross_entropy(lg.float().reshape(-1, VOCAB), x[:, 1:].reshape(-1), reduction="sum").item(); step_mark()
    model.train(); return round(tot / (n * T) / math.log(2), 4)
def evals(): return {n: bpt(n) for n in names}

# ---------------- train ----------------
mem_p = [p for n, p in model.named_parameters() if ".mem." in n]; core_p = [p for n, p in model.named_parameters() if ".mem." not in n]
opt = torch.optim.AdamW([{"params": core_p, "lr": 1e-3, "weight_decay": 0.1}, {"params": mem_p, "lr": 3e-3, "weight_decay": 0.0}], betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
TAG = f"{ARM}{'_loop' + str(LOOP) if LOOP > 1 else ''}{'_nblk' + str(NBLK) if NBLK != 4 else ''}{'_d' + str(D) if D != 256 else ''}{'_' + MEMMODE if MEMMODE != 'add' else ''}{'_cot' if E('COT', '0') == '1' else ''}{'_w' + E('HB_W') if E('HB_W') and E('HB_W') != '8' else ''}{'_r' + str(ROWS) if ROWS != 16384 else ''}{'_shared' if E('SHARE_MEM', '0') == '1' else ('_both' if E('SHARE_MEM', '0') == '2' else '')}{'_diag' + E('DIAG') if E('DIAG') and E('DIAG') != '0' else ''}{'_idf' if E('IDF', '0') == '1' else ''}{'_L' + E('MEM_LAYERS').replace(',', '') if E('MEM_LAYERS') and E('MEM_LAYERS') != '1,3' else ''}"
MODE = E("MODE", "train")
def load_ckpt(path):   # checkpoints saved before the f1/f2 split used blocks.N.ffn.0 / ffn.2
    sd = torch.load(path, map_location=dev); sd = {k.replace(".ffn.0.", ".f1.").replace(".ffn.2.", ".f2."): v for k, v in sd.items()}; model.load_state_dict(sd)
if MODE == "grow":   # hash-table growth without dilution: double the rows, new rows zero-init, old tokens keep their rows (new rows only receive tokens that COLLIDED before)
    load_ckpt(f"{OUT}/{TAG}.pt" if os.path.exists(f"{OUT}/{TAG}.pt") else f"{OUT}/{ARM}.pt"); model.eval()
    before = {n: bpt(n) for n in names}; print("GROW before", json.dumps(before), flush=True)
    for m in model.mems():
        old = m.vals.weight.data; new = torch.zeros(2 * old.shape[0], old.shape[1], device=dev); new[:old.shape[0]] = old; m.vals = nn.Embedding.from_pretrained(new, freeze=False)
    ROWS2 = 2 * ROWS
    def fwd_grow(self, x, ids):   # tokens whose (tok*7919) // ROWS is odd move to the new (zero) region: for them the read changes from a shared row to an empty row
        g = torch.sigmoid(self.gate(x)); out = 0; sel = []
        for k in range(self.W):
            sh = F.pad(ids, (k, 0))[:, :ids.shape[1]] if k else ids; h = (sh * 7919) % ROWS2
            out = out + g[..., k:k + 1] * self.vals(h); sel.append(h)
        self.last_idx = torch.stack(sel, -1); return out
    for m in model.mems(): m.forward = fwd_grow.__get__(m)
    after = {n: bpt(n) for n in names}; print("GROW after(split, moved tokens read zero rows)", json.dumps(after), flush=True)
    for m in model.mems():   # alternative growth that keeps EVERY old read identical: copy old row into both halves (row r and r+ROWS)
        w = m.vals.weight.data; w[ROWS:] = w[:ROWS]
    after2 = {n: bpt(n) for n in names}; print("GROW after(copy, all old reads identical)", json.dumps(after2), flush=True)
    print("GROW_RESULT", json.dumps({"arm": ARM, "before": before, "split": after, "copy": after2}), flush=True); raise SystemExit
if MODE == "write":
    _ck = f"{OUT}/{TAG}.pt" if os.path.exists(f"{OUT}/{TAG}.pt") else f"{OUT}/{ARM}.pt"; load_ckpt(_ck); STEPS = 0; print("loaded", _ck, flush=True)
scaler = torch.amp.GradScaler(enabled=not TPU); t0 = time.time(); log = open(f"{OUT}/{TAG}{'_write' if MODE == 'write' else ''}.jsonl", "w")
for step in range(STEPS + 1):
    if MODE == "write": break
    if step % 1000 == 0 or step == STEPS:
        r = {"step": step, **evals(), "tasks": task_eval() if TASKS else None, "s": round(time.time() - t0)}; print("EVAL", json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
        if step == STEPS: break
    x = task_batch(BS) if (TASKS and random.random() < TASK_P) else batch(BS)
    opt.zero_grad(set_to_none=True)
    for xm in x.chunk(ACC):   # ACC>1: gradient accumulation over micro-batches (same tokens/step, less memory; for 12 GB boxes)
        with torch.autocast(AC, dtype=ACD): lg = model(xm[:, :-1])
        loss = F.cross_entropy(lg.float().reshape(-1, VOCAB), xm[:, 1:].reshape(-1)); scaler.scale(loss / ACC).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); step_mark()
    if step % 100 == 0: print(f"{step} {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
if MODE != "write": torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, f"{OUT}/{TAG}.pt")
if ARM == "none" or TASKS: raise SystemExit

# ---------------- addressing diagnostic + write test ----------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M-Base")
cands = ["red", "blue", "green", "black", "white", "yellow", "brown", "pink", "gray", "orange", "purple"]
colors = [c for c in cands if len(tok(" " + c, add_special_tokens=False).input_ids) == 1][:8]; cid = torch.tensor([tok(" " + c, add_special_tokens=False).input_ids[0] for c in colors], device=dev)
assert len(colors) == 8, colors
TR = ["Item {n} is {c}", "The color of item {n} is {c}"]; TE = ["Item {n}: {c}", "What is item {n}? It is {c}"]
NI = int(E("NI", "32")); facts = {n: random.randrange(8) for n in range(1, NI + 1)}
def enc(tpl, n, c):   # returns ids (1, len) of the prefix up to (not including) the color token, and the color token id
    pre = tok(tpl.split(" {c}")[0].format(n=n) if " {c}" in tpl else tpl.format(n=n, c="").rstrip(), add_special_tokens=False).input_ids
    full = tok(tpl.format(n=n, c=c), add_special_tokens=False).input_ids
    assert full[:len(pre)] == pre and full[len(pre)] == cid[colors.index(c)].item(), (tpl, n, c, pre, full)
    return torch.tensor(full, device=dev)[None], len(pre) - 1
@torch.no_grad()
def rows(tpl, n):   # union of selected rows (all mem layers) at the position predicting the color
    ids, p = enc(tpl, n, colors[0]); model.eval(); model(ids); model.train()
    s = set()
    for li, m in enumerate(model.mems()): s |= {li * 10 ** 7 + int(i) for i in m.last_idx[0, p].tolist()}
    return s
def jac(a, b): return len(a & b) / max(1, len(a | b))
same_item, diff_item = [], []
for n in range(1, 17):
    r = [rows(t, n) for t in TR + TE]
    same_item += [jac(r[i], r[j]) for i in range(4) for j in range(i + 1, 4)]
    diff_item += [jac(rows(TR[0], n), rows(TR[0], m)) for m in range(1, 17) if m != n][:4]
addr = {"same_item_diff_tpl": round(float(np.mean(same_item)), 3), "diff_item_same_tpl": round(float(np.mean(diff_item)), 3)}
print("ADDR", json.dumps(addr), flush=True)
@torch.no_grad()
def acc(tpls):   # per-template 8-way accuracy (chance 0.125)
    model.eval(); r = {}
    for t in tpls:
        ok = 0
        for n, ci in facts.items():
            ids, p = enc(t, n, colors[ci]); lg = model(ids)[0, p, cid]; ok += int(lg.argmax().item() == ci)
        r[t] = round(ok / len(facts), 3)
    model.train(); return r
base = {"acc_train_tpl": acc(TR), "acc_test_tpl": acc(TE), "ja": bpt("fineweb2_ja", 100_000)}
for p in model.parameters(): p.requires_grad_(False)
vals = [p for n, p in model.named_parameters() if ".mem." in n and "values" in n or (".mem." in n and "vals" in n)]
for p in vals: p.requires_grad_(True)
wopt = torch.optim.Adam(vals, lr=float(E("WLR", "1e-2"))); items = [(n, t) for n in facts for t in TR]
for it in range(50):
    random.shuffle(items); tl = 0
    for n, t in items:
        ids, p = enc(t, n, colors[facts[n]]); lg = model(ids)[0, p]; loss = F.cross_entropy(lg[None], cid[facts[n]][None]); wopt.zero_grad(); loss.backward(); wopt.step(); tl += loss.item()
    if it % 10 == 0: print(f"write {it} {tl/len(items):.3f}", flush=True)
after = {"acc_train_tpl": acc(TR), "acc_test_tpl": acc(TE), "ja": bpt("fineweb2_ja", 100_000)}
res = {"arm": ARM, "addr": addr, "before": base, "after": after, "damage_ja": round(after["ja"] - base["ja"], 4)}
print("RESULT", json.dumps(res), flush=True); log.write(json.dumps(res) + "\n"); log.flush()
if MODE == "write":
    FILL = "which was listed in the catalog and later reviewed by the committee for several weeks before anyone noticed the matter again".split()
    dist = {}
    for d in [0, 2, 4, 8, 16, 32]:
        tpl = "Item {n}" + (" " + " ".join((FILL * 4)[:d]) if d else "") + " is {c}"; dist[d] = acc(["Item {n} is {c}"] if d == 0 else [tpl])[tpl if d else "Item {n} is {c}"]
    print("DIST", json.dumps({"arm": ARM, "W": getattr(model.mems()[0], "W", None), "acc_by_distance": dist}), flush=True); log.write(json.dumps({"dist": dist}) + "\n"); log.flush()
    # ---- background row usage (held-out ja 100k tokens) -> freeze the top-p% most used rows during writes (Sparse Memory Finetuning idea)
    base_sd = {n: p_.detach().clone() for n, p_ in model.named_parameters() if ".mem." in n and ("values" in n or "vals" in n)}
    def restore():
        with torch.no_grad():
            for n, p_ in model.named_parameters():
                if n in base_sd: p_.copy_(base_sd[n])
    @torch.no_grad()
    def row_usage():
        model.eval(); b = bins["fineweb2_ja"]; a = torch.from_numpy(np.asarray(b[len(b) - HOLD:len(b) - HOLD + 100_000]).astype(np.int64)); cnt = [None] * len(model.mems())
        for i in range(0, len(a) - T - 1, T * 16):
            x = torch.stack([a[j:j + T] for j in range(i, min(i + T * 16, len(a) - T), T)]).to(dev); model(x)
            for li, m in enumerate(model.mems()):
                idx = m.last_idx.reshape(-1); nrow = sum(v.weight.shape[0] for v in (m.vals if hasattr(m, "vals") and isinstance(m.vals, nn.ModuleList) else [m.vals if hasattr(m, "vals") else m.values]))
                c = torch.bincount(idx % nrow, minlength=nrow).float(); cnt[li] = c if cnt[li] is None else cnt[li] + c
        model.train(); return cnt
    usage = None if ARM == "hashmoe" else row_usage()
    def write_facts(fs, freeze_p):
        restore(); masks = []
        for li, m in enumerate(model.mems() if usage is not None else []):
            vw = m.values.weight if hasattr(m, "values") else m.vals.weight; k = int(freeze_p * usage[li].numel())
            fz = torch.zeros(usage[li].numel(), dtype=torch.bool, device=dev)
            if k > 0: fz[usage[li].topk(k).indices] = True
            masks.append(fz)
        model._masks = masks
        wo = torch.optim.Adam(vals, lr=float(E("WLR", "1e-2"))); its = [(n, t) for n in fs for t in TR]
        for it in range(50):
            random.shuffle(its)
            for n, t in its:
                ids, p = enc(t, n, colors[fs[n]]); lg = model(ids)[0, p]; loss = F.cross_entropy(lg[None], cid[fs[n]][None]); wo.zero_grad(); loss.backward()
                for li, m in enumerate(model.mems() if usage is not None else []):
                    vw = m.values.weight if hasattr(m, "values") else m.vals.weight; vw.grad[model._masks[li]] = 0
                wo.step()
        return {n: p_.detach().clone() - base_sd[n] for n, p_ in model.named_parameters() if n in base_sd}
    @torch.no_grad()
    def acc_facts(fs, tpls):
        model.eval(); r = {}
        for t in tpls:
            ok = 0
            for n, ci in fs.items(): ids, p = enc(t, n, colors[ci]); ok += int(model(ids)[0, p, cid].argmax().item() == ci)
            r[t] = round(ok / len(fs), 3)
        model.train(); return r
    def apply(deltas):
        restore()
        with torch.no_grad():
            for n, p_ in model.named_parameters():
                if n in base_sd:
                    for d in deltas: p_.add_(d[n])
    A = {n: facts[n] for n in list(facts)[:16]}; B = {n: facts[n] for n in list(facts)[16:]}
    for fp in ([0.0] if usage is None else [0.0, 0.01, 0.05, 0.2]):
        dA = write_facts(A, fp); apply([dA]); rA = {"train": acc_facts(A, TR), "test": acc_facts(A, TE), "ja": bpt("fineweb2_ja", 100_000)}
        dB = write_facts(B, fp); apply([dB]); rB = {"train": acc_facts(B, TR), "test": acc_facts(B, TE)}
        apply([dA, dB]); rM = {"A_train": acc_facts(A, TR), "B_train": acc_facts(B, TR), "A_test": acc_facts(A, TE), "B_test": acc_facts(B, TE), "ja": bpt("fineweb2_ja", 100_000)}
        out = {"arm": ARM, "freeze_p": fp, "A_alone": rA, "B_alone": rB, "merged": rM, "damage_A": round(rA["ja"] - base["ja"], 4), "damage_merged": round(rM["ja"] - base["ja"], 4)}
        print("WRITE", json.dumps(out), flush=True); log.write(json.dumps(out) + "\n"); log.flush()
    restore()
log.close()
