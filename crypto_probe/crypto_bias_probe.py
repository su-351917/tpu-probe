# crypto_bias_probe.py : "do hash-table LMs find any structure in public keys?" (for fun; expectation = chance unless the key GENERATOR is biased)
# Tasks (hex-digit sequences "N|p" or "X|k"): predict the secret part digit by digit from the public part, held-out keys.
#   rsa_rand  : N = p*q, p,q random primes of B/2 bits                       (should be chance)
#   rsa_weak  : p,q from a biased generator: p = next_prime(r*M + c), c in a tiny residue set mod M=2310, and p == 3 mod 8  (residue fingerprint; N mod M leaks)
#   rsa_close : p,q within 2^(B/4) of each other (Fermat-weak)              (N's high digits ~ p's high digits: leak by construction)
#   ec_rand   : secp256k1, X = x-coord of k*G, predict low 8 hex digits of k (should be chance)
# Model: 4-layer transformer D=256 + hashbag table over the last 8 hex digits (ARM=hashbag|none). Metric: per-digit accuracy on the secret part vs the
# trivial baseline (most frequent digit per position in the training set). Anything above baseline on held-out keys = structure the generator leaked.
import os, json, random, math, time, torch, torch.nn as nn, torch.nn.functional as F
E = os.environ.get; TASK = E("TASK", "rsa_rand"); B = int(E("BITS", "64")); N_TRAIN, N_TEST = int(E("NTRAIN", "60000")), 2000; STEPS = int(E("STEPS", "3000")); ARM = E("ARM", "hashbag")
D, L, H, ROWS, W = 256, 4, 4, 16384, 8; dev = "cuda"; torch.manual_seed(0); random.seed(0)
def is_prime(n):
    if n < 2: return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0: return n == p
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else: return False
    return True
def next_prime(n):
    n |= 1
    while not is_prime(n): n += 2
    return n
def rand_prime(bits): return next_prime(random.getrandbits(bits) | (1 << (bits - 1)) | 1)
def gen_rsa(kind, n):
    out = []; hb = B // 2; M = 2310; cset = [c for c in random.Random(1).sample(range(1, M), 8) if math.gcd(c, M) == 1][:4]
    for _ in range(n):
        if kind == "rsa_rand": p, q = rand_prime(hb), rand_prime(hb)
        elif kind == "rsa_weak":
            def wp():
                while True:
                    r = random.getrandbits(hb - 12) | (1 << (hb - 13)); c = random.choice(cset); x = next_prime(r * M + c)
                    if x % 8 == 3 and x.bit_length() <= hb: return x
            p, q = wp(), wp()
        else:  # rsa_close
            p = rand_prime(hb); q = next_prime(p + random.getrandbits(B // 4))
        if p > q: p, q = q, p
        out.append((format(p * q, f"0{B // 4}x"), format(p, f"0{hb // 4}x")))
    return out
# --- secp256k1 (minimal) ---
P = 2**256 - 2**32 - 977; Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798; Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
def ec_add(a, b):
    if a is None: return b
    if b is None: return a
    (x1, y1), (x2, y2) = a, b
    if x1 == x2 and (y1 + y2) % P == 0: return None
    l = (3 * x1 * x1 * pow(2 * y1, -1, P)) % P if a == b else ((y2 - y1) * pow(x2 - x1, -1, P)) % P
    x3 = (l * l - x1 - x2) % P; return (x3, (l * (x1 - x3) - y1) % P)
def ec_mul(k, pt):
    r = None
    while k:
        if k & 1: r = ec_add(r, pt)
        pt = ec_add(pt, pt); k >>= 1
    return r
def gen_ec(n):
    out = []
    for _ in range(n):
        k = random.getrandbits(256) % (2**256 - 432420386565659656852420866394968145599) or 1; x, _ = ec_mul(k, (Gx, Gy))
        out.append((format(x, "064x"), format(k & 0xffffffff, "08x")))   # secret part: low 32 bits of k as 8 hex digits
    return out
t0 = time.time(); _cache = E("DATA_CACHE")   # DATA_CACHE=dir: reuse the generated keys across arms (EC generation is the slow part)
_cf = f"{_cache}/{TASK}_{B}_{N_TRAIN + N_TEST}.json" if _cache else None
if _cf and os.path.exists(_cf): data = [tuple(r) for r in json.load(open(_cf))]
else:
    data = gen_ec(N_TRAIN + N_TEST) if TASK == "ec_rand" else gen_rsa(TASK, N_TRAIN + N_TEST)
    if _cf: os.makedirs(_cache, exist_ok=True); json.dump(data, open(_cf, "w"))
train, test = data[:N_TRAIN], data[N_TRAIN:]; print("gen", TASK, len(data), f"{time.time() - t0:.0f}s", "example", train[0], flush=True)
VOC = {c: i for i, c in enumerate("0123456789abcdef|")}; SEP = VOC["|"]; V = len(VOC)
def enc(pub, sec): return [VOC[c] for c in pub] + [SEP] + [VOC[c] for c in sec]
seqlen = len(enc(*train[0])); PUBLEN = len(train[0][0])
class HashBag(nn.Module):
    def __init__(s): super().__init__(); s.vals = nn.Embedding(ROWS, D); nn.init.zeros_(s.vals.weight); s.gate = nn.Linear(D, W)
    def forward(s, x, ids):
        g = torch.sigmoid(s.gate(x)); out = 0
        for k in range(W):
            sh = F.pad(ids, (k, 0))[:, :ids.shape[1]] if k else ids; out = out + g[..., k:k + 1] * s.vals((sh * 7919 + k * 104729) % ROWS)   # offset-specific rows: digits are positional
        return out
class Block(nn.Module):
    def __init__(s, mem):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D); s.attn = nn.MultiheadAttention(D, H, batch_first=True); s.f1, s.f2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D); s.mem = mem
    def forward(s, x, ids, mask):
        y = s.ln1(x); x = x + s.attn(y, y, y, attn_mask=mask, need_weights=False)[0]; y = s.ln2(x); f = s.f2(F.gelu(s.f1(y)))
        if s.mem is not None: f = f + s.mem(y, ids)
        return x + f
class LM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(seqlen, D); s.blocks = nn.ModuleList([Block(HashBag() if (ARM == "hashbag" and i in (1, 3)) else None) for i in range(L)]); s.ln = nn.LayerNorm(D); s.head = nn.Linear(D, V)
    def forward(s, ids):
        T_ = ids.shape[1]; mask = torch.triu(torch.ones(T_, T_, device=ids.device, dtype=torch.bool), 1); x = s.emb(ids) + s.pos(torch.arange(T_, device=ids.device))
        for b in s.blocks: x = b(x, ids, mask)
        return s.head(s.ln(x))
model = LM().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1); print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
TR = torch.tensor([enc(*r) for r in train], device=dev); TE = torch.tensor([enc(*r) for r in test], device=dev)
def digit_acc(X):   # accuracy per secret-digit position (teacher forcing on the public part + previous secret digits)
    model.eval(); hits = torch.zeros(X.shape[1] - PUBLEN - 1, device=dev)
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = X[i:i + 256]; lg = model(x[:, :-1]); pred = lg[:, PUBLEN:].argmax(-1); hits += (pred == x[:, PUBLEN + 1:]).float().sum(0)
    model.train(); return (hits / len(X)).tolist()
# trivial baseline: most frequent digit per secret position in TRAIN, scored on TEST
sec_tr = TR[:, PUBLEN + 1:]; mode = torch.stack([torch.bincount(sec_tr[:, j], minlength=V).argmax() for j in range(sec_tr.shape[1])]); base = (TE[:, PUBLEN + 1:] == mode).float().mean(0).tolist()
for step in range(STEPS + 1):
    idx = torch.randint(0, len(TR), (64,), device=dev); x = TR[idx]; lg = model(x[:, :-1])
    loss = F.cross_entropy(lg[:, PUBLEN:].reshape(-1, V), x[:, PUBLEN + 1:].reshape(-1))   # loss only on the secret part
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 500 == 0 or step == STEPS:
        tr, te = digit_acc(TR[:2000]), digit_acc(TE)
        print("EVAL", json.dumps({"task": TASK, "arm": ARM, "bits": B, "step": step, "loss": round(loss.item(), 3), "train_acc_mean": round(sum(tr) / len(tr), 4), "test_acc_mean": round(sum(te) / len(te), 4), "baseline_mean": round(sum(base) / len(base), 4),
                                  "test_minus_baseline_by_pos": [round(a - b, 3) for a, b in zip(te, base)]}), flush=True)
