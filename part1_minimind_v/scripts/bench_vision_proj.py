import torch, time, sys
MARKER = 12

def original(tokens, h, vf, seqlen):
    if vf.dim() == 3: vf = vf.unsqueeze(1)
    out = []
    for b in range(h.size(0)):
        hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
        while i < len(seq):
            if seq[i] == MARKER:
                start = i
                while i < len(seq) and seq[i] == MARKER: i += 1
                if k < vf.size(1):
                    hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                    k += 1
            else: i += 1
        out.append(hb)
    return torch.stack(out)

def vectorized(tokens, h, vf, seqlen):
    if vf.dim() == 3: vf = vf.unsqueeze(1)
    mask = tokens == MARKER
    counts = mask.sum(dim=1)
    if int(counts.max()) == 0: return h
    flat = vf.reshape(vf.size(0), -1, vf.size(-1))
    n = int(counts[0])
    if bool((counts == counts[0]).all()) and n <= flat.size(1):
        src = flat[:, :n].reshape(-1, h.size(-1)).to(h.dtype)
        return h.masked_scatter(mask.unsqueeze(-1), src)
    return original(tokens, h, vf, seqlen)

def make(bs, seq, dim, ntok, nimg, dev):
    tokens = torch.randint(100, 6000, (bs, seq), device=dev)
    for b in range(bs):
        for j in range(nimg):
            s = 1 + j * (ntok + 5)
            tokens[b, s:s+ntok] = MARKER
    h = torch.randn(bs, seq, dim, device=dev)
    vf = torch.randn(bs, nimg, ntok, dim, device=dev)
    return tokens, h, vf

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
# ---------- 正确性 ----------
print("== 正确性验证 ==")
for bs, seq, ntok, nimg in [(4,450,64,1),(8,768,64,1),(2,512,64,2),(3,300,32,1)]:
    tokens, h, vf = make(bs, seq, 768, ntok, nimg, dev)
    a = original(tokens, h.clone(), vf, seq)
    b_ = vectorized(tokens, h.clone(), vf, seq)
    same = torch.equal(a, b_)
    print(f"  bs={bs} seq={seq} ntok={ntok} nimg={nimg}: 完全一致={same} 最大差异={(a-b_).abs().max().item():.2e}")
# ---------- 性能 ----------
print("== 性能对比（batch 16, seq 450, 64 视觉 token）==")
tokens, h, vf = make(16, 450, 768, 64, 1, dev)
def bench(fn, n=50):
    for _ in range(5): fn(tokens, h.clone(), vf, 450)
    if dev=="cuda": torch.cuda.synchronize()
    t=time.time()
    for _ in range(n): fn(tokens, h.clone(), vf, 450)
    if dev=="cuda": torch.cuda.synchronize()
    return (time.time()-t)/n*1000
o = bench(original); v = bench(vectorized)
print(f"  原始实现   {o:.3f} ms/次")
print(f"  向量化实现 {v:.3f} ms/次")
print(f"  加速 {o/v:.1f}x，单次省 {o-v:.3f} ms")
