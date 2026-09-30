import difflib
import json
import os
import sys

sys.path.insert(0, "/mnt/SSD1/gzh/vlm")
from lenient import lev, strict


def load(path):
    rows = []
    for line in open(path):
        d = json.loads(line)
        rows.append((d.get("response", ""), d.get("labels") or ""))
    return rows


def metrics(rows):
    em = sum(strict(a) == strict(b) for a, b in rows) / len(rows) * 100
    err = sum(lev(strict(a), strict(b)) for a, b in rows)
    tot = sum(len(strict(b)) for _, b in rows)
    return em, err / tot * 100


print("===== 跨分布（OOD，真人手写算术式，n=295）=====")
for tag, fn in [("基线（未微调）", "ood_base.jsonl"),
                ("冻结 ViT 1ep", "ood_frozen1ep.jsonl"),
                ("冻结 ViT 3ep", "ood_frozen3ep.jsonl"),
                ("解冻 ViT 3ep", "ood_vit3ep.jsonl")]:
    if not os.path.exists(fn):
        print("%-16s 待生成" % tag)
        continue
    em, cer = metrics(load(fn))
    print("%-16s 整串 %5.1f%%   CER %7.2f%%" % (tag, em, cer))

print()
print("===== 标签长度对比（混淆因素检查）=====")
for tag, fn in [("同分布测试集", "test_base.jsonl"), ("跨分布 OOD", "ood_base.jsonl")]:
    if not os.path.exists(fn):
        continue
    L = [len(strict(b)) for _, b in load(fn)]
    L.sort()
    print("%-14s 均值 %5.1f  中位数 %4d  最短 %3d  最长 %4d"
          % (tag, sum(L) / len(L), L[len(L) // 2], L[0], L[-1]))

print()
print("===== 控制长度：同分布测试集中最短的 25% 公式 =====")
if os.path.exists("test_base.jsonl") and os.path.exists("test_vit3ep.jsonl"):
    rb, rt = load("test_base.jsonl"), load("test_vit3ep.jsonl")
    idx = sorted(range(len(rb)), key=lambda i: len(strict(rb[i][1])))[:len(rb) // 4]
    sb = [rb[i] for i in idx]
    st = [rt[i] for i in idx]
    eb, cb = metrics(sb)
    et, ct = metrics(st)
    avg = sum(len(strict(b)) for _, b in sb) / len(sb)
    print("短公式子集 n=%d，平均长度 %.1f 字符（与 OOD 长度接近）" % (len(sb), avg))
    print("  基线    : 整串 %5.1f%%  CER %6.2f%%" % (eb, cb))
    print("  最终模型: 整串 %5.1f%%  CER %6.2f%%" % (et, ct))
    print("  → 若此处模型仍优于基线，则 OOD 崩溃主因是视觉分布而非公式长度")
