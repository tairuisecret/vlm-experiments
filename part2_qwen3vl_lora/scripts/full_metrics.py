import json, re, difflib

def norm(s):
    s = s.strip()
    s = re.sub(r"^```(latex)?|```$", "", s).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    return re.sub(r"\s+", "", s)

def levenshtein(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def stats(path):
    rows = [(json.loads(l).get("response", ""), json.loads(l).get("labels") or "") for l in open(path)]
    n = len(rows)
    em = sim = 0
    tot_err = tot_len = 0
    near95 = near90 = 0
    for p, g in rows:
        np_, ng = norm(p), norm(g)
        if np_ == ng: em += 1
        r = difflib.SequenceMatcher(None, np_, ng).ratio()
        sim += r
        if r >= 0.95: near95 += 1
        if r >= 0.90: near90 += 1
        tot_err += levenshtein(np_, ng)
        tot_len += len(ng)
    return dict(n=n, em=em / n * 100, sim=sim / n * 100,
                cer=tot_err / tot_len * 100, acc_char=(1 - tot_err / tot_len) * 100,
                near95=near95 / n * 100, near90=near90 / n * 100)

names = [("基线（未微调）", "clean_base.jsonl"),
         ("LoRA 1 epoch", "clean_r16_best.jsonl"),("ViT解冻 1ep","clean_vit.jsonl"),
         ("LoRA 3 epoch", "clean_r16_3ep_last.jsonl"),
         ("3ep + GRPO", "clean_grpo.jsonl")]
print("模型".ljust(18), "整串正确率", "字符准确率", "字符错误率", "相似度", "≥95%相似", "≥90%相似")
for tag, f in names:
    try: s = stats(f)
    except FileNotFoundError: continue
    print(tag.ljust(16),
          ("%.1f%%" % s["em"]).rjust(9),
          ("%.2f%%" % s["acc_char"]).rjust(10),
          ("%.2f%%" % s["cer"]).rjust(10),
          ("%.2f%%" % s["sim"]).rjust(8),
          ("%.1f%%" % s["near95"]).rjust(8),
          ("%.1f%%" % s["near90"]).rjust(8))
