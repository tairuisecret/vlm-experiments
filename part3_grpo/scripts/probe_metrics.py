import json, re, difflib, os

def norm(s):
    s = re.sub(r"^```(latex)?|```$", "", s.strip()).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    return re.sub(r"\s+", "", s)

def lev(a, b):
    if len(a) < len(b): a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]

def stat(p):
    rows = [(json.loads(l).get("response",""), json.loads(l).get("labels") or "") for l in open(p)]
    sim = sum(difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() for a,b in rows)/len(rows)
    err = sum(lev(norm(a), norm(b)) for a,b in rows); tot = sum(len(norm(b)) for _,b in rows)
    return sim*100, (1-err/tot)*100, len(rows)

print("数据集   协议      | SFT 相似度  GRPO 相似度  变化   | SFT 字符准确  GRPO 字符准确  变化")
for d, dn in [("train","训练集"), ("val","验证集")]:
    for m, mn in [("greedy","贪心  "), ("sample","采样  ")]:
        fs, fg = f"probe_sft_{d}_{m}.jsonl", f"probe_grpo_{d}_{m}.jsonl"
        if not (os.path.exists(fs) and os.path.exists(fg)):
            print(f"{dn}  {mn} | 缺失文件"); continue
        s1,a1,n = stat(fs); s2,a2,_ = stat(fg)
        print(f"{dn}  {mn} |  {s1:6.2f}%     {s2:6.2f}%   {s2-s1:+5.2f} |   {a1:6.2f}%      {a2:6.2f}%    {a2-a1:+5.2f}   (n={n})")
