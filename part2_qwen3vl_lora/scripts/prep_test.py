"""构建独立测试集：从训练/验证从未触及的样本中切出 500 条。

原切分取的是去重后的前 20,500 条；本脚本跳过这部分，从第 20,500 条之后取样，
并再次校验与 train/val 的公式文本零重叠。
"""
import os, json, random
from modelscope import MsDataset

random.seed(7)
OUT = "/mnt/SSD1/gzh/vlm/data"
IMG = f"{OUT}/imgs_test"
os.makedirs(IMG, exist_ok=True)

used = set()
for split in ("train", "val"):
    for line in open(f"{OUT}/{split}.jsonl"):
        used.add(json.loads(line)["messages"][-1]["content"].strip())
print("已用公式数:", len(used))

d = MsDataset.load("AI-ModelScope/LaTeX_OCR", subset_name="synthetic_handwrite", split="train")
seen, rows = set(), []
for i, r in enumerate(d):
    t = r["text"].strip()
    if t in used or t in seen:
        continue
    seen.add(t)
    rows.append((i, r["image"], t))
    if len(rows) >= 500:
        break

with open(f"{OUT}/test.jsonl", "w") as f:
    for i, img, t in rows:
        p = f"{IMG}/test_{i}.png"
        if not os.path.exists(p):
            img.convert("RGB").save(p)
        f.write(json.dumps({"images": [p], "messages": [
            {"role": "user", "content": "<image>Using LaTeX to perform OCR on the image."},
            {"role": "assistant", "content": t}]}, ensure_ascii=False) + "\n")

overlap = len({t for _, _, t in rows} & used)
print("测试集 %d 条，与 train/val 重叠 %d 条" % (len(rows), overlap))
