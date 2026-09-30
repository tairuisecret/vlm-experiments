"""构建跨分布（OOD）测试集：human_handwrite 子集。

当前模型仅在 synthetic_handwrite（合成手写体）上训练，
human_handwrite 为真人手写，渲染方式与训练分布不同，可用于检验
"学到的是真实识别能力，还是仅适配了某一渲染风格"。
"""
import json
import os

from modelscope import MsDataset

OUT = "/mnt/SSD1/gzh/vlm/data"
IMG = OUT + "/imgs_ood"
os.makedirs(IMG, exist_ok=True)

rows = []
seen = set()
for split in ("train", "validation", "test"):
    try:
        d = MsDataset.load("AI-ModelScope/LaTeX_OCR", subset_name="human_handwrite", split=split)
    except Exception:
        continue
    for i, r in enumerate(d):
        t = r["text"].strip()
        if t in seen:
            continue
        seen.add(t)
        rows.append((split, i, r["image"], t))

print("human_handwrite 唯一公式数:", len(rows))
rows = rows[:295]

with open(OUT + "/ood.jsonl", "w") as f:
    for split, i, img, t in rows:
        p = "%s/ood_%s_%d.png" % (IMG, split, i)
        if not os.path.exists(p):
            img.convert("RGB").save(p)
        f.write(json.dumps({"images": [p], "messages": [
            {"role": "user", "content": "<image>Using LaTeX to perform OCR on the image."},
            {"role": "assistant", "content": t}]}, ensure_ascii=False) + "\n")
print("OOD 测试集写入完成:", len(rows), "条")
