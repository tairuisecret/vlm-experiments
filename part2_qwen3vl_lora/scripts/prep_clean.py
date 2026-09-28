import os, json, random
from modelscope import MsDataset
random.seed(0)
OUT="/mnt/SSD1/gzh/vlm/data"; IMG=f"{OUT}/imgs"
os.makedirs(IMG, exist_ok=True)
d = MsDataset.load("AI-ModelScope/LaTeX_OCR", subset_name="synthetic_handwrite", split="train")
seen=set(); rows=[]
for i, r in enumerate(d):
    t=r["text"].strip()
    if t in seen: continue
    seen.add(t); rows.append((i, r["image"], t))
    if len(rows) >= 20500: break
random.shuffle(rows)
val, train = rows[:500], rows[500:]
print("train", len(train), "val", len(val), "公式重叠", len({t for _,_,t in val} & {t for _,_,t in train}))
def dump(split, data):
    with open(f"{OUT}/{split}.jsonl","w") as f:
        for i, img, t in data:
            p=f"{IMG}/{split}_{i}.png"
            if not os.path.exists(p): img.convert("RGB").save(p)
            f.write(json.dumps({"images":[p],"messages":[
                {"role":"user","content":"<image>Using LaTeX to perform OCR on the image."},
                {"role":"assistant","content":t}]}, ensure_ascii=False)+"\n")
dump("val", val); dump("train", train)
print("done")
