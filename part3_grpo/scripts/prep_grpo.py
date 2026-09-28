"""把 SFT 用的数据转成 GRPO 格式：保留 user 轮，把标准答案放进 solution 列。"""
import json, os

SRC = "/mnt/SSD1/gzh/vlm/data"
DST = "/mnt/SSD1/gzh/vlm/data_grpo"
os.makedirs(DST, exist_ok=True)

for split, limit in [("train", 2000), ("val", 200)]:
    n = 0
    with open(f"{SRC}/{split}.jsonl") as fi, open(f"{DST}/{split}.jsonl", "w") as fo:
        for line in fi:
            if n >= limit:
                break
            r = json.loads(line)
            gold = r["messages"][-1]["content"]
            fo.write(json.dumps({
                "images": r["images"],
                "messages": [r["messages"][0]],
                "solution": gold,
            }, ensure_ascii=False) + "\n")
            n += 1
    print(split, n)
