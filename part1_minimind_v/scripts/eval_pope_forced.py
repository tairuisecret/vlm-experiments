"""POPE 强制二选一评测：不让模型自由生成，直接比较 "Yes"/"No" 两个 token 的 logit。

动机：生成式评测下，小模型常输出"图中看不清楚"这类含糊回答，
      被判为 no 会人为压低召回率。强制二选一测的是模型的真实倾向，
      把"判断能力"与"表达是否配合指令"两件事分开。
"""
import argparse, io, os, glob, sys, torch
from PIL import Image
import pyarrow.parquet as pq
sys.path.insert(0, "/data3/gzh/project/minimind-v")
os.chdir("/data3/gzh/project/minimind-v")
from eval_pope import init_model
from model.model_vlm import MiniMindVLM

ap = argparse.ArgumentParser()
ap.add_argument("--weight", default="sft_vlm")
ap.add_argument("--n_per_cat", type=int, default=300)
ap.add_argument("--data_dir", default="/data3/gzh/datasets/pope")
ap.add_argument("--out", default="logs/pope_forced.txt")
args = ap.parse_args()

model, tok, proc = init_model(args.weight, "cuda")
pad = model.config.image_special_token * model.config.image_token_len

def ids_of(words):
    out = set()
    for w in words:
        for form in (w, " " + w):
            t = tok(form, add_special_tokens=False).input_ids
            if t: out.add(t[0])
    return sorted(out)

YES_IDS = ids_of(["Yes", "yes", "是", "有"])
NO_IDS = ids_of(["No", "no", "否", "没"])
print("yes token ids:", YES_IDS, "no token ids:", NO_IDS, flush=True)

lines = []
for f in sorted(glob.glob(os.path.join(args.data_dir, "*.parquet"))):
    cat = os.path.basename(f).replace(".parquet", "")
    rows = []
    for b in pq.ParquetFile(f).iter_batches(batch_size=64):
        rows.extend(b.to_pylist())
        if len(rows) >= args.n_per_cat: break
    rows = rows[:args.n_per_cat]

    tp = fp = tn = fn = 0
    for r in rows:
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        pix = MiniMindVLM.image2tensor(img, proc)
        pix = {k: (v.cuda().half() if v.dtype.is_floating_point else v.cuda()) for k, v in pix.items()}
        msgs = [{"role": "user", "content": r["question"] + " Please answer yes or no.\n" + pad}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            logits = model(enc["input_ids"], attention_mask=enc["attention_mask"], pixel_values=pix).logits[0, -1]
        pred = "yes" if logits[YES_IDS].max() > logits[NO_IDS].max() else "no"
        gold = r["answer"].strip().lower()
        if pred == "yes" and gold == "yes": tp += 1
        elif pred == "yes" and gold == "no": fp += 1
        elif pred == "no" and gold == "no": tn += 1
        else: fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n * 100
    prec = tp / (tp + fp) * 100 if tp + fp else 0.0
    rec = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    line = ("%-13s n=%-4d 准确率 %.1f%%  精确率 %.1f%%  召回率 %.1f%%  F1 %.1f  yes占比 %.1f%%"
            % (cat, n, acc, prec, rec, f1, (tp + fp) / n * 100))
    print(line, flush=True); lines.append(line)

open(args.out, "w").write("\n".join(lines) + "\n")
print("saved ->", args.out)
