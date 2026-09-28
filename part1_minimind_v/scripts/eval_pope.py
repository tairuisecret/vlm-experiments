"""POPE 幻觉评测：对 minimind-v 做物体存在性是非题评测。

POPE 通过询问"图中是否存在某物体"来量化幻觉：
  - random:      随机采样不存在的物体
  - popular:     采样数据集中高频出现的物体（更易诱发幻觉）
  - adversarial: 采样与图中物体常共现的物体（最易诱发幻觉）
指标：准确率、精确率、召回率、F1，以及 yes 占比（模型倾向回答"是"的程度）。
"""
import argparse, io, os, sys, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig

def init_model(weight, device):
    tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
    model = MiniMindVLM(VLMConfig(hidden_size=768, num_hidden_layers=8),
                        vision_model_path="./model/siglip2-base-p32-256-ve")
    sd = torch.load("./out/%s_768.pth" % weight, map_location=device)
    model.load_state_dict({k: v for k, v in sd.items() if "mask" not in k}, strict=False)
    model = model.eval().half().to(device)
    return model, tokenizer, model.processor

YES = ("yes", "是", "有", "存在")
NO  = ("no", "否", "没有", "不存在")

def parse(ans):
    a = ans.strip().lower()
    for w in NO:
        if a.startswith(w): return "no"
    for w in YES:
        if a.startswith(w): return "yes"
    if any(w in a for w in NO): return "no"
    if any(w in a for w in YES): return "yes"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="sft_vlm")
    ap.add_argument("--n_per_cat", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="logs/pope_results.txt")
    ap.add_argument("--data_dir", default="/data3/gzh/datasets/pope")
    args = ap.parse_args()

    import glob, pyarrow.parquet as pq
    buckets = {}
    for f in sorted(glob.glob(os.path.join(args.data_dir, "*.parquet"))):
        cat = os.path.basename(f).replace(".parquet", "")
        pf = pq.ParquetFile(f)
        rows = []
        for batch in pf.iter_batches(batch_size=64):
            rows.extend(batch.to_pylist())
            if len(rows) >= args.n_per_cat:
                break
        buckets[cat] = rows[:args.n_per_cat]

    model, tokenizer, processor = init_model(args.weight, args.device)
    pad = model.config.image_special_token * model.config.image_token_len

    print("类别与样本数:", {k: len(v) for k, v in buckets.items()}, flush=True)

    lines = []
    for cat, rows in buckets.items():
        tp = fp = tn = fn = unk = 0
        for r in rows:
            img = r["image"]
            if isinstance(img, dict): img = img.get("bytes")
            if isinstance(img, (bytes, bytearray)): img = Image.open(io.BytesIO(img))
            img = img.convert("RGB")
            pix = MiniMindVLM.image2tensor(img, processor)
            pix = {k: v.to(args.device).half() if v.dtype.is_floating_point else v.to(args.device)
                   for k, v in pix.items()} if hasattr(pix, "keys") else pix.to(args.device).half()
            msgs = [{"role": "user", "content": r["question"].replace("<image>", "") + " Please answer yes or no.\n" + pad}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt", truncation=True).to(args.device)
            with torch.no_grad():
                out = model.generate(inputs=enc["input_ids"], attention_mask=enc["attention_mask"],
                                     max_new_tokens=8, do_sample=False, pixel_values=pix,
                                     pad_token_id=tokenizer.pad_token_id)
            ans = tokenizer.decode(out[0][len(enc["input_ids"][0]):], skip_special_tokens=True)
            pred, gold = parse(ans), r["answer"].strip().lower()
            if pred == "unknown": unk += 1; pred = "no"
            if pred == "yes" and gold == "yes": tp += 1
            elif pred == "yes" and gold == "no": fp += 1
            elif pred == "no" and gold == "no": tn += 1
            else: fn += 1
        n = tp + fp + tn + fn
        acc = (tp + tn) / n * 100
        prec = tp / (tp + fp) * 100 if tp + fp else 0.0
        rec = tp / (tp + fn) * 100 if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        yes_ratio = (tp + fp) / n * 100
        line = ("%-13s n=%-4d 准确率 %.1f%%  精确率 %.1f%%  召回率 %.1f%%  F1 %.1f  yes占比 %.1f%%  无法解析 %d"
                % (cat, n, acc, prec, rec, f1, yes_ratio, unk))
        print(line, flush=True); lines.append(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("saved ->", args.out)

if __name__ == "__main__":
    main()
