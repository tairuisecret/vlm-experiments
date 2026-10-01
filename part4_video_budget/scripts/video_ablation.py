"""视频 token 预算消融：抽帧数 / 每帧分辨率 对精度、延迟、token 数的影响。

设计：
  - 变量 1：抽帧数 nframes ∈ {1,2,4,8,16,32}（固定分辨率）
  - 变量 2：每帧最大像素 max_pixels（固定帧数）
  - 任务：MVBench 三个子任务，其中 action_antonym 为"正放/倒放"判别，单帧理论上不可解，
          可作为"时序信息是否真的被用上"的探针。
指标：选择题准确率、视频 token 数、单题延迟、显存峰值。
"""
import argparse
import json
import os
import random
import re
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = "/mnt/SSD1/gzh/video/mvbench"
TASK_VIDEO_DIR = {
    "action_antonym": "ssv2_video/video",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "egocentric_navigation": "vlnqa",
}
LETTERS = "ABCDEFGH"


_INDEX = {}


def _build_index(task):
    """一次性索引该任务视频目录下的所有文件：既按文件名也按相对路径尾部匹配。"""
    top = os.path.join(ROOT, TASK_VIDEO_DIR[task].split("/")[0])
    by_name, by_tail = {}, {}
    for root, _, files in os.walk(top):
        for f in files:
            if not f.lower().endswith((".mp4", ".webm", ".avi", ".mkv", ".mov")):
                continue
            full = os.path.join(root, f)
            by_name.setdefault(f, full)
            parent = os.path.basename(root)
            by_tail.setdefault(parent + "/" + f, full)
    return by_name, by_tail


def find_video(task, name):
    if task not in _INDEX:
        _INDEX[task] = _build_index(task)
    by_name, by_tail = _INDEX[task]
    name = name.replace("\\", "/")
    if name in by_tail:
        return by_tail[name]
    base = os.path.basename(name)
    return by_name.get(base)


_MID_CACHE = "/mnt/SSD1/gzh/video/_midframes"


def extract_mid_frame(video_path):
    """抽取视频中间帧存为图片，用作"无时序信息"的单帧对照。"""
    os.makedirs(_MID_CACHE, exist_ok=True)
    out = os.path.join(_MID_CACHE, os.path.basename(video_path) + ".jpg")
    if not os.path.exists(out):
        import torchvision
        from PIL import Image
        v, _, _ = torchvision.io.read_video(video_path, pts_unit="sec", output_format="THWC")
        Image.fromarray(v[len(v) // 2].numpy()).save(out, quality=95)
    return out


def build_prompt(q):
    opts = "\n".join("(%s) %s" % (LETTERS[i], c) for i, c in enumerate(q["candidates"]))
    return ("%s\n%s\nAnswer with the option's letter from the given choices directly."
            % (q["question"], opts))


def parse_choice(text, n):
    m = re.search(r"[(\[]?([A-H])[)\].:\s]", text.strip() + " ")
    if m:
        i = LETTERS.index(m.group(1))
        return i if i < n else -1
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/SSD1/gzh/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--tasks", nargs="+", default=list(TASK_VIDEO_DIR))
    ap.add_argument("--n_q", type=int, default=100, help="每个任务的题数")
    # frames 中的 1 表示"单帧静态图"对照（不含任何时序信息）
    ap.add_argument("--frames", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--pixels", nargs="+", type=int, default=[], help="固定帧数下扫分辨率（像素数）")
    ap.add_argument("--fixed_frames", type=int, default=8, help="分辨率扫描时固定的帧数")
    ap.add_argument("--base_pixels", type=int, default=128 * 28 * 28)
    ap.add_argument("--out", default="/mnt/SSD1/gzh/video/ablation_results.jsonl")
    args = ap.parse_args()

    print("加载模型 ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()
    video_token_id = model.config.video_token_id
    image_token_id = model.config.image_token_id

    data = {}
    for t in args.tasks:
        qs = json.load(open(os.path.join(ROOT, t + ".json")))
        random.seed(0)
        random.shuffle(qs)
        keep = []
        for q in qs:
            if find_video(t, q["video"]):
                keep.append(q)
            if len(keep) >= args.n_q:
                break
        if keep:
            data[t] = keep
        print("%s: %d 题可用" % (t, len(keep)), flush=True)
    if not data:
        raise SystemExit("没有任何任务找到可用视频，请检查路径")

    # 构建实验网格：(帧数, 像素) 组合
    grid = [(f, args.base_pixels) for f in args.frames]
    grid += [(args.fixed_frames, p) for p in args.pixels]

    fout = open(args.out, "a")
    for nframes, max_pixels in grid:
        for task, qs in data.items():
            correct = ntok = n_clamped = 0
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            for q in qs:
                vp = find_video(task, q["video"])
                if nframes == 1:
                    # 单帧对照：抽中间帧作为静态图片输入，完全不含时序信息
                    frame_path = extract_mid_frame(vp)
                    vis = {"type": "image", "image": "file://" + frame_path,
                           "max_pixels": max_pixels}
                else:
                    vis = {"type": "video", "video": "file://" + vp,
                           "nframes": nframes, "max_pixels": max_pixels,
                           # 低分辨率档位下需同步下调 min_pixels，否则触发断言
                           "min_pixels": min(4 * 28 * 28, max_pixels)}
                msgs = [{"role": "user", "content": [vis,
                    {"type": "text", "text": build_prompt(q)}]}]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                # 部分视频总帧数少于请求帧数，逐级减半重试（记录实际使用的帧数）
                while True:
                    try:
                        imgs, vids, kw = process_vision_info(msgs, return_video_kwargs=True)
                        break
                    except ValueError as e:
                        if "nframes should in interval" not in str(e) or vis.get("nframes", 2) <= 2:
                            raise
                        vis["nframes"] = max(2, vis["nframes"] // 2)
                        n_clamped += 1
                # transformers 5.x 的处理器要求 fps 为标量：空列表剔除，单元素列表拆包
                kw = {k: v for k, v in kw.items() if not (isinstance(v, (list, tuple)) and len(v) == 0)}
                kw = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                      for k, v in kw.items()}
                inputs = processor(text=[text], images=imgs, videos=vids,
                                   return_tensors="pt", padding=True, **kw).to("cuda")
                ntok += int(((inputs.input_ids == video_token_id) | (inputs.input_ids == image_token_id)).sum())
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                ans = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                             skip_special_tokens=True)[0]
                gt = q["candidates"].index(q["answer"])
                if parse_choice(ans, len(q["candidates"])) == gt:
                    correct += 1
            dt = time.time() - t0
            rec = dict(task=task, nframes=nframes, max_pixels=max_pixels, n=len(qs),
                       acc=correct / len(qs) * 100,
                       video_tokens=ntok / len(qs),
                       sec_per_q=dt / len(qs), n_clamped=n_clamped,
                       peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    fout.close()
    print("ABLATION_DONE")


if __name__ == "__main__":
    main()
