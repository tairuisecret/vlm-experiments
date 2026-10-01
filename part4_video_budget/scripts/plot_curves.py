import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open("/mnt/SSD1/gzh/video/ablation_results.jsonl") if l.startswith("{")]
TASKS = [("action_antonym", "Action Antonym (temporal)"),
         ("fine_grained_action", "Fine-grained Action"),
         ("egocentric_navigation", "Egocentric Navigation")]
COLORS = ["#2a6fdb", "#d1495b", "#44a08d"]
BASE_PX = 100352

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

# 左：帧数 vs 准确率
ax = axes[0]
for (t, label), c in zip(TASKS, COLORS):
    pts = sorted([(r["nframes"], r["acc"]) for r in rows
                  if r["task"] == t and r["max_pixels"] == BASE_PX])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=c, label=label)
ax.set_xscale("log", base=2)
ax.set_xlabel("frames (1 = single still image)")
ax.set_ylabel("accuracy (%)")
ax.set_title("A. Accuracy vs frame count\n(resolution fixed)", fontsize=10)
ax.grid(alpha=.3)
ax.legend(fontsize=8)

# 中：分辨率 vs 准确率
ax = axes[1]
for (t, label), c in zip(TASKS, COLORS):
    pts = sorted([(r["max_pixels"], r["acc"]) for r in rows
                  if r["task"] == t and r["nframes"] == 8])
    ax.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], "s-", color=c, label=label)
ax.set_xscale("log", base=2)
ax.set_xlabel("max pixels per frame (K)")
ax.set_ylabel("accuracy (%)")
ax.set_title("B. Accuracy vs per-frame resolution\n(8 frames fixed)", fontsize=10)
ax.grid(alpha=.3)

# 右：token 预算 vs 准确率（两种花法对比）
ax = axes[2]
for (t, label), c in zip(TASKS, COLORS):
    fr = sorted([(r["video_tokens"], r["acc"]) for r in rows
                 if r["task"] == t and r["max_pixels"] == BASE_PX])
    px = sorted([(r["video_tokens"], r["acc"]) for r in rows
                 if r["task"] == t and r["nframes"] == 8])
    ax.plot([p[0] for p in fr], [p[1] for p in fr], "o-", color=c, label=label + " (vary frames)")
    ax.plot([p[0] for p in px], [p[1] for p in px], "s--", color=c, alpha=.55,
            label=label + " (vary resolution)")
ax.set_xscale("log", base=2)
ax.set_xlabel("visual tokens per question")
ax.set_ylabel("accuracy (%)")
ax.set_title("C. Same token budget: frames beat resolution", fontsize=10)
ax.grid(alpha=.3)
ax.legend(fontsize=6.5, loc="lower right")

plt.tight_layout()
plt.savefig("/mnt/SSD1/gzh/video/token_budget_curves.png", dpi=140)
print("saved token_budget_curves.png")
