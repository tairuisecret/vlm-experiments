"""验证集泄漏审计: 检查 ms-swift 自动切分出的训练/验证集是否存在样本重叠。

背景: `dataset_id#N` 在数据量不足 N 时会重复采样补足，且重复发生在切分验证集之前，
      导致同一条样本的副本同时落入训练集与验证集。

用法: python check_leakage.py "AI-ModelScope/LaTeX_OCR:human_handwrite#20000" 0.05
"""
import sys
from swift.dataset import load_dataset

def answers(ds):
    out = []
    for r in ds:
        m = r["messages"]
        out.append(m[-1]["content"] if isinstance(m[-1], dict) else str(m[-1]))
    return out

if __name__ == "__main__":
    spec = sys.argv[1]
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    tr, val = load_dataset([spec], split_dataset_ratio=ratio, seed=42)
    t, v = answers(tr), answers(val)
    inter = set(v) & set(t)
    print(f"训练集 {len(t)} 行 / 去重后 {len(set(t))}")
    print(f"验证集 {len(v)} 行 / 去重后 {len(set(v))}")
    print(f"重叠 {len(inter)} 条 = 验证集的 {len(inter)/len(set(v))*100:.1f}%")
    print("结论:", "存在泄漏" if inter else "无泄漏")
