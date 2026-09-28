"""计算 LaTeX-OCR 的归一化精确匹配率与字符级相似度。

用法: python eval_metrics.py pred1.jsonl [pred2.jsonl ...]
输入为 ms-swift `swift infer --result_path` 产出的 jsonl（含 response / labels 字段）。

两个指标的分工:
  - 归一化精确匹配: 去掉 $$ 包裹、代码块标记与全部空白后整串比对，衡量"完全正确"
  - 字符级相似度:   difflib 编辑距离比率，衡量"错得多离谱"
基线模型常把公式包进 $$...$$，不做归一化会把格式差异误记为识别错误。
"""
import json, re, sys, difflib

def norm(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(latex)?|```$", "", s).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    return re.sub(r"\s+", "", s)

def metrics(path: str):
    rows = []
    for line in open(path):
        d = json.loads(line)
        rows.append((d.get("response", ""), d.get("labels") or ""))
    em = sum(norm(p) == norm(g) for p, g in rows) / len(rows) * 100
    sim = sum(difflib.SequenceMatcher(None, norm(p), norm(g)).ratio() for p, g in rows) / len(rows) * 100
    return len(rows), em, sim

if __name__ == "__main__":
    print("file".ljust(34) + "n".rjust(6) + "EM%".rjust(8) + "SIM%".rjust(8))
    for f in sys.argv[1:]:
        n, em, sim = metrics(f)
        print(f"{f:<34}{n:>6}{em:>8.1f}{sim:>8.2f}")
