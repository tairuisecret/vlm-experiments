"""严格口径 vs 宽松口径：拆分"书写约定差异"与"真实识别错误"。

宽松口径额外抹平：间距命令、\\left\\right、\\big/\\Big 系列、字体命令、单字符外层花括号、省略号等价形式。
两种口径下 CER 的差值，即为可归因于"书写约定不一致"的部分。
"""
import json
import os
import re
import sys

BS = chr(92)        # 字面反斜杠
B2 = chr(92) * 2    # 正则中匹配一个字面反斜杠需要两个

SPACING = re.compile(B2 + r"[,!;: ]|" + B2 + r"quad|" + B2 + r"qquad|" + B2 + r"thinspace")
LEFTRIGHT = re.compile(B2 + r"left|" + B2 + r"right")
BIGS = re.compile(B2 + r"[Bb]i?gg?[lrm]?")
FONT_BRACE = re.compile(B2 + r"(mathrm|mathbf|mathcal|mathit|mathsf|textrm|text|cal|bf|rm|it|sf)" + r"\s*\{([^{}]*)\}")
FONT_BARE = re.compile(B2 + r"(cal|bf|rm|it|sf)(?![a-zA-Z])")
DOTS = re.compile(B2 + r"cdots|" + B2 + r"ldots|" + B2 + r"dots")
SINGLE_BRACE = re.compile(r"\{(\w)\}")


def strict(s):
    s = re.sub(r"^```(latex)?|```$", "", s.strip()).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    return re.sub(r"\s+", "", s)


def lenient(s):
    s = strict(s)
    s = SPACING.sub("", s)
    s = LEFTRIGHT.sub("", s)
    s = BIGS.sub("", s)
    s = FONT_BRACE.sub(r"\2", s)
    s = FONT_BARE.sub("", s)
    s = DOTS.sub("...", s)
    for _ in range(3):
        s = SINGLE_BRACE.sub(r"\1", s)
    return s


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def stat(path, f):
    rows = []
    for line in open(path):
        d = json.loads(line)
        rows.append((d.get("response", ""), d.get("labels") or ""))
    em = sum(f(a) == f(b) for a, b in rows) / len(rows) * 100
    err = sum(lev(f(a), f(b)) for a, b in rows)
    tot = sum(len(f(b)) for _, b in rows)
    return em, err / tot * 100


if __name__ == "__main__":
    files = [("基线", "test_base.jsonl"),
             ("冻结ViT 3ep", "test_frozen3ep.jsonl"),
             ("解冻ViT 3ep", "test_vit3ep.jsonl")]
    print("模型".ljust(14) + "| 严格口径: 整串  CER  | 宽松口径: 整串  CER")
    res = {}
    for tag, fn in files:
        if not os.path.exists(fn):
            print(tag, "缺失")
            continue
        se, sc = stat(fn, strict)
        le, lc = stat(fn, lenient)
        res[tag] = (sc, lc)
        print("%s|  %5.1f%%  %6.2f%%  |  %5.1f%%  %6.2f%%" % (tag.ljust(14), se, sc, le, lc))
    if "基线" in res and "解冻ViT 3ep" in res:
        b, t = res["基线"], res["解冻ViT 3ep"]
        print()
        print("CER 降幅：严格口径 %.2f pt，宽松口径 %.2f pt" % (b[0] - t[0], b[1] - t[1]))
        print("基线 CER 中可归因于书写约定的比例：%.1f%%" % ((b[0] - b[1]) / b[0] * 100))
        print("微调后 CER 中可归因于书写约定的比例：%.1f%%" % ((t[0] - t[1]) / t[0] * 100))
