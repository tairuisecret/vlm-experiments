"""GRPO 奖励函数：LaTeX-OCR 的可验证奖励（RLVR）。

设计思路：
  1. 主奖励 latex_sim —— 归一化后的字符级相似度（0~1 连续值）。
     用连续相似度而非 0/1 精确匹配，是因为该任务标签平均 108 字符，
     精确匹配的奖励极度稀疏（基线正确率仅 2%），几乎没有梯度信号；
     而相似度能让"错 30 个字符 → 错 10 个字符"的改进也获得正反馈。
  2. 辅助奖励 latex_format —— 输出是否为纯 LaTeX（不含自然语言解释、代码块包裹）。
     基线模型的主要失败模式之一就是输出 $$...$$ 包裹与多余说明。

注意：奖励与评测指标使用同一套归一化逻辑，保证"优化目标 = 评测目标"。
"""
import re
import difflib
from typing import List

from swift.rewards.orm import ORM


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(latex)?|```$", "", s).strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    return re.sub(r"\s+", "", s)


class LatexSimilarity(ORM):
    """字符级相似度奖励，范围 [0, 1]。"""

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        refs = kwargs.get("solution") or kwargs.get("label") or []
        out = []
        for i, c in enumerate(completions):
            ref = refs[i] if i < len(refs) else ""
            if not ref:
                out.append(0.0)
                continue
            out.append(difflib.SequenceMatcher(None, normalize(c), normalize(ref)).ratio())
        return out


class LatexFormat(ORM):
    """格式奖励：输出应为纯 LaTeX，不带自然语言解释或代码块包裹。"""

    BAD = ("```", "$$", "the image", "this is", "图中", "这张")

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        out = []
        for c in completions:
            low = c.strip().lower()
            penalty = any(b in low for b in self.BAD)
            has_tex = ("\\" in c)
            out.append(0.0 if penalty else (0.2 if has_tex else 0.0))
        return out


# 注册到 swift 的奖励函数表，供 --reward_funcs 按名字引用
from swift.rewards import orms

orms["latex_sim"] = LatexSimilarity
orms["latex_format"] = LatexFormat
