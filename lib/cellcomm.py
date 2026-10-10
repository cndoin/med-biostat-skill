"""细胞—细胞通讯（配体/受体）分析（包装层）。

算法在 `vendor/_cellcomm/core.py`（移植自 CellPhoneDB，MIT）。本模块只做接口
收敛与口径提示。

**何时用**
- 单细胞数据已有 cluster 注释，想知道「哪些细胞类型之间通过什么配受体在对话」；
- 结合空间数据（`lib.spatial`）判断通讯是否受空间邻近支持。

**口径红线**
1. p 值是**置换细胞标签**得到的经验 p；最小可分辨 p = 1/n_perm，报 p 必报 n_perm。
2. 互作有**方向性**：ligand 在 source、receptor 在 target，两向不可对调。
3. 输入必须是**归一化后**的表达（log1p/CPM），原始 count 会产生测序深度假阳性。
4. `EXAMPLE_LR_PAIRS` 只是**接口演示用的经典对**；生产分析请自供
   CellPhoneDB / CellChatDB 官方配受体表。
5. 结果是「相对富集 + 显著性」，不可跨实验直接比较绝对值。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

from vendor._cellcomm import core as _core
from vendor._cellcomm.core import EXAMPLE_LR_PAIRS  # noqa: F401  再导出

__all__ = ["score", "summary", "pathway_summary", "EXAMPLE_LR_PAIRS"]


def score(expr, labels: Sequence[Any], lr_pairs,
          n_perm: int = 1000, threshold: float = 0.1,
          seed: int = 0, min_cells: int = 10) -> Dict[str, Any]:
    """细胞通讯打分。返回 ``means`` / ``pvalues`` / ``percents`` / ``significant``。"""
    return _core.communication_score(expr, labels, lr_pairs, n_perm=n_perm,
                                     threshold=threshold, seed=seed,
                                     min_cells=min_cells)


def summary(result: Dict[str, Any], top_n: int = 15):
    """把显著互作汇总为「cluster 对 × 互作数」排名表。"""
    return _core.communication_summary(result, top_n=top_n)


def pathway_summary(result: Dict[str, Any], pathways: Dict[str, Iterable[str]]):
    """按通路（如 chemokine / checkpoint）归类汇总。"""
    return _core.pathway_communication(result, pathways)
