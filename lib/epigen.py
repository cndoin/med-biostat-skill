"""表观遗传 / DNA 甲基化分析（EWAS 核心）。

**方法来源（公开发表的标准方法）**
- beta ⇄ M 值转换：Du P, et al. *BMC Bioinformatics* 2010;11:587
  （M = log2(beta/(1−beta))，方差更稳定，回归应在 M 值上做）
- **moderated t / 经验贝叶斯方差收缩**：Smyth GK. *Stat Appl Genet Mol Biol*
  2004;3:Article3（limma 的核心思想。limma 是 LGPL，**不内嵌**，按论文实现）
- 多重检验：Benjamini–Hochberg（`vendor/_enrich/stats.py` 已有实现）

**红线**
1. **回归必须建在 M 值上，报告在 beta 尺度上**。beta 值在两端（0 与 1 附近）方差被
   压缩，直接对 beta 做 t 检验会系统性低估两端的差异。
2. **delta beta 与 p 值要一起看**。样本量大时，Δβ = 0.01 也能「显著」——
   这在生物学上通常没有意义。EWAS 的通行做法是同时设 Δβ 与 FDR 双阈值。
3. **细胞组成是 EWAS 的头号混杂**。全血甲基化数据里，病例/对照的细胞比例差异
   会制造成千上万个「显著」CpG。本模块提供 `cell_composition_proxy` 做粗筛，
   但它**不能替代** Houseman 参考矩阵法或实测细胞分选。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import lm

__all__ = [
    "beta_to_m", "m_to_beta", "diff_methylation", "dmp_summary",
    "cell_composition_proxy",
]


def beta_to_m(beta, eps: float = 1e-6):
    """beta 值 → M 值：``M = log2(beta/(1−beta))``。

    beta 恰好等于 0 或 1 时对数无定义（这正是实际数据里最常见的情形），
    故先夹到 ``[eps, 1−eps]``，并**要求调用方知道**这件事——被夹过的位点
    与「真 0」不可区分，会在 `diff_methylation` 的结果里带 ``n_clipped`` 计数。
    """
    b = np.clip(np.asarray(beta, dtype=float), eps, 1.0 - eps)
    return np.log2(b / (1.0 - b))


def m_to_beta(m):
    """M 值 → beta 值：``beta = 2^M / (1 + 2^M)``。``beta_to_m`` 的逆运算。"""
    x = np.power(2.0, np.asarray(m, dtype=float))
    return x / (1.0 + x)


# moderated t 引擎已收敛到 `lib.lm`（**唯一裁决点**，与蛋白组共用一套）。
# 历史原因：本文件原先自带 `_fit_f_dist`；当 `lib.proteomics` 也需要同一套
# 方差收缩时，两份实现会让 EWAS 与蛋白组的 p 值**不可比**——而 RNA-蛋白
# 是否同向恰恰依赖两侧 p 值可比。故统一委托，本文件不再保留副本。


def diff_methylation(beta, group: Sequence[Any],
                     covariates: Optional[np.ndarray] = None,
                     use_m: bool = True, moderated: bool = True,
                     eps: float = 1e-6) -> Dict[str, Any]:
    """差异甲基化：逐 CpG 检验两组间的 beta 值差异（EWAS 主分析）。

    :param beta: (样本 × CpG) 的 beta 值矩阵
    :param group: 长度 = 样本数的二分类标签
    :param covariates: 需调整的协变量（**细胞组成一定要放进来**，见模块头红线 3）
    :param use_m: 在 M 值上建模（默认 True，见红线 1）
    :param moderated: 用经验贝叶斯 moderated t（默认 True；小样本强烈建议开）
    :return: dict
        - ``table``: DataFrame(CpG, delta_beta, t, pvalue, padj, n_case, n_ctrl)
        - ``n_clipped``: beta 被夹到 [eps, 1−eps] 的位点数（含 0/1 的原值）
        - ``d0`` / ``s0_sq``: 先验参数（moderated 时给出）
        - ``method`` / ``engine``
    """
    B = np.asarray(beta, dtype=float)
    if B.ndim != 2:
        raise ValueError("beta 必须是二维矩阵 (样本 × CpG)")
    group = np.asarray(group)
    if len(group) != B.shape[0]:
        raise ValueError(f"group 长度 {len(group)} != 样本数 {B.shape[0]}")
    levels = list(pd.unique(group))
    if len(levels) != 2:
        raise ValueError(f"group 必须是二分类，当前有 {len(levels)} 类：{levels}")

    n_clipped = int(np.sum((B <= eps) | (B >= 1.0 - eps)))
    work = beta_to_m(B, eps=eps) if use_m else B

    is_case = (group == levels[1]).astype(float)
    cols = [np.ones(len(is_case)), is_case]
    if covariates is not None:
        if len(covariates) != len(is_case):
            raise ValueError("covariates 行数与样本数不一致")
        cols.append(np.asarray(covariates, dtype=float))
    X = np.column_stack(cols)
    n, p = X.shape
    if n - p <= 0:
        raise ValueError(f"样本数 {n} 不足以拟合 {p} 个参数")

    # 检验引擎收敛到 lib.lm（与 lib.proteomics 共用，保证两侧 p 值可比）
    fit = lm.moderated_lm(work, X, coef_index=1, moderated=moderated)
    pval = fit["pvalue"]
    d0, s0sq, dof_t = fit["d0"], fit["s0_sq"], fit["dof"]

    # delta beta 报告在 beta 尺度（红线 1）：M 尺度系数 → beta 尺度近似差
    grp1, grp0 = (group == levels[1]), (group == levels[0])
    if use_m:
        # 用组均值差作为 beta 尺度的效应量（更直观，且不受 logit 非线性影响）
        delta = np.nanmean(B[grp1], axis=0) - np.nanmean(B[grp0], axis=0)
    else:
        delta = fit["coef"]

    padj = lm.bh_adjust(pval)
    m = B.shape[1]

    table = pd.DataFrame({
        "cpg": [f"cpg{i}" for i in range(m)],
        "delta_beta": delta,
        "t": fit["t"],
        "pvalue": pval,
        "padj": padj,
        "n_case": int(grp1.sum()),
        "n_ctrl": int(grp0.sum()),
    })
    return {
        "table": table,
        "n_clipped": n_clipped,
        "d0": float(d0),
        "s0_sq": float(s0sq) if np.isfinite(s0sq) else float("inf"),
        # 不能用 int()：d0 可以是 inf（各 CpG 方差一致 → 完全收缩），
        # 此时检验自由度是 inf，`int(inf)` 抛 OverflowError。
        # 这个分支在"各特征方差真的一致"时会稳定触发（实测 n=30/σ=0.05 的
        # 模拟数据就命中），不是边角情况。
        "dof": float(dof_t),
        "use_m": bool(use_m),
        "moderated": bool(moderated),
        "method": ("limma-style moderated t (Smyth 2004)" if moderated
                   else "ordinary t-test"),
        "engine": "lib.epigen.diff_methylation",
    }


def dmp_summary(result: Dict[str, Any], delta_cut: float = 0.05,
                fdr_cut: float = 0.05) -> Dict[str, Any]:
    """差异甲基化位点（DMP）汇总。

    **必须同时设 Δβ 与 FDR 两条阈值**（红线 2）：只看 FDR 会把大量
    「统计显著但生物学上毫无意义」的位点算进来（大样本下 Δβ=0.01 也能
    padj<0.05）。
    """
    t = result["table"]
    sig = t[(t["padj"] < fdr_cut) & (t["delta_beta"].abs() >= delta_cut)]
    up = sig[sig["delta_beta"] > 0]
    dn = sig[sig["delta_beta"] < 0]
    return {
        "n_tested": int(len(t)),
        "n_dmp": int(len(sig)),
        "n_hyper": int(len(up)),
        "n_hypo": int(len(dn)),
        "delta_cut": float(delta_cut),
        "fdr_cut": float(fdr_cut),
        "top": sig.reindex(sig["padj"].sort_values().index).head(10),
        "engine": "lib.epigen.dmp_summary",
    }


def cell_composition_proxy(beta, markers: Optional[Dict[str, Sequence[int]]] = None
                           ) -> pd.DataFrame:
    """用「阳性对照探针」估计细胞组成的**粗代理值**。

    ⚠️ **这不是 Houseman 法，不能当作细胞比例使用**。它只是把几组
       细胞特异性探针的均值算出来，用于快速判断「两组之间细胞组成是否差别很大」——
       如果差别大，下游 EWAS 结果必须先把细胞组成放进协变量，否则绝大多数
       「显著 CpG」都是细胞比例差异的副产品。

    :param markers: ``{细胞类型: [探针列索引, ...]}``；不给则用常见阳性对照探针
        的默认位置（若矩阵列数不足会跳过并在结果里标注）
    """
    B = np.asarray(beta, dtype=float)
    if markers is None:
        # 常用的细胞特异性阳性对照探针（列索引需与你的 450K/EPIC 矩阵一致）
        markers = {
            "B_cell": [0, 1, 2],
            "T_cell": [3, 4, 5],
            "monocyte": [6, 7, 8],
            "neutrophil": [9, 10, 11],
            "NK": [12, 13, 14],
        }
    out = {}
    for name, cols in markers.items():
        cols = [c for c in cols if 0 <= c < B.shape[1]]
        if not cols:
            out[name] = np.full(B.shape[0], np.nan)
            continue
        out[name] = B[:, cols].mean(axis=1)
    df = pd.DataFrame(out)
    df.attrs["warning"] = ("细胞组成粗代理值，非 Houseman 估计值；"
                           "仅用于判断是否需要调整细胞组成")
    return df
