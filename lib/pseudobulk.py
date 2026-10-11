"""pseudobulk.py —— 单细胞 → 供体层级聚合（**防伪重复的第一道闸门**）。

为什么必须有这个模块
--------------------
单细胞数据里，**细胞不是独立样本**。同一个供体的几千个细胞是反复测量，
直接拿细胞当样本做差异检验会把 n 虚增两三个数量级，p 值必然假阳性——
这是单细胞领域最经典、也最常被审稿人抓的统计错误。

正确做法：先把细胞聚合到**供体**（或其他生物学独立单位），
在供体层级做检验。本模块就是这条链的第一环。

**本模块只做聚合，不做检验** —— 聚合完的矩阵交给 `lib.lm.moderated_lm`
或 `lib.compose.*`。分工明确，避免"聚合口径"和"检验口径"互相污染。

口径（两条，必须随数字一起输出）
------------------------------
- ``sum_counts``（**默认，推荐**）：先把每个供体的 **raw counts 相加**，
  再做文库归一化（target_sum）+ log1p。这是标准做法——先加后归一化，
  各细胞按实际深度加权。
- ``mean_log``：对**已 log 化**的表达按供体取均值。只在手上只有 log 数据时
  用它；它是**有偏的**（log 的均值 ≠ 均值的 log，且低表达基因的偏差更大）。
  选了它就必须在结果里带着这个口径报告，不能假装等价。

⚠️ **聚合不解决混杂**：供体层级只解决"伪重复"，不解决批次/年龄共线。
  整合与协变量调整是 `lib.batch` 与调用方设计矩阵的事。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["aggregate", "aggregate_by_group", "donor_summary"]


def aggregate(counts, donor: Sequence[Any],
              gene_names: Optional[Sequence[str]] = None,
              how: str = "sum_counts",
              min_cells: int = 10,
              target_sum: float = 1e4,
              log: bool = True) -> Dict[str, Any]:
    """把 (细胞 × 基因) 矩阵聚合为 (供体 × 基因)。

    :param counts: (细胞 × 基因) 矩阵。**sum_counts 口径下必须是 raw counts**
        （归一化后的值相加没有意义，"计数可加"是假定的前提）
    :param donor: 长度 = 细胞数的供体标签（也可以是样本/克隆等任意独立单位）
    :param gene_names: 基因名，长度 = 列数
    :param how: ``sum_counts``（默认）或 ``mean_log``
    :param min_cells: 低于此细胞数的供体被标记 ``low_cells=True``（**不自动丢弃**——
        丢样本是调用方的决定，本模块只把事实报出来）
    :param target_sum: 文库归一化目标总数（仅 sum_counts + log 时生效）
    :param log: 是否 log1p（默认 True；下游线性模型需要）
    :return: dict
        - ``expr``: DataFrame (供体 × 基因)
        - ``meta``: DataFrame(donor, n_cells, total_counts, low_cells)
        - ``n_cells`` / ``n_donors``
        - ``how`` / ``log`` / ``target_sum``: **口径，必须一起报告**
        - ``n_low_cells_donors``: 细胞数不足的供体数
    """
    C = np.asarray(counts, dtype=float)
    if C.ndim != 2:
        raise ValueError("counts 必须是二维矩阵 (细胞 × 基因)")
    donor = np.asarray(donor)
    if donor.shape[0] != C.shape[0]:
        raise ValueError(
            f"donor 长度 {donor.shape[0]} != 细胞数 {C.shape[0]}；"
            f"标签错位会让每个供体的表达被算成别人的")
    if donor.size == 0:
        raise ValueError("donor 为空")
    if how not in ("sum_counts", "mean_log"):
        raise ValueError(f"how 只能是 sum_counts / mean_log，当前 {how!r}")
    if np.any(~np.isfinite(C)):
        raise ValueError("counts 含 NaN/Inf；缺失值必须在聚合前显式处理，"
                         "这里静默跳过会把「没测到」当成「表达量为 0」")

    n_cells, n_genes = C.shape
    genes = (list(gene_names) if gene_names is not None
             else [f"g{i}" for i in range(n_genes)])
    if len(genes) != n_genes:
        raise ValueError(f"gene_names 长度 {len(genes)} != 基因数 {n_genes}")

    levels = list(pd.unique(donor))
    idx = {d: np.where(donor == d)[0] for d in levels}

    if how == "sum_counts":
        # 先按供体求和（计数可加），再整体归一化 —— 顺序反了口径就变了
        mat = np.vstack([C[i].sum(axis=0) for i in idx.values()])
        totals = mat.sum(axis=1)
        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(totals > 0, target_sum / totals, 0.0)
                mat = np.log1p(mat * scale[:, None])
    else:
        mat = np.vstack([C[i].mean(axis=0) for i in idx.values()])
        totals = np.asarray(C.sum(axis=1))
        totals = np.array([totals[i].sum() for i in idx.values()])

    n_per = np.array([len(idx[d]) for d in levels])
    meta = pd.DataFrame({
        "donor": levels,
        "n_cells": n_per,
        "total_counts": totals,
        "low_cells": n_per < min_cells,
    })
    expr = pd.DataFrame(mat, index=[str(d) for d in levels], columns=genes)

    return {
        "expr": expr,
        "meta": meta,
        "donors": [str(d) for d in levels],
        "n_cells": int(n_cells),
        "n_donors": int(len(levels)),
        "how": how,
        "log": bool(log),
        "target_sum": float(target_sum) if (how == "sum_counts" and log) else None,
        "min_cells": int(min_cells),
        "n_low_cells_donors": int((n_per < min_cells).sum()),
        "engine": "lib.pseudobulk.aggregate",
    }


def aggregate_by_group(counts, donor: Sequence[Any], group: Sequence[Any],
                       gene_names: Optional[Sequence[str]] = None,
                       how: str = "sum_counts",
                       min_cells: int = 10,
                       target_sum: float = 1e4,
                       log: bool = True) -> Dict[str, Any]:
    """按「供体 × 分组」（如 供体 × 细胞类型）二维聚合。

    细胞类型 × 年龄的研究里这是常用形态：一个供体在每个细胞类型下各出一行，
    于是"细胞类型内随年龄的变化"与"供体间差异"可以分开建模。

    ⚠️ **注意这仍然是供体层级的重复测量**：同一供体的多个细胞类型**不独立**，
    后续建模必须把供体当随机效应或做配对设计，不能直接当独立样本。
    """
    donor = np.asarray(donor)
    group = np.asarray(group)
    if group.shape[0] != donor.shape[0]:
        raise ValueError(f"group 长度 {group.shape[0]} != 细胞数 {donor.shape[0]}")

    out_rows, meta_rows = [], []
    keys = []
    for g in pd.unique(group):
        sel = group == g
        res = aggregate(counts[sel] if hasattr(counts, "__getitem__") else np.asarray(counts)[sel],
                        donor[sel], gene_names=gene_names, how=how,
                        min_cells=min_cells, target_sum=target_sum, log=log)
        e = res["expr"].copy()
        e.insert(0, "_donor", res["donors"])
        e.insert(0, "_group", str(g))
        out_rows.append(e)
        m = res["meta"].copy()
        m.insert(0, "group", str(g))
        meta_rows.append(m)
        keys.append(str(g))

    expr = pd.concat(out_rows, axis=0, ignore_index=True)
    meta = pd.concat(meta_rows, axis=0, ignore_index=True)
    return {
        "expr": expr,
        "meta": meta,
        "groups": keys,
        "how": how,
        "log": bool(log),
        "min_cells": int(min_cells),
        "n_rows": int(len(expr)),
        "engine": "lib.pseudobulk.aggregate_by_group",
    }


def donor_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    """聚合结果的事实汇总（报数字，不替调用方做决定）。"""
    m = res["meta"]
    n = m["n_cells"].to_numpy(dtype=float)
    return {
        "n_donors": int(len(m)),
        "n_cells_total": int(n.sum()),
        "cells_per_donor_min": int(n.min()),
        "cells_per_donor_median": float(np.median(n)),
        "cells_per_donor_max": int(n.max()),
        "n_low_cells_donors": int(m["low_cells"].sum()),
        "how": res["how"],
        "log": res["log"],
        "engine": "lib.pseudobulk.donor_summary",
    }
