"""分析法 ③：WGCNA 加权基因共表达网络分析。

算法来自 `vendor/_wgcna/core.py`（移植自 PyWGCNA，MIT）。本模块只做**包装**：
把表达矩阵 / 基因名 / 性状表转成上游要的形式，并把结果整理成可直接写进论文的格式。

典型用途：从 bulk RNA-seq 找与性状（如白内障晶状体混浊度、眼轴长度）相关的共表达模块，
再从模块里挑 hub gene 做 downstream（富集 / MR / 实验验证）。

⚠️ 口径提醒：模块边界依赖软阈值 power 与聚类切点，**两个参数都必须随结果一起报告**，
   否则别人无法复现（换 power 换 deep_split 就会得到不同的模块）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from vendor._wgcna import core as _core


def wgcna(expr, traits=None, gene_names: Optional[Sequence[str]] = None,
          power: Optional[float] = None, network_type: str = "unsigned",
          tom_type: str = "signed", min_module_size: int = 20,
          merge_cut_height: float = 0.25, deep_split: int = 0,
          n_top_hub: int = 10) -> Dict[str, Any]:
    """一条命令跑完 WGCNA 并给出可直接报告的结果。

    :param expr: 表达矩阵，行=样本、列=基因
    :param traits: 性状表（行=样本）；给定则额外算模块-性状相关
    :return: 含 power / 模块划分 / 模块大小 / ME / kME / hub gene / 模块-性状相关
    """
    e = _core._as_expr(expr)
    if gene_names is not None:
        e.columns = [str(g) for g in gene_names]
    res = _core.detect_modules(
        e, power=power, network_type=network_type, tom_type=tom_type,
        min_module_size=min_module_size, merge_cut_height=merge_cut_height,
        deep_split=deep_split,
    )
    colors = res["colors"]
    out: Dict[str, Any] = {
        "power": res["power"],
        "network_type": network_type,
        "tom_type": tom_type,
        "n_modules": res["n_modules"],
        "module_sizes": res["module_sizes"],
        "colors": colors,
        "genes_by_module": _genes_by_module(colors, list(e.columns)),
        "merged_modules": res["merged_modules"],
        "cluster_rule": res["cluster_rule"],
        "deep_split": deep_split,
        "min_module_size": min_module_size,
    }
    me = res["eigengenes"]
    if me is not None:
        out["eigengenes"] = me["eigengenes"]
        out["var_explained"] = me["var_explained"]
        kme = _core.module_membership(e, colors, me["eigengenes"])
        out["kME"] = kme
        out["hub_genes"] = _core.hub_genes(kme, top_n=n_top_hub)
        if traits is not None:
            tr = traits if isinstance(traits, pd.DataFrame) else pd.DataFrame(traits)
            out["module_trait"] = _core.module_trait_correlation(me["eigengenes"], tr)
    return out


def _genes_by_module(colors: Sequence[Any], genes: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c, g in zip(colors, genes):
        out.setdefault(str(c), []).append(str(g))
    return out


def soft_threshold(expr, powers: Optional[Sequence[float]] = None,
                   network_type: str = "unsigned", rsq_cut: float = 0.9) -> Dict[str, Any]:
    """单独跑软阈值筛选（想手动定 power 时用）。"""
    return _core.pick_soft_threshold(_core._as_expr(expr), powers=powers,
                                     network_type=network_type, rsq_cut=rsq_cut)


def module_preservation_hint(expr, colors: Sequence[Any]) -> Dict[str, Any]:
    """粗粒度模块质量提示：模块内平均相关 vs 模块间平均相关。

    这不是标准的 modulePreservation（Z-summary，需独立数据集），
    只用于**快速判断模块是否内聚**——不要拿它当保存性统计量报告。
    """
    e = _core._as_expr(expr)
    cor = np.corrcoef(e.to_numpy(dtype=float), rowvar=False)
    colors = np.asarray(colors, dtype=object)
    within, between = [], []
    mods = [c for c in pd.unique(colors) if str(c) != "grey"]
    idx = {m: np.flatnonzero(colors == m) for m in mods}
    for m in mods:
        ii = idx[m]
        if len(ii) > 1:
            sub = cor[np.ix_(ii, ii)]
            within.append(float(np.mean(sub[np.triu_indices(len(ii), 1)])))
    for a in range(len(mods)):
        for b in range(a + 1, len(mods)):
            sub = cor[np.ix_(idx[mods[a]], idx[mods[b]])]
            between.append(float(np.mean(sub)))
    return {
        "mean_within_module_cor": float(np.mean(within)) if within else float("nan"),
        "mean_between_module_cor": float(np.mean(between)) if between else float("nan"),
        "n_modules": len(mods),
        "caution": "非 Z-summary 保存性统计量；仅作内聚性提示",
    }
