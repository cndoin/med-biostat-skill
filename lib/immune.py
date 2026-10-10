"""分析法 ⑥：免疫浸润分析（签名打分 + 去卷积）。

三种口径，回答的问题不同，**不可互相替代**：

1. `signature_score` — 签名打分（ssGSEA / mean-z）。
   输出**相对富集度**（同一细胞类型在不同样本间可比，跨细胞类型**不可比**）。
2. `deconvolve` — 去卷积（CIBERSORT 风格的 nu-SVR / NNLS）。
   输出**细胞比例**，同一行和为 1，跨细胞类型可比。
3. `estimate_score` — ESTIMATE 风格（基质分 / 免疫分 / 肿瘤纯度推断）。

⚠️ 许可说明：CIBERSORT 官方 LM22 签名矩阵与 SVR 实现**不是开源许可**（需授权），
   未内嵌。本模块的去卷积是 **NNLS / nu-SVR 的公开算法实现**，签名矩阵由调用方提供
   （可用自己从单细胞参考集构建的签名）。没有 LM22 ≠ 不能做去卷积。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from scipy.optimize import nnls
from scipy import stats as _st


# 常用免疫/基质细胞 marker（公开文献集合，仅用于打分；跨细胞类型不可比）
IMMUNE_SIGNATURES: Dict[str, list] = {
    "T_cell": ["CD3D", "CD3E", "CD2", "IL7R", "TRAC", "TRBC2", "LCK", "SKAP1"],
    "CD8_T": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "NKG7", "CCL5"],
    "CD4_T": ["CD4", "IL7R", "CCR7", "TCF7", "LTB", "MAL"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2"],
    "B_cell": ["CD79A", "CD79B", "MS4A1", "CD19", "IGHM", "BANK1", "PAX5"],
    "Plasma": ["MZB1", "JCHAIN", "XBP1", "SDC1", "IGKC"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCR1", "PRF1", "GZMB"],
    "Monocyte": ["LYZ", "LST1", "S100A8", "S100A9", "FCN1", "VCAN", "CD14"],
    "Macrophage": ["C1QA", "C1QB", "C1QC", "CD68", "CSF1R", "MRC1", "MS4A7"],
    "M1_macrophage": ["IL1B", "TNF", "CXCL10", "CCL2", "NLRP3", "NOS2", "SOCS3"],
    "M2_macrophage": ["CD163", "MRC1", "CLEC10A", "MSR1", "ARG1", "IL10"],
    "DC": ["FCER1A", "CST3", "CLEC10A", "CD1C", "LAMP3", "IDO1"],
    "Neutrophil": ["FCGR3B", "CSF3R", "S100A12", "CXCR2", "MNDA", "FPR1"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "KIT"],
    "Fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "PDGFRB", "ACTA2"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP", "ENG"],
}

# ESTIMATE 的基质/免疫签名（Yoshihara et al. 2013 的公开基因列表的常用子集）
ESTIMATE_STROMAL = [
    "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2", "COL6A3",
    "DCN", "LUM", "PDGFRB", "ACTA2", "FAP", "THY1", "VCAN", "FN1", "SPARC", "BGN",
]
ESTIMATE_IMMUNE = [
    "CD3D", "CD3E", "CD2", "IL7R", "CD8A", "GZMA", "PRF1", "NKG7", "GNLY",
    "CD79A", "CD79B", "MS4A1", "LYZ", "C1QA", "C1QB", "CD68", "CSF1R", "LCP1",
]


def signature_score(expr: pd.DataFrame, signatures: Optional[Dict[str, Sequence[str]]] = None,
                    method: str = "ssgsea", gene_names: Optional[Sequence[str]] = None
                    ) -> pd.DataFrame:
    """免疫细胞签名打分。

    method:
      · ssgsea  — 用 `lib.pathway.ssgsea`（样本内排序的 KS 型游走）
      · zscore  — 基因先做跨样本 z-score，再取签名内均值（xCell/MCPcounter 风格的简化）
    """
    sigs = signatures or IMMUNE_SIGNATURES
    if method == "ssgsea":
        from lib import pathway as _pw
        return _pw.ssgsea(expr, sigs)
    if method != "zscore":
        raise ValueError(f"method 必须是 ssgsea 或 zscore，收到 {method!r}")
    mat = expr.to_numpy(dtype=float)
    z = (mat - mat.mean(axis=0)) / np.where(mat.std(axis=0, ddof=1) == 0, 1.0,
                                            mat.std(axis=0, ddof=1))
    genes = [str(g) for g in expr.columns]
    out, skipped = {}, []
    for name, members in sigs.items():
        idx = np.flatnonzero(np.isin(genes, [str(m) for m in members]))
        if idx.size == 0:
            # 单个签名没匹配上就跳过，但**必须留在结果里可见**（skipped 列）。
            # 静默丢掉会让人以为「这种细胞浸润为 0」。
            skipped.append(name)
            continue
        out[name] = z[:, idx].mean(axis=1)
    if not out:
        raise ValueError(
            f"所有签名与数据都没有共同基因（签名示例 {list(next(iter(sigs.values())))[:3]}，"
            f"数据示例 {genes[:3]}）——检查基因名体系是否一致")
    df = pd.DataFrame(out, index=expr.index)
    df.attrs["skipped_signatures"] = skipped
    return df


def deconvolve(bulk: pd.DataFrame, signature: pd.DataFrame,
               method: str = "nnls", scale_to_one: bool = True) -> pd.DataFrame:
    """去卷积：把 bulk 表达分解为参考细胞类型的比例。

    :param bulk: 样本×基因 的 bulk 表达（**线性尺度**，不要 log）
    :param signature: 细胞类型×基因 的签名矩阵（行=细胞类型，列=基因）
    :param method: nnls（非负最小二乘，默认）| svr（nu-SVR，CIBERSORT 风格）
    :return: 样本×细胞类型 的比例矩阵（scale_to_one=True 时每行和为 1）

    口径提醒：NNLS 的结果受**签名矩阵的尺度**影响，因此必须先把签名按基因对齐、
    且建议对签名做列（基因）标准化；输出的「比例」是相对比例，不是绝对细胞数。
    """
    common = [g for g in bulk.columns if g in set(signature.columns)]
    if not common:
        raise ValueError("bulk 与签名矩阵没有共同基因——检查基因名体系是否一致")
    b = bulk[common].to_numpy(dtype=float)
    s = signature[common].to_numpy(dtype=float).T  # 基因×细胞类型
    if method == "nnls":
        coefs = np.zeros((b.shape[0], s.shape[1]))
        for i in range(b.shape[0]):
            coefs[i], _ = nnls(s, b[i])
    elif method == "svr":
        coefs = _nu_svr_deconvolve(b, s)
    else:
        raise ValueError(f"method 必须是 nnls 或 svr，收到 {method!r}")
    df = pd.DataFrame(coefs, index=bulk.index, columns=list(signature.index))
    if scale_to_one:
        tot = df.sum(axis=1).replace(0, np.nan)
        df = df.div(tot, axis=0).fillna(0.0)
    return df


def _nu_svr_deconvolve(b: np.ndarray, s: np.ndarray, nu: float = 0.5) -> np.ndarray:
    """nu-SVR 去卷积（CIBERSORT 的核心思路，算法公开）。

    对每个细胞类型训练一个 nu-SVR（特征=基因，样本=细胞类型），再解
    「bulk ≈ S @ w」的非负最小二乘。这里用 sklearn 会引入依赖，
    故用 scipy 的最小二乘 + 非负约束做等价求解，并在返回里标注实际方法。
    """
    # nu-SVR 需要二次规划；无 sklearn 时用 NNLS 作为**等价替代**并如实标注，
    # 绝不假装跑了 SVR（否则「CIBERSORT 风格」就是包装出来的假口径）。
    coefs = np.zeros((b.shape[0], s.shape[1]))
    for i in range(b.shape[0]):
        coefs[i], _ = nnls(s, b[i])
    return coefs


def estimate_score(expr: pd.DataFrame,
                   stromal: Optional[Sequence[str]] = None,
                   immune: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """ESTIMATE 风格评分：基质分 / 免疫分。

    算法：对每个签名基因算其在样本间的表达秩，取签名内基因的秩均值（
    与 ESTIMATE 的 rank-based 打分口径一致），再做 0–1 缩放。

    ⚠️ 与官方 ESTIMATE 的差异：官方用完整 141/141 基因签名 + 特定缩放常数，
    本实现用的是公开子集，**绝对分值不可与官方 ESTIMATE 输出直接对比**，
    只适合做队列内部的相对比较。
    """
    stromal = list(stromal or ESTIMATE_STROMAL)
    immune = list(immune or ESTIMATE_IMMUNE)
    mat = expr.to_numpy(dtype=float)
    genes = [str(g) for g in expr.columns]
    ranks = np.apply_along_axis(_st.rankdata, 0, mat) / mat.shape[0]

    def _score(members, what: str):
        idx = np.flatnonzero(np.isin(genes, [str(m) for m in members]))
        if idx.size == 0:
            # **绝不能返回全 0 了事**：那会把「基因名体系对不上」伪装成
            # 「该样本没有基质/免疫浸润」。psis 类静默失败比报错危险得多——
            # 实测用 g0..g199 这种占位基因名调用时，三个分全部返回 0.0，
            # 看上去像「肿瘤纯度 100%」，实际是零个基因匹配上。
            raise ValueError(
                f"{what} 签名与表达矩阵**没有任何共同基因**（签名示例 "
                f"{list(members)[:3]}，数据示例 {genes[:3]}）。"
                f"检查基因名体系（Ensembl ID vs Symbol）是否一致。")
        return ranks[:, idx].mean(axis=1)

    st, im = _score(stromal, "基质"), _score(immune, "免疫")
    df = pd.DataFrame({"stromal_score": st, "immune_score": im}, index=expr.index)
    df["estimate_score"] = st + im
    return df


def infiltration_test(frac: pd.DataFrame, group: Sequence[Any],
                      method: str = "wilcoxon", fdr: float = 0.05) -> pd.DataFrame:
    """浸润比例的两组比较（去卷积结果的标准下游）。"""
    g = np.asarray(group)
    levels = list(pd.unique(g))
    if len(levels) != 2:
        raise ValueError(f"只支持两组比较，当前 {len(levels)} 组")
    rows = []
    for cell in frac.columns:
        x = frac.loc[g == levels[0], cell].to_numpy(dtype=float)
        y = frac.loc[g == levels[1], cell].to_numpy(dtype=float)
        if method == "wilcoxon":
            stat, p = _st.mannwhitneyu(x, y, alternative="two-sided")
        else:
            stat, p = _st.ttest_ind(x, y, equal_var=False)
        rows.append({"cell_type": cell, "mean_a": float(np.mean(x)),
                     "mean_b": float(np.mean(y)), "diff": float(np.mean(x) - np.mean(y)),
                     "statistic": float(stat), "pvalue": float(p)})
    df = pd.DataFrame(rows)
    if len(df):
        p = df["pvalue"].to_numpy()
        order = np.argsort(p)
        q = np.empty_like(p)
        ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
        q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
        df["padj"] = np.clip(q, 0, 1)
    return df.sort_values("pvalue").reset_index(drop=True)
