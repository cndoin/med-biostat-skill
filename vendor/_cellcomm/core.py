# PORTED-FILE · 来源 ventolab/CellphoneDB（MIT）
#   源文件 src/core/methods/cpdb_statistical_analysis_helper.py
# 移植方式：**算法照抄、依赖换底**。CellphoneDB 版把「配受体表」绑死在它自带的
#   数据库 + complex（多亚基）展开逻辑上，并用 multiprocessing 做置换；本文件
#   改成「用户传配受体表」的纯 numpy/pandas 接口，去 complex 展开、去并池。
# 逐函数对照（CPDB 源函数 -> 本文件）：
#   mean_analysis    -> communication_score 内的 real 计算（(x>0)(y>0)(x+y)/2）
#   percent_analysis -> communication_score 内的 perc 计算（(p1>thr)(p2>thr)）
#   shuffle_meta + _statistical_analysis + shuffled_greater_than_real
#                    -> 置换循环（打乱细胞标签重算均值）与经验 p 值
# 与上游的**已知差异（如实标注）**：
#   1. 不含 complex（如 TGFB1_TGFBR1_TGFBR2 多亚基）展开；需 complex 时请传入
#      已展开好的 pair 表。
#   2. 置换为单进程向量化实现（CPDB 用多进程）；分布一致，仅速度不同。
#   3. communication_summary / pathway_communication / EXAMPLE_LR_PAIRS
#      为本文件新增，非上游代码。
"""细胞—细胞通讯（配体/受体）分析。

**算法来源**：CellPhoneDB（ventolab/CellphoneDB, MIT），见文件头 PORTED 说明。

**红线（写死，别漏）**
1. p 值是**置换检验**（打乱细胞标签）得到的「经验 p」；最小可分辨 p = 1/n_perm。
   报 p 必须同时报 ``n_perm``。
2. 互作**方向性**：`mean(c1→c2)` 用 ligand 在 c1、receptor 在 c2。反过来是另一列，
   两者**不可对调**（配体/受体的生物学角色不同）。
3. CPDB 的 mean 不是表达量本身，而是**配受体同时表达时的平均**；跨 cluster 组合
   之间的绝对值不可直接横向比较「谁更强」，只能看显著性 + 相对排序。
4. 输入必须是**归一化后**的表达（log1p 或 CPM），直接传原始 count 会因测序深度
   差异产生假阳性。
5. 细胞数过少的 cluster（默认 <10）无法做置换统计 —— 直接剔除并在结果里列出，
   不静默参与比较。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "communication_score",
    "communication_summary",
    "EXAMPLE_LR_PAIRS",
    "pathway_communication",
]


# ── 示例配受体表（教学/冒烟用；生产请自供 CellPhoneDB / CellChatDB 官方表）──
# 说明：以下为**文献中高频、角色明确**的经典人类配受体对，仅用于接口演示与
# 回归测试。它不是任何数据库的完整拷贝，也不声称覆盖全部已知互作。
EXAMPLE_LR_PAIRS: List[Tuple[str, str]] = [
    ("CXCL12", "CXCR4"), ("CCL2", "CCR2"), ("CCL5", "CCR5"),
    ("IL6", "IL6R"), ("IL10", "IL10RA"), ("IL2", "IL2RA"),
    ("TNF", "TNFRSF1A"), ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"),
    ("VEGFA", "KDR"), ("VEGFA", "FLT1"), ("EGF", "EGFR"),
    ("IGF1", "IGF1R"), ("FGF2", "FGFR1"), ("PDGFA", "PDGFRA"),
    ("CD80", "CD28"), ("CD86", "CD28"), ("CD274", "PDCD1"),
    ("PDCD1LG2", "PDCD1"), ("CD40", "CD40LG"), ("CD40LG", "CD40"),
    ("HLA-A", "CD8A"), ("LGALS9", "HAVCR2"),
    ("JAG1", "NOTCH1"), ("DLL4", "NOTCH1"), ("WNT5A", "FZD5"),
    ("SEMA4D", "PLXNB1"), ("EFNB1", "EPHB2"), ("NRG1", "ERBB4"),
    ("APOE", "LRP1"), ("SPP1", "CD44"), ("THBS1", "CD47"),
    ("ICAM1", "ITGAL"), ("VCAM1", "ITGB1"), ("FN1", "ITGB1"),
]


def _as_lr_pairs(lr_pairs) -> pd.DataFrame:
    if isinstance(lr_pairs, pd.DataFrame):
        cols = {c.lower(): c for c in lr_pairs.columns}
        lig = cols.get("ligand") or cols.get("gene1") or lr_pairs.columns[0]
        rec = cols.get("receptor") or cols.get("gene2") or lr_pairs.columns[1]
        df = lr_pairs[[lig, rec]].copy()
        df.columns = ["ligand", "receptor"]
    else:
        rows = [(str(a), str(b)) for a, b in lr_pairs]
        df = pd.DataFrame(rows, columns=["ligand", "receptor"])
    df["ligand"] = df["ligand"].astype(str)
    df["receptor"] = df["receptor"].astype(str)
    return df


def _cluster_means(X: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """各 cluster 的基因均值 (k × 基因)。"""
    out = np.zeros((k, X.shape[1]))
    np.add.at(out, codes, X)
    cnt = np.bincount(codes, minlength=k).astype(float)
    out /= np.maximum(cnt, 1.0)[:, None]
    return out


def _cluster_percents(X: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """各 cluster 中「表达 >0」的细胞比例 (k × 基因)。"""
    pos = (X > 0).astype(float)
    out = np.zeros((k, X.shape[1]))
    np.add.at(out, codes, pos)
    cnt = np.bincount(codes, minlength=k).astype(float)
    out /= np.maximum(cnt, 1.0)[:, None]
    return out


def communication_score(expr, labels: Sequence[Any], lr_pairs,
                        n_perm: int = 1000, threshold: float = 0.1,
                        seed: int = 0, min_cells: int = 10) -> Dict[str, Any]:
    """细胞通讯打分（CellPhoneDB 口径 + 置换 p 值）。

    :param expr: (细胞 × 基因) DataFrame，**必须是归一化后的表达**
    :param labels: 长度 = 细胞数的 cluster 标签
    :param lr_pairs: 配受体表——DataFrame(ligand, receptor) 或 [(lig, rec), ...]
    :param n_perm: 置换次数（最小可分辨 p = 1/n_perm）
    :param threshold: 阳性比例阈值（CPDB 默认 0.1）
    :param min_cells: cluster 最少细胞数，低于则剔除并列在 ``excluded_clusters``
    :return: dict
        - ``means`` / ``pvalues`` / ``percents``: DataFrame（索引 = 配受体对，
          列 = "cluster1|cluster2"）
        - ``significant``: pval < 0.05 且 percent == 1 的互作（长表）
        - ``n_perm`` / ``threshold`` / ``n_pairs`` / ``n_missing_genes``
        - ``excluded_clusters``: 因细胞数不足被剔除的 cluster
        - ``method``: ``"CellPhoneDB-mean+perm(via MIT)"``
    """
    if not isinstance(expr, pd.DataFrame):
        expr = pd.DataFrame(np.asarray(expr, dtype=float))
    if expr.shape[0] != len(labels):
        raise ValueError(f"labels 长度 {len(labels)} != 细胞数 {expr.shape[0]}")
    labels = np.asarray([str(x) for x in labels])
    lr = _as_lr_pairs(lr_pairs)

    levels = list(pd.unique(labels))
    counts = {lv: int((labels == lv).sum()) for lv in levels}
    excluded = [lv for lv in levels if counts[lv] < min_cells]
    keep_levels = [lv for lv in levels if counts[lv] >= min_cells]
    if len(keep_levels) < 1:
        raise ValueError(
            f"所有 cluster 细胞数都 <{min_cells}（min_cells={min_cells}），无法做置换检验。")
    mask = np.isin(labels, keep_levels)
    labels = labels[mask]
    X = expr.to_numpy(dtype=float)[mask, :]
    level_to_code = {lv: i for i, lv in enumerate(keep_levels)}
    codes = np.array([level_to_code[x] for x in labels])
    k = len(keep_levels)

    genes = [str(g) for g in expr.columns]
    gene_idx = {g: i for i, g in enumerate(genes)}
    pair_ok, missing = [], []
    for lig, rec in zip(lr["ligand"], lr["receptor"]):
        if lig in gene_idx and rec in gene_idx:
            pair_ok.append((lig, rec))
        else:
            missing.append((lig, rec))
    if not pair_ok:
        raise ValueError(
            "配受体表中没有任何一对的基因同时出现在表达矩阵里——检查基因命名"
            "（符号 vs Ensembl ID）与是否做过 HGNC 统一。")
    names = [f"{a}|{b}" for a, b in pair_ok]
    g1 = np.array([gene_idx[a] for a, _ in pair_ok])
    g2 = np.array([gene_idx[b] for _, b in pair_ok])

    # 全部有序 cluster 组合（含自身）
    c1_idx = np.repeat(np.arange(k), k)
    c2_idx = np.tile(np.arange(k), k)
    colnames = [f"{keep_levels[i]}|{keep_levels[j]}" for i, j in zip(c1_idx, c2_idx)]

    # 用显式循环构造（pair 数量不大，逻辑更清晰、易核）
    M_real = _cluster_means(X, codes, k)
    P_real = _cluster_percents(X, codes, k)
    real = np.zeros((len(pair_ok), k * k))
    perc = np.zeros((len(pair_ok), k * k))
    for pi in range(len(pair_ok)):
        x = M_real[c1_idx, g1[pi]]
        y = M_real[c2_idx, g2[pi]]
        real[pi] = (x > 0) * (y > 0) * (x + y) / 2.0
        p1 = P_real[c1_idx, g1[pi]]
        p2 = P_real[c2_idx, g2[pi]]
        perc[pi] = ((p1 > threshold) * (p2 > threshold)).astype(float)

    rng = np.random.default_rng(seed)
    greater = np.zeros_like(real)
    for _ in range(n_perm):
        perm = rng.permutation(len(codes))
        Mp = _cluster_means(X[perm], codes, k)
        for pi in range(len(pair_ok)):
            x = Mp[c1_idx, g1[pi]]
            y = Mp[c2_idx, g2[pi]]
            val = (x > 0) * (y > 0) * (x + y) / 2.0
            greater[pi] += (val > real[pi]).astype(float)
    pvalues = greater / float(n_perm)
    # CPDB 口径：真实值为 0 的组合 p=1（不可能是「显著」）
    pvalues = np.where(real > 0, pvalues, 1.0)

    means_df = pd.DataFrame(real, index=names, columns=colnames)
    pval_df = pd.DataFrame(pvalues, index=names, columns=colnames)
    perc_df = pd.DataFrame(perc, index=names, columns=colnames)

    sig_rows = []
    sig_mask = (pval_df.to_numpy() < 0.05) & (perc_df.to_numpy() == 1.0) & (means_df.to_numpy() > 0)
    for pi, name in enumerate(names):
        for ci, col in enumerate(colnames):
            if sig_mask[pi, ci]:
                sig_rows.append({"pair": name, "source_target": col,
                                 "mean": float(real[pi, ci]),
                                 "pvalue": float(pvalues[pi, ci]),
                                 "percent": 1})
    sig = (pd.DataFrame(sig_rows).sort_values("pvalue").reset_index(drop=True)
           if sig_rows else pd.DataFrame(columns=["pair", "source_target", "mean", "pvalue", "percent"]))

    return {
        "means": means_df,
        "pvalues": pval_df,
        "percents": perc_df,
        "significant": sig,
        "n_perm": int(n_perm),
        "threshold": float(threshold),
        "n_pairs": len(pair_ok),
        "n_missing_genes": len(missing),
        "missing_pairs": [f"{a}|{b}" for a, b in missing],
        "excluded_clusters": [f"{lv}(n={counts[lv]})" for lv in excluded],
        "n_cells_used": int(mask.sum()),
        "method": "CellPhoneDB-mean+perm(via MIT)",
        "engine": "lib.cellcomm.communication_score",
    }


def communication_summary(result: Dict[str, Any], top_n: int = 15) -> pd.DataFrame:
    """把 ``communication_score`` 的显著互作汇总成「cluster 对 × LR 对」排名表。"""
    sig = result.get("significant")
    if sig is None or len(sig) == 0:
        return pd.DataFrame(columns=["source", "target", "n_significant_pairs",
                                     "mean_score", "pairs"])
    df = sig.copy()
    src, tgt = zip(*[s.split("|", 1) for s in df["source_target"]])
    df["source"], df["target"] = src, tgt
    grp = (df.groupby(["source", "target"])
             .agg(n_significant_pairs=("pair", "nunique"),
                  mean_score=("mean", "mean"),
                  pairs=("pair", lambda s: ";".join(sorted(set(s))[:6])))
             .reset_index()
             .sort_values(["n_significant_pairs", "mean_score"], ascending=False))
    return grp.head(top_n).reset_index(drop=True)


def pathway_communication(result: Dict[str, Any],
                          pathways: Dict[str, Iterable[str]]) -> pd.DataFrame:
    """把 LR 对按通路归类汇总（如 chemokine / TGF-beta / checkpoint）。"""
    means = result.get("means")
    pvals = result.get("pvalues")
    rows = []
    for pname, members in pathways.items():
        mem = set(str(m) for m in members)
        sel = [i for i in means.index if i in mem]
        if not sel:
            continue
        sub_m = means.loc[sel]
        sub_p = pvals.loc[sel]
        rows.append({
            "pathway": pname,
            "n_pairs": len(sel),
            "n_significant": int((sub_p.to_numpy() < 0.05).sum()),
            "mean_score": float(sub_m.to_numpy().mean()),
            "max_score": float(sub_m.to_numpy().max()),
        })
    return pd.DataFrame(rows).sort_values(
        ["n_significant", "mean_score"], ascending=False).reset_index(drop=True)
