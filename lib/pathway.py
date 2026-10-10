"""分析法 ⑤：基因集富集分析四件套（ORA / GSEA / GSVA / ssGSEA）。

来源分工（口径不可混）：
  · ORA 超几何检验  —— `vendor/_enrich/stats.py`（源 GSEApy，BSD-3）
  · GSEA / ssGSEA   —— `vendor/_enrich/algorithm.py`（源 GSEApy，BSD-3）
  · GSVA            —— 本模块**照抄算法**实现（Hänzelmann et al. 2013）。
    GSVA 上游在 R 包 GSVA（GPL）里，Python 端 gseapy 的 GSVA 类依赖
    GSEAbase + numba/joblib 体系，无法单独内嵌，故按算法论文实现：
      ① 对每个基因算跨样本的累积密度（kcdf: gaussian / poisson / none）
      ② 转成样本内表达水平的秩统计量
      ③ 对基因集做 KS 型随机游走，取最大偏离为 GSVA score

⚠️ 三种方法回答的是**三个不同的问题**，不要互相替代：
  · ORA：给定一组差异基因，问「我的基因集是否被过度代表」（只数个数，丢掉效应量）
  · GSEA：给定全基因组的排序（如 logFC），问「基因集是否集中在两端」（用效应量）
  · GSVA/ssGSEA：不改基因集、不改样本，把「基因×样本」转成「通路×样本」的活性矩阵
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scipy import stats as _st

from vendor._enrich import algorithm as _alg
from vendor._enrich.stats import calc_pvalues as _calc_pvalues, multiple_testing_correction as _mtc


# ---------------------------------------------------------------------------
# ① ORA：超几何检验（源 GSEApy stats）
# ---------------------------------------------------------------------------
def ora(genes: Sequence[str], gene_sets: Dict[str, Sequence[str]],
        background: Optional[Sequence[str]] = None,
        fdr: float = 0.05) -> pd.DataFrame:
    """过表征分析（超几何检验）。

    :param genes: 目标基因列表（如差异基因）
    :param gene_sets: {基因集名: 基因列表}
    :param background: 背景基因集（**必须给**，不给就等于用 GSEApy 默认 20000，
                       会让 p 值系统性偏小——这是 ORA 最常见的假阳性来源）
    """
    # GSEApy 的 calc_pvalues 只认 set 或 int；传 list 会直接抛错。
    bg = set(str(x) for x in background) if background is not None else 20000
    # 返回 zip(*vals)：(term, pval, odds_ratio, x=重叠数, m=基因集大小, hits)
    terms, pvals, odds, xs, ms, hits = _calc_pvalues(
        list(genes), {k: list(v) for k, v in gene_sets.items()}, background=bg)
    bg_n = len(bg) if background is not None else 20000
    rows = []
    for t, p, od, x, m, hs in zip(terms, pvals, odds, xs, ms, hits):
        rows.append({"term": t, "overlap": int(x), "set_size": int(m),
                     "background": int(bg_n), "pvalue": float(p),
                     "odds_ratio": float(od),
                     "hits": list(hs) if hasattr(hs, "__iter__") else []})
    df = pd.DataFrame(rows)
    if len(df):
        q = _mtc(df["pvalue"].to_numpy(), alpha=fdr, method="benjamini-hochberg")
        # ⚠️ 上游返回的是 **(q, rej)** —— q 在**第 0 位**。取 [1] 会把 padj
        # 变成 bool 的 0/1（实测 p=0.2356 -> padj=0.0），是彻底错误的校正值。
        df["padj"] = np.asarray(q[0] if isinstance(q, tuple) else q, dtype=float)
    else:
        df["padj"] = []
    df["background_given"] = background is not None
    return df.sort_values("pvalue").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ② GSEA：排序基因的加权 KS 检验（源 GSEApy algorithm）
# ---------------------------------------------------------------------------
def gsea(ranking: pd.Series, gene_sets: Dict[str, Sequence[str]],
         weight: float = 1.0, n_perm: int = 1000, min_size: int = 15,
         max_size: int = 500, seed: int = 123) -> pd.DataFrame:
    """GSEA（prerank 模式）：输入全基因组排序指标（如 logFC、t 统计量）。

    返回 ES / NES / 名义 p / FDR。**NES 与 p 必须由置换得到**，
    n_perm 太小会让 p 值的分辨率不足（最小 p = 1/n_perm）。
    """
    r = pd.Series(ranking).sort_values(ascending=False)
    gene_list = [str(g) for g in r.index]
    cor_vec = r.to_numpy(dtype=float)
    rows = []
    es_list, null_list, names = [], [], []
    for name, members in gene_sets.items():
        members = [str(m) for m in members if str(m) in set(gene_list)]
        if not (min_size <= len(members) <= max_size):
            continue
        es, esnull, hit, res = _alg.enrichment_score(
            gene_list, cor_vec, members, weight=weight, nperm=n_perm, seed=seed)
        es_list.append(float(np.ravel(es)[0]))
        null_list.append(np.asarray(esnull, dtype=float).ravel())
        names.append(name)
    if not names:
        return pd.DataFrame(columns=["term", "ES", "NES", "pvalue", "FDR", "set_size"])
    es_arr = np.array(es_list)
    null_arr = np.array(null_list)
    sig = list(_alg.gsea_significance(es_arr, null_arr))
    nes = np.array([s[1] for s in sig], dtype=float)
    pv = np.array([s[2] for s in sig], dtype=float)
    fdr = np.array([s[3] for s in sig], dtype=float)
    return pd.DataFrame({
        "term": names,
        "ES": es_arr,
        "NES": nes,
        "pvalue": pv,
        "FDR": fdr,
        "set_size": [len(gene_sets[n]) for n in names],
    }).sort_values("pvalue").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ③ ssGSEA：单样本富集打分（源 GSEApy fast_ssgsea / enrichment_score）
# ---------------------------------------------------------------------------
def ssgsea(expr: pd.DataFrame, gene_sets: Dict[str, Sequence[str]],
           weight: float = 0.25, scale: bool = True) -> pd.DataFrame:
    """ssGSEA：对每个样本单独打分，输出 通路×样本 的活性矩阵。

    与 GSEA 的区别：ssGSEA **不需要表型分组**，每个样本独立打分，
    所以能做「每个病人一个通路活性值」——临床队列常用。
    """
    mat = expr.to_numpy(dtype=float)
    genes = [str(g) for g in expr.columns]
    samples = list(expr.index)
    out = {}
    for name, members in gene_sets.items():
        tag = np.isin(genes, [str(m) for m in members]).astype(float)
        if tag.sum() == 0:
            continue
        scores = []
        for j in range(mat.shape[0]):
            order = np.argsort(-mat[j])            # 样本内表达从高到低
            rank = np.empty(len(order), dtype=float)
            rank[order] = np.arange(1, len(order) + 1, dtype=float)
            scores.append(float(_alg.fast_ssgsea(tag, rank)))
        s = np.array(scores, dtype=float)
        if scale:
            rng = s.max() - s.min()
            s = s / rng if rng > 0 else s
        out[name] = s
    return pd.DataFrame(out, index=samples)


# ---------------------------------------------------------------------------
# ④ GSVA：照抄 Hänzelmann 2013 算法实现（非内嵌，许可原因）
# ---------------------------------------------------------------------------
def _kcdf_matrix(mat: np.ndarray, kcdf: str) -> np.ndarray:
    """把表达矩阵转成「跨样本的累积密度」矩阵（GSVA 第一步）。

    kcdf="gaussian"：用每个基因的经验分布做高斯核密度估计的 CDF；
    kcdf="poisson" ：用 Poisson CDF（原始 count 数据）；
    kcdf="none"    ：直接用**秩**（等价于经验 CDF）。
    """
    n_samples, n_genes = mat.shape
    if kcdf == "none":
        out = np.apply_along_axis(_st.rankdata, 0, mat)
        return out / n_samples
    out = np.zeros_like(mat, dtype=float)
    for g in range(n_genes):
        x = mat[:, g]
        if kcdf == "gaussian":
            mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
            sd = sd if sd > 0 else 1.0
            out[:, g] = _st.norm.cdf((x - mu) / sd)
        elif kcdf == "poisson":
            lam = float(np.mean(x))
            lam = max(lam, 1e-9)
            out[:, g] = _st.poisson.cdf(np.round(x).astype(int), lam)
        else:
            raise ValueError(f"kcdf 必须是 gaussian/poisson/none，收到 {kcdf!r}")
    return out


def gsva(expr: pd.DataFrame, gene_sets: Dict[str, Sequence[str]],
         kcdf: str = "gaussian", weight: float = 1.0,
         min_size: int = 5) -> pd.DataFrame:
    """GSVA：把 基因×样本 矩阵转成 通路×样本 的活性矩阵。

    算法（Hänzelmann, Castelo & Guinney 2013, BMC Bioinformatics 14:7）：
      ① 每个基因跨样本算累积密度（kcdf）
      ② 每个样本内，对基因按密度排序，算 rank 统计量
      ③ 对基因集做 KS 型随机游走：集合内 +|z|^weight/|S|，集合外 -1/(N-|S|)
      ④ 取游走的最大正偏离与最大负偏离之差作为该样本的 GSVA score

    ⚠️ 与 R 包 GSVA 的差异（如实标注）：上游在密度估计与 tie 处理上有额外细节
    （如 `tau` 参数、核带宽修正），本实现是**算法主干**，数值可能与 R 版有
    小幅差异；趋势与显著性结论应一致，但**绝对数值不可与 R 版逐位对比**。
    """
    mat = expr.to_numpy(dtype=float)
    genes = [str(g) for g in expr.columns]
    samples = list(expr.index)
    dens = _kcdf_matrix(mat, kcdf)
    out = {}
    for name, members in gene_sets.items():
        idx = np.flatnonzero(np.isin(genes, [str(m) for m in members]))
        if len(idx) < min_size:
            continue
        in_set = np.zeros(len(genes), dtype=bool)
        in_set[idx] = True
        n_in, n_out = int(in_set.sum()), int((~in_set).sum())
        if n_out == 0:
            continue
        scores = np.zeros(mat.shape[0])
        for j in range(mat.shape[0]):
            z = dens[j]
            order = np.argsort(-z)                 # 密度从高到低
            walk_in = np.where(in_set[order], np.abs(z[order]) ** weight / n_in,
                               -1.0 / n_out)
            walk = np.cumsum(walk_in)
            # 最大正偏离 与 最大负偏离 之差（GSVA 定义）
            scores[j] = max(walk.max(), 0.0) - max(-walk.min(), 0.0)
        out[name] = scores
    return pd.DataFrame(out, index=samples)


# ---------------------------------------------------------------------------
# ⑤ GSEA/GSVA 的下游：两组通路活性比较
# ---------------------------------------------------------------------------
def pathway_activity_test(activity: pd.DataFrame, group: Sequence[Any],
                          method: str = "t", fdr: float = 0.05) -> pd.DataFrame:
    """通路活性矩阵的两组比较（GSVA/ssGSEA 结果的常规下游）。"""
    g = np.asarray(group)
    levels = list(pd.unique(g))
    if len(levels) != 2:
        raise ValueError(f"只支持两组比较，当前有 {len(levels)} 组")
    a = activity.loc[g == levels[0]].to_numpy(dtype=float)
    b = activity.loc[g == levels[1]].to_numpy(dtype=float)
    rows = []
    for i, term in enumerate(activity.columns):
        x, y = a[:, i], b[:, i]
        if method == "wilcoxon":
            stat, p = _st.mannwhitneyu(x, y, alternative="two-sided")
        else:
            stat, p = _st.ttest_ind(x, y, equal_var=False)
        d = float(np.mean(x) - np.mean(y))
        pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
                         / max(len(x) + len(y) - 2, 1))
        rows.append({"term": term, "diff": d, "statistic": float(stat),
                     "pvalue": float(p), "cohens_d": d / pooled if pooled > 0 else float("nan")})
    df = pd.DataFrame(rows)
    if len(df):
        q = _mtc(df["pvalue"].to_numpy(), alpha=fdr, method="benjamini-hochberg")
        # ⚠️ 上游返回的是 **(q, rej)** —— q 在**第 0 位**。取 [1] 会把 padj
        # 变成 bool 的 0/1（实测 p=0.2356 -> padj=0.0），是彻底错误的校正值。
        df["padj"] = np.asarray(q[0] if isinstance(q, tuple) else q, dtype=float)
    return df.sort_values("pvalue").reset_index(drop=True)
