"""分析法 ④：单细胞 / 单核转录组全流程（QC → 归一化 → HVG → PCA → 邻居 → 聚类 → marker）。

算法来自 `vendor/_sc/core.py`（移植自 scanpy，BSD-3-Clause）。本模块是包装层。

**与 scanpy 的已知差异必须随结果一起说明**（否则会被当成 bug）：
  · 聚类是 **Louvain**，不是 Leiden（Leiden 的 Python 实现许可未确认，未内嵌）
  · 邻居图连接度用自适应高斯核，不是 umap 模糊单纯形集
  · t-SNE 是精确梯度版（O(n²)，n>2000 慎用），UMAP 未内嵌
  · HVG 支持 seurat / cell_ranger 两种 flavor，seurat_v3 需要的 LOESS 未内嵌

矩阵约定：**行=细胞、列=基因**（与 AnnData.X 同向）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from vendor._sc import core as _core


def qc(counts, gene_names: Optional[Sequence[str]] = None,
       mito_prefix: str = "MT-") -> pd.DataFrame:
    """细胞级 QC 指标：n_genes / n_counts / pct_mito。"""
    return _core.qc_metrics(counts, gene_names=gene_names, mito_prefix=mito_prefix)


def standard_pipeline(counts, gene_names: Optional[Sequence[str]] = None,
                      min_genes: int = 200, max_pct_mito: float = 20.0,
                      min_cells: int = 3, target_sum: float = 1e4,
                      n_top_genes: int = 2000, flavor: str = "seurat",
                      n_comps: int = 30, n_neighbors: int = 15,
                      resolution: float = 1.0, seed: int = 0,
                      mito_prefix: str = "MT-") -> Dict[str, Any]:
    """标准单细胞流程，一步返回 QC 过滤 + 归一化 + HVG + PCA + 聚类结果。

    每一步的中间产物都保留在返回值里（`normalized` / `hvg` / `pca` / `neighbors`），
    便于逐步排查——不要只留最后的 labels（出问题就无从查起）。
    """
    mat = counts.toarray().astype(float) if hasattr(counts, "toarray") else np.asarray(counts, dtype=float)
    steps: Dict[str, Any] = {"n_cells_in": int(mat.shape[0]), "n_genes_in": int(mat.shape[1])}

    fc = _core.filter_cells(mat, min_genes=min_genes, max_pct_mito=max_pct_mito,
                            gene_names=gene_names, mito_prefix=mito_prefix)
    mat = mat[fc["keep_index"]]
    steps["filter_cells"] = {"n_kept": fc["n_kept"], "rules": fc["rules"]}

    fg = _core.filter_genes(mat, min_cells=min_cells)
    mat = mat[:, fg["keep_index"]]
    genes = None if gene_names is None else [str(gene_names[i]) for i in fg["keep_index"]]
    steps["filter_genes"] = {"n_kept": fg["n_kept"], "rules": fg["rules"]}

    norm = _core.normalize_total(mat, target_sum=target_sum)
    log = _core.log1p(norm)
    steps["normalize"] = {"target_sum": float(target_sum)}

    hvg = _core.highly_variable_genes(log, gene_names=genes, flavor=flavor,
                                      n_top_genes=min(n_top_genes, mat.shape[1]))
    steps["hvg"] = {"n_selected": int(hvg["highly_variable"].sum()), "flavor": flavor}

    sub = log[:, hvg["highly_variable"]]
    scaled = _core.scale(sub)
    p = _core.pca(scaled, n_comps=min(n_comps, min(scaled.shape) - 1))
    steps["pca"] = {"n_comps": p["n_comps"],
                    "var_explained_first10": float(np.sum(p["variance_ratio"][:10]))}

    nb = _core.neighbors(p["scores"], n_neighbors=n_neighbors)
    cl = _core.louvain(nb["connectivities"], resolution=resolution, seed=seed)

    return {
        "steps": steps,
        "labels": cl["labels"],
        "n_clusters": cl["n_clusters"],
        "modularity": cl["modularity"],
        "algorithm": cl["algorithm"],
        "pca": p,
        "neighbors": nb,
        "normalized": norm,
        "log": log,
        "hvg": hvg,
        "genes": genes,
        "cell_index_kept": fc["keep_index"],
        "caveat": "Louvain（非 Leiden）；连接度=自适应高斯核（非 umap 模糊单纯形集）",
    }


def cluster(counts, n_neighbors: int = 15, resolution: float = 1.0,
            n_comps: int = 30, seed: int = 0) -> Dict[str, Any]:
    """只做聚类（输入已是归一化+对数化后的矩阵时可跳过前面步骤）。"""
    p = _core.pca(_core.scale(counts), n_comps=n_comps)
    nb = _core.neighbors(p["scores"], n_neighbors=n_neighbors)
    cl = _core.louvain(nb["connectivities"], resolution=resolution, seed=seed)
    cl["pca"] = p
    return cl


def markers(log_expr, labels: Sequence[Any], gene_names: Optional[Sequence[str]] = None,
            method: str = "wilcoxon", n_top: int = 50,
            min_logfc: float = 0.25, max_padj: float = 0.05) -> Dict[str, Any]:
    """找每个簇的 marker 基因，并按显著性过滤。

    logFC 口径：mean(expm1(簇内)) - mean(expm1(其余))（scanpy 口径）。
    过滤条件 |logFC| >= min_logfc 且 padj <= max_padj；两者都可设为 None 关掉。
    """
    res = _core.rank_genes_groups(log_expr, labels, gene_names=gene_names, method=method)
    filtered = {}
    for g, df in res["by_group"].items():
        d = df.copy()
        if min_logfc is not None:
            d = d[d["logFC"].abs() >= min_logfc]
        if max_padj is not None:
            d = d[d["padj"] <= max_padj]
        filtered[g] = d.head(n_top)
    return {
        "all": res["by_group"],
        "significant": filtered,
        "method": method,
        "filter": {"min_abs_logfc": min_logfc, "max_padj": max_padj},
        "logfc_rule": res["logfc_rule"],
    }


def score_signature(log_expr, gene_list: Sequence[str], gene_names: Sequence[str],
                    n_bins: int = 25, ctrl_size: int = 50, seed: int = 0) -> np.ndarray:
    """基因集打分（表达水平匹配对照），用于细胞类型注释 / 通路活性。"""
    return _core.score_genes(log_expr, gene_list, gene_names, n_bins=n_bins,
                             ctrl_size=ctrl_size, seed=seed)


def annotate_by_markers(log_expr, gene_names: Sequence[str],
                        marker_sets: Dict[str, Sequence[str]],
                        n_bins: int = 25, seed: int = 0) -> pd.DataFrame:
    """用 marker 集给每个细胞打分，取分最高的细胞类型作为注释。

    ⚠️ 这是**打分式注释**，不是分类器：它不做统计检验，也不保证互斥。
    结果必须再用 marker 基因的差异表达复核（见 `markers`）。
    """
    scores = {}
    for name, genes in marker_sets.items():
        scores[name] = _core.score_genes(log_expr, genes, gene_names,
                                         n_bins=n_bins, seed=seed)
    df = pd.DataFrame(scores)
    df["annotated"] = df.idxmax(axis=1)
    df["margin"] = df[list(scores)].apply(lambda r: r.nlargest(2).iloc[0] - r.nlargest(2).iloc[1], axis=1)
    return df


def embedding(log_expr, method: str = "pca", n_comps: int = 30,
              perplexity: float = 30.0, n_iter: int = 500, seed: int = 0) -> Dict[str, Any]:
    """降维嵌入：pca（默认）或 tsne。UMAP 未内嵌（依赖 numba/pynndescent）。"""
    if method == "pca":
        p = _core.pca(_core.scale(log_expr), n_comps=n_comps)
        return {"embedding": p["scores"], "method": "pca", "variance_ratio": p["variance_ratio"]}
    if method == "tsne":
        p = _core.pca(_core.scale(log_expr), n_comps=min(n_comps, 30))
        t = _core.tsne(p["scores"], perplexity=perplexity, n_iter=n_iter, seed=seed)
        return {"embedding": t["embedding"], "method": "tsne",
                "kl_divergence": t["kl_divergence"], "rule": t["rule"]}
    raise ValueError(f"method 必须是 pca 或 tsne（UMAP 未内嵌），收到 {method!r}")


# ── 细胞类型注释 ──────────────────────────────────────────────────────────────
def annotate_cells(log_expr, clusters, marker_sets, min_score: float = 0.0,
                   ambiguity_ratio: float = 0.8):
    """基于 marker 基因集的 cluster 级细胞类型注释。

    :param log_expr: (细胞 × 基因) 的**已归一化 log 表达** DataFrame
    :param clusters: 长度 = 细胞数的 cluster 标签
    :param marker_sets: ``{细胞类型: [基因名, ...]}``
    :param min_score: 最高分低于此值判为 ``Unassigned``
    :param ambiguity_ratio: ``最高分/次高分`` 低于此值则标记为「不确定」
    :return: dict
        - ``scores``: DataFrame（cluster × 细胞类型），cluster 内 marker 的 z-score 均值
        - ``annotation``: ``{cluster: {type, confidence, ratio_to_runner_up, ambiguous, ...}}``
        - ``unmatched_markers``: 各 marker set 中**不在表达矩阵里**的基因
        - ``matched_markers``: 各 marker set 实际匹配到的个数
        - ``rule`` / ``engine``

    **红线（三条都踩过）**
    1. **marker 一个都没匹配上时必须报错**，不能静默返回 0 分——那会让所有 cluster
       都被注释成同一个类型，而且看起来还挺像回事。
       最常见的真实原因是基因命名不一致（符号 vs Ensembl ID）。
    2. **注释是假设不是结论**。本函数额外给出「最高分 / 次高分」作为不确定性提示，
       分不开时明确标 ``ambiguous``，而不是硬给一个类型。
    3. **打分必须先跨基因标准化（z-score）再平均**。直接平均原始表达会被高表达
       基因完全主导——不同基因的表达量级能差几个数量级。
    """
    import pandas as pd

    if not hasattr(log_expr, "columns"):
        raise ValueError("log_expr 必须是 DataFrame（列名 = 基因名）")
    genes = [str(g) for g in log_expr.columns]
    gset = set(genes)
    labels = pd.Series([str(c) for c in clusters], index=log_expr.index)
    if len(labels) != log_expr.shape[0]:
        raise ValueError(f"clusters 长度 {len(labels)} != 细胞数 {log_expr.shape[0]}")

    matched, unmatched = {}, {}
    for ct, mk in marker_sets.items():
        hit = [str(m) for m in mk if str(m) in gset]
        unmatched[str(ct)] = [str(m) for m in mk if str(m) not in gset]
        if not hit:
            raise ValueError(
                f"细胞类型 {ct!r} 的 marker 一个都没匹配上表达矩阵"
                f"（示例 {list(mk)[:3]}）——请检查基因命名（符号 vs Ensembl ID）")
        matched[str(ct)] = hit

    X = log_expr.to_numpy(dtype=float)
    sd = X.std(axis=0)
    z = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    zdf = pd.DataFrame(z, index=log_expr.index, columns=genes)

    rows = {}
    for cl in pd.unique(labels):
        m = (labels == cl).to_numpy()
        rows[str(cl)] = {ct: float(zdf.loc[m, hit].to_numpy().mean())
                         for ct, hit in matched.items()}
    scores = pd.DataFrame(rows).T          # (cluster × 细胞类型)

    annotation = {}
    for cl, row in scores.iterrows():
        ordered = row.sort_values(ascending=False)
        top = float(ordered.iloc[0])
        second = float(ordered.iloc[1]) if len(ordered) > 1 else float("-inf")
        if top < min_score:
            annotation[cl] = {"type": "Unassigned", "confidence": top,
                              "ambiguous": True, "note": "最高分低于 min_score"}
            continue
        ratio = (top / second) if second > 0 else float("inf")
        annotation[cl] = {
            "type": str(ordered.index[0]),
            "confidence": top,
            "runner_up": str(ordered.index[1]) if len(ordered) > 1 else None,
            "ratio_to_runner_up": ratio,
            "ambiguous": bool(ratio < ambiguity_ratio),
            "n_markers": len(matched[str(ordered.index[0])]),
        }

    return {
        "scores": scores,
        "annotation": annotation,
        "unmatched_markers": {k: v for k, v in unmatched.items() if v},
        "matched_markers": {k: len(v) for k, v in matched.items()},
        "rule": ("cluster 内 marker 基因 z-score 均值最高者为注释类型；"
                 "ratio_to_runner_up < ambiguity_ratio 时标为不确定"),
        "engine": "lib.singlecell.annotate_cells",
    }


# ── 表达率统计（S0 判决性预检的直接产出）──────────────────────────────────────
def expression_rate(log_expr, groups, gene_names: Optional[Sequence[str]] = None,
                    genes: Optional[Sequence[str]] = None,
                    detection_threshold: float = 0.0) -> Dict[str, Any]:
    """按分组统计每个基因的**检出率**与平均表达。

    这是「判决性预检」的直接产出：判据「风险基因在该组织可检出吗」中的
    "可检出"**必须由表达率定义**，不能凭印象或凭 marker 是否出现在差异表里。

    **口径（随结果一起输出；改口径必须改这里，不允许在调用处临时换）**
      · ``detection``：``X > detection_threshold`` 判为检出。默认 **0**
        （"非零即检出"）。对已 log1p 的归一化矩阵与原始 counts，0 的含义一致
        （都等价于"至少 1 个 count"）。数据若经过中心化等变换，**必须显式改阈值**
        ——否则"检出率"会变成负数比例，毫无意义。
      · ``denominator``：该组的**全部细胞数**。不丢弃细胞、不下采样。
        丢弃细胞会同时改变分子与分母，是最隐蔽的一类偏差。
      · ``mean_expr``：该组**全部细胞**的算术平均（**含未检出细胞**）。
      · ``mean_expr_positive``：仅该组**阳性细胞**的平均。
        两者都返回，避免"平均表达"四个字在不同人嘴里指不同的量。

    :param log_expr: (细胞 × 基因) 矩阵（DataFrame 或 ndarray）
    :param groups: 长度 = 细胞数 的分组标签（细胞类型 / 供体 / 年龄段）
    :param gene_names: 基因名；``log_expr`` 是 DataFrame 时可省略
    :param genes: 只统计这些基因（如风险基因列表），None = 全部
    :return: dict
        - ``table``: 长表（group × gene），列见下
        - ``pct_matrix``: 宽表 (分组 × 基因) 的检出率百分比
        - ``mean_matrix``: 宽表 (分组 × 基因) 的平均表达
        - ``n_cells``: ``{分组: 细胞数}``
        - ``detection_rule`` / ``mean_rule`` / ``engine``: 口径与来源
    """
    if hasattr(log_expr, "columns"):
        all_genes = [str(c) for c in log_expr.columns]
        X = log_expr.to_numpy(dtype=float)
        idx = list(log_expr.index)
    else:
        X = np.asarray(log_expr, dtype=float)
        if gene_names is None:
            raise ValueError("ndarray 输入必须给 gene_names")
        all_genes = [str(g) for g in gene_names]
        idx = list(range(X.shape[0]))

    if X.shape[0] != len(groups):
        raise ValueError(f"groups 长度 {len(groups)} != 细胞数 {X.shape[0]}")

    gnames = [str(g) for g in groups]
    pos = {g: i for i, g in enumerate(all_genes)}

    if genes is None:
        sel = list(range(len(all_genes)))
    else:
        sel, missing = [], []
        for g in genes:
            g = str(g)
            if g in pos:
                sel.append(pos[g])
            else:
                missing.append(g)
        if not sel:
            raise ValueError(
                f"指定基因一个都不在矩阵里（示例 {[str(x) for x in genes][:3]}）"
                "——请检查基因命名（符号 vs Ensembl ID）与大小写")
        sel_genes_missing = missing

    sel = np.asarray(sel, dtype=int)
    sub = X[:, sel]
    hit = sub > float(detection_threshold)

    uniq = list(dict.fromkeys(gnames))          # 保持出现顺序
    rows, pct, mean, ncell = [], {}, {}, {}
    for g in uniq:
        m = np.array([x == g for x in gnames])
        n = int(m.sum())
        ncell[g] = n
        h = hit[m]                              # (n, k)
        s = sub[m]
        n_det = h.sum(axis=0).astype(float)
        p = np.divide(n_det, n, out=np.zeros_like(n_det), where=n > 0) * 100.0
        mn = s.mean(axis=0) if n > 0 else np.full(s.shape[1], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            mpos = np.where(n_det > 0, s.sum(axis=0) / np.where(n_det > 0, n_det, 1.0), np.nan)
        for j, k in enumerate(sel):
            rows.append({
                "group": g, "gene": all_genes[k], "n_cells": n,
                "n_detected": int(n_det[j]), "pct_detected": float(p[j]),
                "mean_expr": float(mn[j]), "mean_expr_positive": float(mpos[j]),
            })
        pct[g] = {all_genes[k]: float(p[j]) for j, k in enumerate(sel)}
        mean[g] = {all_genes[k]: float(mn[j]) for j, k in enumerate(sel)}

    table = pd.DataFrame(rows)
    pct_matrix = pd.DataFrame(pct).T
    mean_matrix = pd.DataFrame(mean).T

    out = {
        "table": table,
        "pct_matrix": pct_matrix,
        "mean_matrix": mean_matrix,
        "n_cells": ncell,
        "detection_threshold": float(detection_threshold),
        "detection_rule": (
            f"X > {float(detection_threshold)} 判为检出（默认 0 = 非零即检出）；"
            "分母为该组全部细胞数，不丢细胞、不下采样"),
        "mean_rule": "mean_expr = 全部细胞算术平均（含阴性）；mean_expr_positive = 仅阳性细胞平均",
        "engine": "lib.singlecell.expression_rate",
    }
    if genes is not None and sel_genes_missing:
        out["genes_not_found"] = sorted(set(sel_genes_missing))
    return out


def detectable_genes(rates: Dict[str, Any], gene_list: Optional[Sequence[str]] = None,
                     min_pct: float = 10.0, min_groups: int = 1) -> Dict[str, Any]:
    """判定哪些基因算「可检出」——预检判据「≥N% 风险基因可检出」的执行者。

    判据：在**至少 ``min_groups`` 个分组**中 ``pct_detected >= min_pct``。

    ⚠️ **阈值必须显式给出并随结果输出**。默认 ``min_pct=10`` 是单细胞领域的
    常见惯例（"某基因在该细胞类型表达"的宽松门槛），但它**不是定理**；
    换阈值会改变结论，所以阈值进返回值，不允许只报数字不报门槛。
    """
    if "pct_matrix" not in rates:
        raise ValueError("请传入 expression_rate() 的返回值")
    pm = rates["pct_matrix"]
    gl = [str(g) for g in gene_list] if gene_list is not None else list(pm.columns)
    gl = [g for g in gl if g in pm.columns]
    if not gl:
        raise ValueError("gene_list 与 pct_matrix 无交集")

    sub = pm[gl]
    n_ok = (sub >= float(min_pct)).sum(axis=0)          # 每个基因达标的分组数
    ok = [g for g in gl if int(n_ok[g]) >= int(min_groups)]
    return {
        "detectable": ok,
        "not_detectable": [g for g in gl if g not in set(ok)],
        "n_total": len(gl),
        "n_detectable": len(ok),
        "fraction": (len(ok) / len(gl)) if gl else float("nan"),
        "min_pct": float(min_pct),
        "min_groups": int(min_groups),
        "rule": (f"在 >= {min_groups} 个分组中 pct_detected >= {min_pct}% 判为可检出"),
        "engine": "lib.singlecell.detectable_genes",
    }
