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
