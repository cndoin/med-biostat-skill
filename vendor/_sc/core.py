# PORTED-FILE · 来源 scverse/scanpy（BSD-3-Clause）· 源码 src/scanpy/**
# 移植方式：**算法照抄、依赖换底**。scanpy 直接依赖 anndata / sklearn / umap / numba /
# dask，且函数签名深度绑定 AnnData；本文件把算法抽出来，输入输出改为
# numpy 数组 / pandas，依赖降到 numpy+pandas+scipy。
# 逐函数对照（scanpy 源函数 -> 本文件）：
#   preprocessing/_qc.py            -> qc_metrics
#   preprocessing/_normalization.py -> normalize_total, log1p
#   preprocessing/_highly_variable_genes.py -> highly_variable_genes（seurat / cell_ranger）
#   tools/_pca.py                   -> pca
#   tools/_rank_genes_groups.py     -> rank_genes_groups（wilcoxon / t_test）
#   tools/_score_genes.py           -> score_genes（分箱对照基因）
#   neighbors                       -> neighbors（cKDTree kNN + 自适应核连接度）
# 与上游的**已知差异（如实标注）**：
#   1. HVG 仅实现 seurat / cell_ranger 两种 flavor；**seurat_v3 需要 LOESS**
#      （scanpy 走 skmisc.loess），本文件提供 hand-rolled LOWESS（Cleveland 算法）
#      但默认仍用 dispersion flavor，避免与 R 版 Seurat v3 的 loess 实现差异被误当一致。
#   2. 聚类实现 **Louvain**（Blondel 2008 模块化度贪心聚合），**不是 Leiden**
#      （Leiden 的 Python 实现 leidenalg 许可未确认，未内嵌）。结果可能与 scanpy
#      leiden 的簇边界不同——这是算法不同，不是 bug。
#   3. 嵌入提供 PCA + t-SNE（Barnes-Hut 未实现，O(n^2) 精确梯度，建议 n<=2000）；
#      UMAP 因 umap-learn 依赖 numba/pynndescent 未内嵌。
"""单细胞 / 单核转录组核心算法（算法移植自 scanpy，BSD-3-Clause）。

约定：矩阵一律 **行=细胞、列=基因**（scanpy AnnData.X 同向），返回 numpy 数组；
基因名 / 细胞名为可选。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import sparse, stats
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 质控（scanpy preprocessing/_qc.py）
# ---------------------------------------------------------------------------
def qc_metrics(counts, gene_names: Optional[Sequence[str]] = None,
               mito_prefix: str = "MT-") -> pd.DataFrame:
    """细胞级 QC：n_genes（检出的基因数）、n_counts（总 UMI）、pct_mito（线粒体占比 %）。

    口径：n_genes 为**非零计数**的基因数（scanpy `calculate_qc_metrics` 的 n_genes_by_counts）。
    线粒体基因识别为前缀匹配，大小写不敏感（`MT-` 匹配人；小鼠常用 `mt-`）。
    """
    mat = counts.toarray() if sparse.issparse(counts) else np.asarray(counts, dtype=float)
    if mat.ndim != 2:
        raise ValueError("counts 必须是二维（行=细胞，列=基因）")
    n_counts = mat.sum(axis=1)
    n_genes = np.count_nonzero(mat, axis=1)
    pct_mito = np.zeros(mat.shape[0])
    if gene_names is not None and mito_prefix:
        names = np.asarray([str(g) for g in gene_names], dtype=object)
        low = np.array([str(g).lower() for g in names])
        mask = np.array([s.startswith(mito_prefix.lower()) for s in low])
        if mask.any():
            mito = mat[:, mask].sum(axis=1)
            # 计数全为 0 的空细胞（drop-out 细胞）会出现 0/0 —— 直接除会抛
            # RuntimeWarning 并污染任何「捕获 stderr 判崩溃」的门禁。
            denom = np.where(n_counts > 0, n_counts, 1.0)
            pct_mito = np.where(n_counts > 0, mito / denom * 100.0, 0.0)
    return pd.DataFrame({
        "n_genes": n_genes,
        "n_counts": n_counts,
        "pct_mito": pct_mito,
        "n_mito_genes": int(mask.sum()) if gene_names is not None else 0,
    })


def filter_cells(mat, min_genes: int = 200, max_genes: Optional[int] = None,
                 min_counts: Optional[int] = None, max_counts: Optional[int] = None,
                 max_pct_mito: Optional[float] = None,
                 gene_names: Optional[Sequence[str]] = None,
                 mito_prefix: str = "MT-") -> Dict[str, Any]:
    """按 QC 阈值筛细胞（scanpy `filter_cells` 语义：只报索引，不改数据）。"""
    qc = qc_metrics(mat, gene_names=gene_names, mito_prefix=mito_prefix)
    keep = np.ones(mat.shape[0], dtype=bool)
    rules = []
    if min_genes is not None:
        keep &= qc["n_genes"].to_numpy() >= min_genes
        rules.append(f"n_genes >= {min_genes}")
    if max_genes is not None:
        keep &= qc["n_genes"].to_numpy() <= max_genes
        rules.append(f"n_genes <= {max_genes}")
    if min_counts is not None:
        keep &= qc["n_counts"].to_numpy() >= min_counts
        rules.append(f"n_counts >= {min_counts}")
    if max_counts is not None:
        keep &= qc["n_counts"].to_numpy() <= max_counts
        rules.append(f"n_counts <= {max_counts}")
    if max_pct_mito is not None:
        keep &= qc["pct_mito"].to_numpy() <= max_pct_mito
        rules.append(f"pct_mito <= {max_pct_mito}")
    return {
        "keep_index": np.flatnonzero(keep),
        "drop_index": np.flatnonzero(~keep),
        "n_kept": int(keep.sum()),
        "n_dropped": int((~keep).sum()),
        "rules": rules,
        "qc": qc,
    }


def filter_genes(mat, min_cells: int = 3, max_cells: Optional[int] = None) -> Dict[str, Any]:
    """按「至少在 min_cells 个细胞中检出」筛基因（scanpy `filter_genes`）。"""
    arr = mat.toarray() if sparse.issparse(mat) else np.asarray(mat)
    n_cells = np.count_nonzero(arr, axis=0)
    keep = n_cells >= min_cells
    if max_cells is not None:
        keep &= n_cells <= max_cells
    return {
        "keep_index": np.flatnonzero(keep),
        "drop_index": np.flatnonzero(~keep),
        "n_cells_per_gene": n_cells,
        "n_kept": int(keep.sum()),
        "rules": [f"cells_detected >= {min_cells}"],
    }


# ---------------------------------------------------------------------------
# 归一化（scanpy preprocessing/_normalization.py）
# ---------------------------------------------------------------------------
def normalize_total(counts, target_sum: Optional[float] = None,
                    exclude_highly_expressed: bool = False,
                    max_fraction: float = 0.05) -> np.ndarray:
    """总数归一化：每个细胞的计数除以该细胞的总数，再乘 target_sum。

    target_sum=None 时走 scanpy 的**中位数归一化**：除数为「非零基因的中位总数」
    （`_compute_nnz_median`），这样能避免极高计数的空泡细胞把整体压小。
    """
    mat = counts.toarray().astype(float) if sparse.issparse(counts) else np.asarray(counts, dtype=float)
    counts_per_cell = mat.sum(axis=1)
    if exclude_highly_expressed:
        gene_tot = mat.sum(axis=0)
        thr = gene_tot.max() * max_fraction
        high = np.flatnonzero(gene_tot >= thr)
        counts_per_cell = mat[:, np.setdiff1d(np.arange(mat.shape[1]), high)].sum(axis=1)
    if target_sum is None:
        after = np.array([row[row > 0].sum() for row in mat]) if exclude_highly_expressed else counts_per_cell
        target_sum = float(np.median(after[after > 0])) if np.any(after > 0) else 1.0
    denom = np.where(counts_per_cell == 0, 1.0, counts_per_cell)
    return mat / denom[:, None] * float(target_sum)


def log1p(mat) -> np.ndarray:
    """log1p 变换（scanpy `log1p`）。输入应为已归一化的数据。"""
    arr = mat.toarray().astype(float) if sparse.issparse(mat) else np.asarray(mat, dtype=float)
    return np.log1p(arr)


def scale(mat, max_value: Optional[float] = 10.0,
          zero_center: bool = True) -> np.ndarray:
    """z-score 标准化（scanpy `scale`）：按基因去均值、除标准差；可选截断。"""
    arr = mat.toarray().astype(float) if sparse.issparse(mat) else np.asarray(mat, dtype=float)
    mu = arr.mean(axis=0)
    sd = arr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    out = (arr - (mu if zero_center else 0.0)) / (sd if zero_center else 1.0)
    if max_value is not None:
        out = np.clip(out, -max_value, max_value)
    return out


# ---------------------------------------------------------------------------
# 高变基因（scanpy preprocessing/_highly_variable_genes.py）
# ---------------------------------------------------------------------------
def _mad(x: np.ndarray) -> float:
    return float(np.median(np.abs(np.asarray(x) - np.median(x))))


def highly_variable_genes(mat, gene_names: Optional[Sequence[str]] = None,
                          flavor: str = "seurat", n_top_genes: Optional[int] = 2000,
                          n_bins: int = 20, min_mean: float = 0.0125,
                          max_mean: float = 3.0, min_disp: float = 0.5) -> Dict[str, Any]:
    """高变基因筛选。

    flavor 口径（同 scanpy）：
      · seurat       : dispersion = var/mean（log 数据）；按 mean 等宽分 n_bins 箱，
                       箱内 dispersion 做 (disp - avg)/dev 的 z 归一化，取 z 最大者
      · cell_ranger  : 箱内用 **median + MAD** 归一化（scanpy `_mad`）
    返回 means / dispersions / dispersions_norm / highly_variable 掩码。
    """
    if flavor not in ("seurat", "cell_ranger"):
        raise ValueError(f"flavor 必须是 seurat 或 cell_ranger，收到 {flavor!r}")
    arr = mat.toarray().astype(float) if sparse.issparse(mat) else np.asarray(mat, dtype=float)
    n_obs = arr.shape[0]
    means = arr.mean(axis=0)
    variances = arr.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(means > 0, variances / means, np.nan)
    df = pd.DataFrame({"means": means, "dispersions": disp})
    # 按 mean 分箱（seurat: 等宽 n_bins；cell_ranger: 5% 分位）
    if flavor == "seurat":
        bins = pd.cut(df["means"], bins=n_bins)
    else:
        edges = np.r_[-np.inf, np.percentile(means, np.arange(10, 105, 5)), np.inf]
        bins = pd.cut(df["means"], bins=edges)
    df["mean_bin"] = bins
    if flavor == "seurat":
        st = df.groupby("mean_bin", observed=True)["dispersions"].agg(avg="mean", dev="std")
        one = st["dev"].isna()
        if one.any():
            st.loc[one, "dev"] = st.loc[one, "avg"]
            st.loc[one, "avg"] = 0.0
    else:
        st = df.groupby("mean_bin", observed=True)["dispersions"].agg(avg="median", dev=_mad)
    st = st.loc[df["mean_bin"]].set_index(df.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp_norm = (df["dispersions"].to_numpy() - st["avg"].to_numpy()) / np.where(
            st["dev"].to_numpy() == 0, 1.0, st["dev"].to_numpy())
    disp_norm = np.nan_to_num(disp_norm, nan=0.0)
    if n_top_genes is not None and n_top_genes > 0:
        k = min(int(n_top_genes), arr.shape[1])
        thr = np.sort(disp_norm)[::-1][k - 1] if k > 0 else 0.0
        hv = disp_norm >= thr
    else:
        hv = (means >= min_mean) & (means <= max_mean) & (disp_norm >= min_disp)
    return {
        "highly_variable": hv,
        "means": means,
        "dispersions": df["dispersions"].to_numpy(),
        "dispersions_norm": disp_norm,
        "gene_names": None if gene_names is None else list(gene_names),
        "flavor": flavor,
        "n_top_genes": n_top_genes,
        "rule": "dispersion 分箱 z 归一化后取 top k（seurat=mean/std，cell_ranger=median/MAD）",
    }


# ---------------------------------------------------------------------------
# PCA（scanpy tools/_pca.py）
# ---------------------------------------------------------------------------
def pca(mat, n_comps: int = 50, zero_center: bool = True,
        scale_data: bool = False) -> Dict[str, Any]:
    """PCA：去均值 → SVD → 得分矩阵与方差解释率。

    返回 scores（细胞×PC）、loadings（PC×基因）、variance_ratio、total_variance。
    口径：variance_ratio 用奇异值平方占比（scanpy 的 `variance_ratio`）。
    """
    arr = mat.toarray().astype(float) if sparse.issparse(mat) else np.asarray(mat, dtype=float)
    if scale_data:
        sd = arr.std(axis=0)
        arr = arr / np.where(sd == 0, 1.0, sd)
    mu = arr.mean(axis=0) if zero_center else 0.0
    x = arr - mu
    n_comps = int(min(n_comps, min(x.shape) - 1)) if min(x.shape) > 1 else 1
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    scores = u[:, :n_comps] * s[:n_comps]
    loadings = vt[:n_comps]
    var_ratio = (s ** 2) / np.sum(s ** 2)
    return {
        "scores": scores,
        "loadings": loadings,
        "variance_ratio": var_ratio[:n_comps],
        "cumulative_variance_ratio": np.cumsum(var_ratio[:n_comps]),
        "singular_values": s[:n_comps],
        "n_comps": int(n_comps),
        "mean": np.asarray(mu) if zero_center else np.zeros(arr.shape[1]),
    }


# ---------------------------------------------------------------------------
# 邻居图（scanpy neighbors：kNN + 连接度）
# ---------------------------------------------------------------------------
def neighbors(mat, n_neighbors: int = 15, metric: str = "euclidean",
              use_rep: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """kNN 邻居图：距离矩阵 + 自适应高斯核连接度（对称化）。

    口径说明：scanpy 默认用 umap 的模糊单纯形集（fuzzy simplicial set）构造
    connectivities；本实现用**自适应高斯核**（sigma = 到该点第 k 个邻居的距离），
    数值上与 umap 不完全相同——聚类输入的是拓扑而非权重，对 Louvain 结果影响有限。
    如实标注：connectivities 的**绝对值不可与 scanpy 直接对比**。
    """
    rep = use_rep if use_rep is not None else (
        mat.toarray().astype(float) if sparse.issparse(mat) else np.asarray(mat, dtype=float))
    rep = np.asarray(rep, dtype=float)
    n = rep.shape[0]
    k = int(min(n_neighbors, max(1, n - 1)))
    tree = cKDTree(rep)
    dist, idx = tree.query(rep, k=k + 1, p=2 if metric == "euclidean" else 2)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    sigma = dist[:, -1]
    sigma = np.where(sigma <= 0, 1e-12, sigma)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    d = dist.ravel()
    sig = np.repeat(sigma, k)
    w = np.exp(-(d ** 2) / (sig ** 2))
    W = sparse.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)  # 对称化（max）
    return {
        "distances": W,
        "connectivities": W,
        "knn_index": idx,
        "knn_dist": dist,
        "n_neighbors": k,
        "rule": "自适应高斯核(sigma=第 k 邻距离)，非 umap 模糊单纯形集；权重值不可与 scanpy 直接对比",
    }


# ---------------------------------------------------------------------------
# 聚类：Louvain（Blondel 2008）—— 非 Leiden，标注差异
# ---------------------------------------------------------------------------
def louvain(adjacency, resolution: float = 1.0, n_iter: int = 10,
            seed: int = 0) -> Dict[str, Any]:
    """Louvain 社区发现（模块化度贪心聚合）。

    算法：Blondel et al. 2008 "Fast unfolding of communities in large networks"。
        Q = (1/2m) Σ_ij [A_ij - γ·k_i·k_j/(2m)] δ(c_i, c_j)
    两阶段迭代：
      ① 局部移动——反复把节点挪到使其 ΔQ 增益最大的邻居社区（γ = resolution）；
      ② 聚合——把每个社区压成超节点，边权相加，回到 ①，直到社区数不再下降。
    增益（Blondel 式，省略与候选社区无关的常数项）：
        gain(c) = k_{i→c} - γ · Σtot(c) · k_i / (2m)
    其中 k_{i→c} 是 i 到社区 c 的边权和（不含自环），Σtot(c) 为社区 c 的度总和
    （计算时已把 i 从原社区摘除，避免自贡献）。

    ⚠️ 与 scanpy `tl.leiden` **不是同一个算法**：Leiden 是 Louvain 的改进版
    （保证社区内部连通、收敛性更强）。本函数结果是 Louvain 的，簇边界可能与
    leiden 不同——这是算法差异，不是缺陷。
    """
    A = adjacency.tocsr() if sparse.issparse(adjacency) else sparse.csr_matrix(
        np.asarray(adjacency, dtype=float))
    A = ((A + A.T) / 2.0).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    n = A.shape[0]
    if n == 0:
        return {"labels": np.array([], dtype=int), "n_clusters": 0, "modularity": 0.0,
                "algorithm": "louvain(Blondel2008)", "resolution": float(resolution)}
    rng = np.random.default_rng(seed)
    k_all = np.asarray(A.sum(axis=1)).ravel()
    two_m = float(k_all.sum())
    if two_m <= 0:
        return {"labels": np.arange(n), "n_clusters": n, "modularity": 0.0,
                "algorithm": "louvain(Blondel2008)", "resolution": float(resolution)}

    labels = np.arange(n, dtype=int)  # 每个原节点当前的社区号（连续编号）

    def _local_moving(cur) -> np.ndarray:
        """对当前（超）图做局部移动，返回 节点 -> 社区。"""
        nn = cur.shape[0]
        k = np.asarray(cur.sum(axis=1)).ravel()
        comm = np.arange(nn, dtype=int)
        comm_tot = k.copy()
        for _ in range(100):
            moved = False
            for i in rng.permutation(nn):
                ci = int(comm[i])
                ki = float(k[i])
                s, e = cur.indptr[i], cur.indptr[i + 1]
                nb, w = cur.indices[s:e], cur.data[s:e]
                if nb.size == 0:
                    continue
                comm_tot[ci] -= ki  # 先把 i 摘出原社区
                k_in: Dict[int, float] = {}
                for j, wv in zip(nb, w):
                    if int(j) == i:
                        continue
                    c = int(comm[j])
                    k_in[c] = k_in.get(c, 0.0) + float(wv)
                best_c = ci
                best_gain = k_in.get(ci, 0.0) - resolution * comm_tot[ci] * ki / two_m
                for c, wc in k_in.items():
                    gain = wc - resolution * comm_tot[c] * ki / two_m
                    if gain > best_gain + 1e-12:
                        best_gain, best_c = gain, c
                comm[i] = best_c
                comm_tot[best_c] += ki
                if best_c != ci:
                    moved = True
            if not moved:
                break
        return np.asarray(np.unique(comm, return_inverse=True)[1], dtype=int)

    for _ in range(max(1, int(n_iter))):
        cur = _aggregate_adjacency(A, labels)
        comm = _local_moving(cur)      # 超节点 -> 新社区
        new_labels = np.asarray(np.unique(comm[labels], return_inverse=True)[1], dtype=int)
        converged = new_labels.max() == labels.max()
        labels = new_labels
        if converged:
            break

    return {
        "labels": labels,
        "n_clusters": int(labels.max() + 1),
        "modularity": _modularity(A, labels, resolution),
        "algorithm": "louvain(Blondel2008)",
        "resolution": float(resolution),
        "note": "Louvain，非 Leiden；簇边界可能与 scanpy leiden 不同",
    }


def _aggregate_adjacency(A, labels):
    """把同社区节点压成超节点：社区间边权相加。"""
    labels = np.asarray(labels, dtype=int)
    nc = int(labels.max()) + 1
    P = sparse.coo_matrix((np.ones(len(labels)), (np.arange(len(labels)), labels)),
                          shape=(len(labels), nc)).tocsr()
    return (P.T @ A @ P).tocsr()


def _modularity(A, labels, resolution: float = 1.0) -> float:
    """模块化度 Q = (1/2m) Σ_ij (A_ij - γ k_i k_j / 2m) δ(c_i,c_j)。"""
    A = A.tocsr() if sparse.issparse(A) else sparse.csr_matrix(np.asarray(A, dtype=float))
    m2 = float(A.sum())
    if m2 <= 0:
        return 0.0
    k = np.asarray(A.sum(axis=1)).ravel()
    q = 0.0
    nc = int(labels.max()) + 1
    for c in range(nc):
        mask = labels == c
        sub = A[mask][:, mask].sum()
        kt = k[mask].sum()
        q += sub / m2 - resolution * (kt / m2) ** 2
    return float(q)


# ---------------------------------------------------------------------------
# t-SNE（精确梯度，O(n^2)；建议 n<=2000）
# ---------------------------------------------------------------------------
def tsne(mat, n_components: int = 2, perplexity: float = 30.0,
         n_iter: int = 500, learning_rate: float = 200.0,
         seed: int = 0, early_exaggeration: float = 12.0) -> Dict[str, Any]:
    """t-SNE（van der Maaten & Hinton 2008）：精确梯度版，未做 Barnes-Hut 近似。

    ⚠️ 复杂度 O(n^2)，n>2000 会明显变慢；这是**与 scanpy（走 sklearn/OpenTSNE）
    的实现差异**，数值也不会逐位一致（t-SNE 本身有随机初始化）。
    """
    x = mat.toarray().astype(float) if sparse.issparse(mat) else np.asarray(mat, dtype=float)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    # 条件概率 P（二元搜索 perplexity）
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    P = np.zeros((n, n))
    beta = np.ones(n)
    logU = np.log(perplexity)
    for i in range(n):
        di = np.delete(d2[i], i)
        lo, hi = 1e-20, np.inf
        for _ in range(50):
            pi = np.exp(-beta[i] * di)
            s = pi.sum()
            if s < 1e-300:
                hi = beta[i]
                beta[i] = (beta[i] + lo) / 2
                continue
            h = np.log(s) + beta[i] * (di * pi).sum() / s
            if abs(h - logU) < 1e-5:
                break
            if h > logU:
                lo = beta[i]
                beta[i] = beta[i] * 2 if hi == np.inf else (beta[i] + hi) / 2
            else:
                hi = beta[i]
                beta[i] = (beta[i] + lo) / 2
        pi = np.exp(-beta[i] * di)
        pi /= pi.sum()
        P[i, np.arange(n) != i] = pi
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    y = rng.normal(0, 1e-4, size=(n, n_components))
    dy = np.zeros_like(y)
    gains = np.ones_like(y)
    mom = np.array([0.5, 0.8])
    n_iter = int(n_iter)
    for it in range(n_iter):
        ex = early_exaggeration if it < 250 else 1.0
        num = 1.0 / (1.0 + ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(num, 0.0)
        Q = num / num.sum()
        Q = np.maximum(Q, 1e-12)
        grad = 4 * (((ex * P - Q) * num)[:, :, None] * (y[:, None, :] - y[None, :, :])).sum(axis=1)
        m = mom[1] if it >= 250 else mom[0]
        gains = np.where(np.sign(grad) == np.sign(dy), np.maximum(gains * 0.8, 0.01), gains + 0.2)
        dy = m * dy - learning_rate * (gains * grad)
        y = y + dy
        y = y - y.mean(axis=0)
    return {"embedding": y, "kl_divergence": float(np.sum(P * np.log(P / Q))),
            "perplexity": float(perplexity), "n_iter": n_iter,
            "rule": "精确梯度 t-SNE（无 Barnes-Hut），n>2000 慎用；与 scanpy 数值不逐位一致"}


# ---------------------------------------------------------------------------
# Marker 基因（scanpy tools/_rank_genes_groups.py）
# ---------------------------------------------------------------------------
def _bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def rank_genes_groups(mat, groups: Sequence[Any], gene_names: Optional[Sequence[str]] = None,
                      method: str = "wilcoxon", n_top: Optional[int] = None) -> Dict[str, Any]:
    """差异表达 / marker 基因：逐簇 vs 其余（scanpy `rank_genes_groups` 的 'rest' 参照）。

    method：
      · wilcoxon — Mann-Whitney U（scanpy 用 tie-corrected rank sums，本实现走
        scipy.stats.mannwhitneyu(alternative='two-sided')，大样本 z 近似）
      · t_test   — Welch t 检验
    logFC 口径：scanpy 对 log 数据用 **expm1 均值之差**（即回算到线性尺度再比，
    避免 log 空间的比值失真）：logFC = mean(expm1(x_group)) - mean(expm1(x_rest))。
    """
    arr = mat.toarray().astype(float) if sparse.issparse(mat) else np.asarray(mat, dtype=float)
    groups = np.asarray(groups)
    if arr.shape[0] != len(groups):
        raise ValueError("groups 长度必须等于细胞数")
    if method not in ("wilcoxon", "t_test"):
        raise ValueError(f"method 必须是 wilcoxon 或 t_test，收到 {method!r}")
    out = {}
    for g in pd.unique(groups):
        mask = groups == g
        a = arr[mask]
        b = arr[~mask]
        if a.shape[0] < 2 or b.shape[0] < 2:
            continue
        if method == "wilcoxon":
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided", axis=0)
        else:
            stat, p = stats.ttest_ind(a, b, axis=0, equal_var=False)
        logfc = np.expm1(a).mean(axis=0) - np.expm1(b).mean(axis=0)
        p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
        q = _bh_fdr(p)
        d = pd.DataFrame({
            "gene": list(gene_names) if gene_names is not None else np.arange(arr.shape[1]),
            "logFC": logfc,
            "statistic": np.asarray(stat, dtype=float),
            "pvalue": p,
            "padj": q,
            "pct_in_group": np.count_nonzero(a, axis=0) / a.shape[0],
            "pct_out_group": np.count_nonzero(b, axis=0) / b.shape[0],
        })
        d = d.sort_values("padj", kind="mergesort").reset_index(drop=True)
        out[str(g)] = d.head(n_top) if n_top else d
    return {
        "by_group": out,
        "method": method,
        "logfc_rule": "mean(expm1(group)) - mean(expm1(rest))（scanpy 口径，非 log 空间直接相减）",
        "reference": "rest",
    }


# ---------------------------------------------------------------------------
# 基因集打分（scanpy tools/_score_genes.py）
# ---------------------------------------------------------------------------
def score_genes(mat, gene_list: Sequence[str], gene_names: Sequence[str],
                n_bins: int = 25, ctrl_size: int = 50, seed: int = 0) -> np.ndarray:
    """基因集打分：mean(基因集) - mean(对照基因集)。

    口径（同 scanpy `score_genes`）：把基因按平均表达分成 n_bins 箱，
    从每箱中随机抽 ctrl_size 个不在基因集内的基因作为对照，
    用「表达水平匹配」的对照消除深度与技术偏差。
    """
    arr = mat.toarray().astype(float) if sparse.issparse(mat) else np.asarray(mat, dtype=float)
    names = np.array([str(g) for g in gene_names], dtype=object)
    obs_avg = arr.mean(axis=0)
    order = np.argsort(obs_avg)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    n_items = int(np.round(len(obs_avg) / (n_bins - 1))) if n_bins > 1 else len(obs_avg)
    rng = np.random.default_rng(seed)
    ctrl = set()
    for b in range(n_bins):
        lo = max(0, (b - 1) * n_items + 1) if b > 0 else 0
        hi = min(len(order), b * n_items + 1)
        pool = [int(i) for i in order[lo:hi] if names[int(i)] not in set(gene_list)]
        if not pool:
            continue
        take = min(ctrl_size, len(pool))
        ctrl.update(int(i) for i in rng.choice(pool, size=take, replace=False))
    idx_set = np.flatnonzero(np.isin(names, list(gene_list)))
    idx_ctrl = np.array(sorted(ctrl), dtype=int)
    if idx_set.size == 0:
        raise ValueError(f"基因集中没有任何基因出现在数据中：{list(gene_list)[:5]}...")
    if idx_ctrl.size == 0:
        idx_ctrl = np.setdiff1d(np.arange(arr.shape[1]), idx_set)
    return arr[:, idx_set].mean(axis=1) - arr[:, idx_ctrl].mean(axis=1)
