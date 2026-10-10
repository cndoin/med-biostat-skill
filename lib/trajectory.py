"""分析法 ⑭：轨迹推断 / 拟时序（pseudotime）——扩散伪时间 + 最小生成树。

两类做法，适用条件不同：
  1. `diffusion_pseudotime`（DPT, Haghverdi 2016）
     — 用扩散图（diffusion map）捕捉数据的全局几何，按扩散距离定伪时间。
       适合**连续轨迹**（如晶状体上皮 → 纤维细胞分化）。
  2. `mst_pseudotime`
     — 在降维空间上建最小生成树（MST），按图距离定伪时间。
       适合**有明显分支**的轨迹（如细胞命运分叉）。

⚠️ 三条必须说清的话：
  1. 伪时间是**相对顺序**，不是真实时间；不能说「第 3 天」。
  2. **起点必须人为指定**。不给 root 就用「最极端的那个点」自动选，
     结果可能与生物学方向相反——必须人工复核。
  3. 轨迹推断**不能证明因果**。它说明「这些细胞可以排成一条连续路径」，
     不等于「这个基因驱动了分化」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.spatial.distance import pdist, squareform


def _knn_affinity(X: np.ndarray, k: int = 15, sigma: Optional[float] = None) -> np.ndarray:
    """自适应高斯核亲和矩阵（与 diffusion map 的标准构造一致）。"""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    n = X.shape[0]
    k = int(min(k, n - 1))
    if sigma is None:
        # 局部尺度：到第 k 个邻居的距离
        knn_d = np.sqrt(np.sort(d2, axis=1)[:, k])
        sigma = np.median(knn_d[knn_d > 0]) if np.any(knn_d > 0) else 1.0
        sigma = float(max(sigma, 1e-12))
    A = np.exp(-d2 / (sigma ** 2))
    np.fill_diagonal(A, 0.0)
    return A


def diffusion_map(X: np.ndarray, n_components: int = 10, k: int = 15,
                  alpha: float = 0.5) -> Dict[str, Any]:
    """扩散图降维（Coifman & Lafon 2006）。

    步骤：亲和矩阵 → 按 α 做密度归一化 → 行归一化成马尔可夫链 → 特征分解。
    返回扩散成分（非平凡的、按特征值降序）。
    """
    X = np.asarray(X, dtype=float)
    A = _knn_affinity(X, k=k)
    d = A.sum(axis=1)
    d = np.where(d == 0, 1.0, d)
    if alpha > 0:
        A = A / (np.outer(d ** alpha, d ** alpha))  # 密度归一化，消除采样密度影响
        d = A.sum(axis=1)
        d = np.where(d == 0, 1.0, d)
    K = A / d[:, None]                              # 行随机矩阵
    vals, vecs = np.linalg.eig(K)
    order = np.argsort(-np.abs(vals))
    vals = np.abs(vals[order])
    vecs = np.real(vecs[:, order])
    # 第 0 个是平凡解（全 1，特征值 1），丢弃
    comps = vecs[:, 1:1 + n_components]
    return {"components": comps, "eigenvalues": vals[1:1 + n_components],
            "full_eigenvalues": vals, "alpha": alpha, "k": k}


def diffusion_pseudotime(X: np.ndarray, root: Optional[int] = None,
                         n_components: int = 10, k: int = 15) -> Dict[str, Any]:
    """DPT 伪时间：扩散空间中到 root 的扩散距离。

    :param root: 起点细胞索引；None 时取第一扩散成分的极值点（**必须人工复核**）
    """
    dm = diffusion_map(X, n_components=n_components, k=k)
    C = dm["components"]
    lam = np.asarray(dm["eigenvalues"], dtype=float)
    # 扩散距离必须按谱加权：D² = Σ_j [λ_j/(1−λ_j)]·(ψ_j(x) − ψ_j(y))²。
    # 不加权等于把所有扩散成分一视同仁，快衰减的高频成分（噪声方向）
    # 会占主导，伪时间不再反映数据的主几何（实测伪时间在轨迹起点附近
    # 直接跳到 1，序完全错乱）。λ/(1−λ) 是标准 DPT 的权重。
    with np.errstate(divide="ignore", invalid="ignore"):
        w = lam / (1.0 - lam)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    C = C * np.sqrt(w)[None, :]
    if root is None:
        root = int(np.argmin(C[:, 0]))   # 第一扩散成分的最小值端
    d = np.linalg.norm(C - C[root], axis=1)
    pt = d / d.max() if d.max() > 0 else np.zeros_like(d)
    return {"pseudotime": pt, "root": int(root), "root_auto": root is None,
            "components": C, "eigenvalues": dm["eigenvalues"],
            "rule": "扩散空间中到 root 的欧氏距离，归一化到 [0,1]"}


def mst_pseudotime(X: np.ndarray, root: Optional[int] = None,
                   k: int = 10) -> Dict[str, Any]:
    """MST 伪时间：kNN 图上建最小生成树，按图上的最短路径距离定序。

    分支识别：MST 上**度 ≥ 3** 的节点即分支点。
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    d = squareform(pdist(X))
    # 只保留 kNN 边（其余设 0 = 不连通），再建 MST
    k = int(min(k, n - 1))
    mask = np.zeros_like(d, dtype=bool)
    for i in range(n):
        nb = np.argsort(d[i])[1:k + 1]
        mask[i, nb] = True
        mask[nb, i] = True
    D = np.where(mask, d, 0.0)
    mst = minimum_spanning_tree(D).toarray()
    G = mst + mst.T
    G[G == 0] = 0
    dist = shortest_path(G, directed=False)
    if root is None:
        # 取 MST 的一个**端点**（度为 1）作为起点——比随机选更符合直觉
        deg = np.count_nonzero(G > 0, axis=1)
        ends = np.flatnonzero(deg == 1)
        root = int(ends[0]) if ends.size else int(np.argmax(dist.sum(axis=1)))
    pt = dist[root]
    pt = np.nan_to_num(pt, nan=0.0, posinf=0.0)
    mx = pt.max()
    pt = pt / mx if mx > 0 else pt
    deg = np.count_nonzero(G > 0, axis=1)
    return {
        "pseudotime": pt, "root": int(root),
        "branch_points": [int(i) for i in np.flatnonzero(deg >= 3)],
        "endpoints": [int(i) for i in np.flatnonzero(deg == 1)],
        "mst": G, "graph_distance": dist,
        "rule": "kNN 图上 MST 的最短路径距离，归一化到 [0,1]",
    }


def branch_assignment(pseudotime_result: Dict[str, Any], labels: Sequence[Any],
                      n_neighbors: int = 5) -> pd.DataFrame:
    """把细胞按伪时间排序后，统计各簇的伪时间分布（看分化顺序）。"""
    pt = np.asarray(pseudotime_result["pseudotime"], dtype=float)
    lab = np.asarray(labels, dtype=object)
    rows = []
    for lev in pd.unique(lab):
        x = pt[lab == lev]
        rows.append({"cluster": str(lev), "n": int(x.size),
                     "pseudotime_mean": float(np.mean(x)),
                     "pseudotime_median": float(np.median(x)),
                     "pseudotime_sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                     "q10": float(np.quantile(x, 0.1)), "q90": float(np.quantile(x, 0.9))})
    return pd.DataFrame(rows).sort_values("pseudotime_mean").reset_index(drop=True)


def velocity_free_genes(expr: pd.DataFrame, pseudotime: Sequence[float],
                        min_cells: int = 10) -> pd.DataFrame:
    """沿伪时间找单调变化的基因（轨迹相关基因的粗筛）。

    用 Spearman 相关（不要求线性）+ BH 校正。
    ⚠️ 这是**相关性**筛查，不是因果；而且伪时间本身有噪声，
    相关性强不等于真的随分化变化。
    """
    from scipy import stats as _st

    pt = np.asarray(pseudotime, dtype=float)
    mat = expr.to_numpy(dtype=float)
    rows = []
    for j in range(mat.shape[1]):
        x = mat[:, j]
        if np.count_nonzero(x) < min_cells:
            continue
        rho, p = _st.spearmanr(pt, x)
        rows.append({"gene": str(expr.columns[j]), "rho": float(rho), "pvalue": float(p)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    p = df["pvalue"].to_numpy()
    order = np.argsort(p)
    q = np.empty_like(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    df["padj"] = np.clip(q, 0, 1)
    df["direction"] = np.where(df["rho"] > 0, "up", "down")
    return df.reindex(df["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)
