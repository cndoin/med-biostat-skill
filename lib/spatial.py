"""分析法 ⑬：空间转录组（空间自相关 + 空间可变基因 + 邻域富集 + 细胞间通讯）。

四类分析，回答的问题不同：
  1. `morans_i`           — 单基因的空间自相关（全局 Moran's I + 置换检验）
  2. `spatial_variable_genes` — 全基因扫描，挑出空间模式最强的基因
  3. `neighborhood_enrichment` — 细胞类型/簇的空间共定位（谁挨着谁）
  4. `ligand_receptor`    — 配体-受体共表达打分（细胞间通讯的粗粒度筛查）

⚠️ 空间数据的**特征决定了能问什么**：
  · Visium（spot 55 µm，含多个细胞）→ 只能做区域层面，不能做单细胞分辨率的推断
  · Xenium / MERFISH（单细胞分辨率）→ 可做细胞邻域与通讯
  用 spot 数据做「细胞间通讯」结论 = 分辨率不匹配，会被审稿人直接质疑。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import sparse, stats as _st


def build_knn_graph(coords, n_neighbors: int = 6,
                    include_self: bool = False) -> "sparse.csr_matrix":
    """按空间坐标建 kNN 图（无权邻接）。"""
    xy = np.asarray(coords, dtype=float)
    n = xy.shape[0]
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2)
    if not include_self:
        np.fill_diagonal(d2, np.inf)
    k = int(min(n_neighbors, n - 1)) if not include_self else int(min(n_neighbors, n))
    idx = np.argsort(d2, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    W = sparse.coo_matrix((np.ones(rows.size), (rows, idx.ravel())),
                          shape=(n, n)).tocsr()
    return W.maximum(W.T)  # 对称化（kNN 不对称）


def morans_i(x: Sequence[float], W, n_perm: int = 999, seed: int = 0) -> Dict[str, Any]:
    """全局 Moran's I 空间自相关 + 置换检验。

    I = (n/S0) · Σ_ij w_ij (x_i - x̄)(x_j - x̄) / Σ_i (x_i - x̄)²

    解读：I 接近 1 = 强空间聚集；接近 0 = 随机；负值 = 空间排斥（棋盘格）。
    **p 值必须用置换检验**（打乱空间位置），不能用正态近似——
    空间数据的独立性假设本来就不成立。
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    Wm = W.toarray() if sparse.issparse(W) else np.asarray(W, dtype=float)
    S0 = float(Wm.sum())
    if S0 == 0:
        raise ValueError("邻接矩阵全为空——检查坐标或 n_neighbors")
    z = x - x.mean()
    denom = float((z ** 2).sum())
    if denom == 0:
        return {"I": float("nan"), "pvalue": float("nan"), "note": "基因表达无变异"}
    I = (n / S0) * float(z @ (Wm @ z)) / denom
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        zp = z[rng.permutation(n)]
        null[b] = (n / S0) * float(zp @ (Wm @ zp)) / denom
    # 双侧 p 值
    p = float((np.sum(np.abs(null - null.mean()) >= abs(I - null.mean())) + 1) / (n_perm + 1))
    return {"I": float(I), "expected_I": float(-1.0 / (n - 1)) if n > 1 else float("nan"),
            "pvalue": p, "n_perm": n_perm,
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1))}


def spatial_variable_genes(expr: pd.DataFrame, coords, n_neighbors: int = 6,
                           n_perm: int = 199, seed: int = 0,
                           top_n: Optional[int] = 50,
                           min_expr_cells: int = 5) -> pd.DataFrame:
    """扫描全部基因的空间自相关，挑出空间可变基因（SVG）。

    ⚠️ 这是一次**数千个基因的置换检验**——必须做多重检验校正（BH），
    否则「最显著的空间基因」里混着大量假阳性。
    """
    W = build_knn_graph(coords, n_neighbors=n_neighbors)
    mat = expr.to_numpy(dtype=float)
    detected = np.count_nonzero(mat, axis=0)
    rows = []
    rng = np.random.default_rng(seed)
    Wm = W.toarray()
    for j in range(mat.shape[1]):
        if detected[j] < min_expr_cells:
            continue
        r = morans_i(mat[:, j], W, n_perm=n_perm, seed=seed)
        rows.append({"gene": str(expr.columns[j]), "I": r["I"], "pvalue": r["pvalue"]})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    p = df["pvalue"].to_numpy()
    order = np.argsort(p)
    q = np.empty_like(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    df["padj"] = np.clip(q, 0, 1)
    df = df.sort_values("I", ascending=False).reset_index(drop=True)
    return df.head(top_n) if top_n else df


def neighborhood_enrichment(labels: Sequence[Any], coords, n_neighbors: int = 10,
                            n_perm: int = 200, seed: int = 0) -> pd.DataFrame:
    """细胞类型/簇的邻域富集：两类的邻接对数是否多于随机。

    用 z 分数 = (观测 - 随机均值) / 随机标准差，随机化方式是**打乱标签**。
    """
    lab = np.asarray(labels, dtype=object)
    W = build_knn_graph(coords, n_neighbors=n_neighbors)
    Wm = (W.toarray() > 0).astype(float)
    np.fill_diagonal(Wm, 0.0)
    levels = list(pd.unique(lab))
    rng = np.random.default_rng(seed)

    def _counts(l: np.ndarray) -> np.ndarray:
        C = np.zeros((len(levels), len(levels)))
        for a in range(len(levels)):
            ia = np.flatnonzero(l == levels[a])
            for b in range(a, len(levels)):
                ib = np.flatnonzero(l == levels[b])
                c = float(Wm[np.ix_(ia, ib)].sum())
                if a == b:
                    c /= 2.0   # 同类内部每对被数两次
                C[a, b] = C[b, a] = c
        return C

    obs = _counts(lab)
    null = np.stack([_counts(rng.permutation(lab)) for _ in range(n_perm)])
    mu, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
    rows = []
    for a in range(len(levels)):
        for b in range(a, len(levels)):
            z = (obs[a, b] - mu[a, b]) / sd[a, b] if sd[a, b] > 0 else 0.0
            rows.append({"type_a": str(levels[a]), "type_b": str(levels[b]),
                         "observed": float(obs[a, b]), "expected": float(mu[a, b]),
                         "z": float(z)})
    return pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)


def ligand_receptor(expr: pd.DataFrame, labels: Sequence[Any],
                    pairs: Dict[str, Tuple[str, str]],
                    min_frac: float = 0.1) -> pd.DataFrame:
    """配体-受体共表达打分（CellChat 风格的粗粒度筛查）。

    :param pairs: {通路名: (配体基因, 受体基因)}
    打分 = mean(配体在 A 类的表达) × mean(受体在 B 类的表达)。

    ⚠️ 这是**共表达提示**，不是通讯概率：不做置换检验、不校正背景、
    也不考虑分泌/膜定位。正式结论需要 CellChat / NicheNet 级别的模型，
    而且**spot 级数据（Visium）不能做细胞分辨率的通讯推断**。
    """
    lab = np.asarray(labels, dtype=object)
    levels = list(pd.unique(lab))
    genes = [str(g) for g in expr.columns]
    mat = expr.to_numpy(dtype=float)
    rows = []
    for name, (lig, rec) in pairs.items():
        if lig not in genes or rec not in genes:
            continue
        li = genes.index(lig)
        ri = genes.index(rec)
        for a in levels:
            for b in levels:
                la = mat[lab == a, li]
                rb = mat[lab == b, ri]
                fa = float(np.mean(la > 0))
                fb = float(np.mean(rb > 0))
                if fa < min_frac or fb < min_frac:
                    continue
                rows.append({"pair": name, "ligand": lig, "receptor": rec,
                             "source": str(a), "target": str(b),
                             "score": float(np.mean(la) * np.mean(rb)),
                             "frac_source": fa, "frac_target": fb})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
