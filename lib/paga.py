# -*- coding: utf-8 -*-
"""PAGA 抽象图（簇间连通性）—— 按 Wolf et al. 2019 论文的口径独立实现。

PAGA 要解决的问题很朴素：把上万个细胞压成 k 个簇之后，**哪两个簇在原始数据里
真的是挨着的？** 只有回答了这个问题才敢把簇抽象成一个"进化/分化路径图"。

判据不是"两边各有几个细胞"，而是**观测到的 kNN 连边数 vs 随机情形下的期望连边数**：

    ratio[a,b] = 观测连边 / 期望连边 ；  score = log(ratio)

``score > 0`` = 比随机更连通（图栅上是亲邻）；``= 0`` = 与随机无异；``< 0`` = 相互回避。
不取这个对数比率会误导：大簇与任意簇都有很多连边，那只是因为它大。

为什么不直接搬 scanpy 的 ``tl.paga``：本仓库 vendored 的 scanpy 片段里没有
``tl.paga`` 这条线（当初只取了 preprocessing/tools 的 qc/neighbors/louvain 等），
缺少拥挤不少于、且这里只需要抽象图这一件事，没必要引入整条 tl 链路。

**已知差异（如实标注）**：scanpy 用 umap 的模糊单纯形集构造 connectivities，
本实现用 ``vendor/_sc/core.py::neighbors`` 的自适应高斯核（本仓库唯一裁决点）。
因此 ratio 的**绝对值不可与 scanpy 逐位对比**；只能比对拓扑结论
（哪几对簇连通、MST 是不是同一棵）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vendor._sc import core as _sc

__all__ = ["paga_connectivity", "paga_tree"]


def _cluster_sizes(labels: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
    cats: List[Any] = []
    seen: Dict[Any, int] = {}
    codes = np.empty(labels.shape[0], dtype=int)
    for i, v in enumerate(labels):
        key = v.item() if hasattr(v, "item") else v
        if key not in seen:
            seen[key] = len(cats)
            cats.append(key)
        codes[i] = seen[key]
    sizes = np.bincount(codes, minlength=len(cats)).astype(float)
    return codes, cats


def paga_connectivity(embedding, labels: Sequence[Any], *,
                      n_neighbors: int = 15,
                      threshold: Optional[float] = None,
                      min_ratio: float = 1e-3) -> Dict[str, Any]:
    """簇间连通性（PAGA abstracted graph 的输入）。

    :param embedding: (细胞 × 维度) 降维矩阵，通常是 PCA（scale 过的表达）
    :param labels: 每个细胞的簇标签
    :param n_neighbors: kNN 图的 k
    :param threshold: 给定时，``|score| < threshold`` 的位置置 0（画抽象图时去噪）
    :param min_ratio: 观测连边为 0 时的下限保护，避免 log(0) 得到 -inf
    :returns: ``{"connectivity": DataFrame, "observed": ..., "expected": ...,
                 "edges": DataFrame, "cluster_sizes": Series, ...}``

    三条口径：
      1. 期望连边按**有向 kNN 图**推：每个节点出 k 条边且不含自身，
         故 ``expected[a,b] = k · n_a · (n_b - δ_ab) / (N - 1)``，
         对 b 求和得 ``k·n_a``，与每个节点出 k 条边的事实自洽；
      2. 观测值取 ``(D + Dᵀ)/2`` 对称化 —— kNN 图本身不对称（k 近邻关系不是对称关系），
         不处理会得到带方向的"伪亲疏"；
      3. 对角（簇内）也照算并保留：簇内紧凑度是判断"这个簇是不是被过度细分"的依据，
         把它扔掉会让分裂过度的人造簇看起来和真簇一样。
    """
    X = np.asarray(embedding, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"embedding 必须是二维，收到形状 {X.shape}")
    lab = np.asarray(list(labels))
    if lab.shape[0] != X.shape[0]:
        raise ValueError(f"labels 长度 {lab.shape[0]} 与细胞数 {X.shape[0]} 不一致")

    codes, cats = _cluster_sizes(lab)
    K = len(cats)
    if K < 2:
        raise ValueError(f"至少要有 2 个簇才能谈簇间连通性，当前 {K} 个")
    n = X.shape[0]
    if n_neighbors < 1 or n_neighbors >= n:
        raise ValueError(f"n_neighbors 必须在 [1, {n - 1}] 内，收到 {n_neighbors}")

    nb = _sc.neighbors(X, n_neighbors=int(n_neighbors))
    idx = np.asarray(nb["knn_index"])
    k = int(nb["n_neighbors"])

    # ── 观测连边计数（有向），再对称化
    src = np.repeat(np.arange(n), k)
    dst = idx.ravel()
    D = np.zeros((K, K), dtype=float)
    np.add.at(D, (codes[src], codes[dst]), 1.0)
    observed = (D + D.T) / 2.0

    # ── 随机期望（同图的置换期望）
    # 每个节点出 k 条边且不含自身 → 对 b 求和得 k·n_a，与实际边数自洽
    sizes = np.bincount(codes, minlength=K).astype(float)
    expected = np.array([[k * sizes[a] * (sizes[b] - (1.0 if a == b else 0.0)) / (n - 1.0)
                          for b in range(K)] for a in range(K)])

    lo = float(min_ratio)
    ratio = (observed + lo) / (expected + lo)
    score = np.log(ratio)

    if threshold is not None:
        score = np.where(np.abs(score) >= float(threshold), score, 0.0)

    conn = pd.DataFrame(score, index=[str(c) for c in cats],
                        columns=[str(c) for c in cats])
    # 长表：画网络图 / 导出 Cytoscape 用（去重上三角，含对角）
    rows = []
    for a in range(K):
        for b in range(a, K):
            rows.append({
                "source": str(cats[a]), "target": str(cats[b]),
                "observed": float(observed[a, b]),
                "expected": float(expected[a, b]),
                "ratio": float(ratio[a, b]),
                "score": float(score[a, b]),
                "within_cluster": bool(a == b),
            })
    edges = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    return {
        "connectivity": conn,
        "ratio": pd.DataFrame(ratio, index=conn.index, columns=conn.columns),
        "observed": pd.DataFrame(observed, index=conn.index, columns=conn.columns),
        "expected": pd.DataFrame(expected, index=conn.index, columns=conn.columns),
        "edges": edges,
        "cluster_sizes": pd.Series(sizes, index=conn.index, dtype=float),
        "n_clusters": K,
        "n_cells": int(n),
        "n_neighbors": k,
        "threshold_applied": threshold,
        "knn_source": "vendor._sc.core.neighbors（自适应高斯核）",
        "engine": "lib.paga.paga_connectivity",
        "caveat": ("score = log(观测连边/期望连边)；>0 比随机更连通，=0 与随机无异。"
                   "**绝对值不可跨数据集/跨参数比**（它随 k 与图构造方式变化），"
                   "只能在同一张图内按 score 排序或用 threshold 判「哪些簇是真邻居」。"
                   "connectivities 用的是自适应高斯核而不是 umap 模糊单纯形集，"
                   "**不可与 scanpy tl.paga 逐位对比**，只能比对拓扑"),
    }


def paga_tree(result: Dict[str, Any], *, root: Optional[str] = None) -> Dict[str, Any]:
    """在抽象图上建**最小生成树**（谱系图骨架）—— 用距离 = ``-score``。

    :param result: ``paga_connectivity`` 的返回值
    :param root: 可选，指定根簇时不建 MST，改为算每个簇到根的路径距离
    :returns: ``{"edges": DataFrame, "adjacency": DataFrame, ...}``

    为什么直接当 MST 用有风险（必须写在明处）：
      PAGA 的 MST 是**连通性图上**的 MST，不是发育树上 —— 它只说"这两簇在数据里挨着"，
      不说"谁分化成谁"。要赋予方向必须先有独立证据（拟时序 RNA velocity / 时间标签）。
    """
    conn = result["connectivity"]
    cats = list(conn.index)
    K = len(cats)
    dist = np.maximum(-conn.to_numpy(dtype=float), 0.0)
    np.fill_diagonal(dist, 0.0)
    for a in range(K):                       # 对称化（score 本身对称，取保险）
        for b in range(a + 1, K):
            m = max(dist[a, b], dist[b, a])
            dist[a, b] = dist[b, a] = m

    # Prim（确定性：不依赖任何随机初始化，边权相等时取下标最小的一端）
    in_tree = [0]
    edges: List[Dict[str, Any]] = []
    while len(in_tree) < K:
        best = None
        for a in in_tree:
            for b in range(K):
                if b in in_tree:
                    continue
                if best is None or dist[a, b] < best[0] - 1e-15:
                    best = (dist[a, b], a, b)
        if best is None:
            raise ValueError("抽象图不连通，无法建生成树（考虑降低 threshold 或增大 k）")
        _, a, b = best
        in_tree.append(b)
        edges.append({"source": str(cats[a]), "target": str(cats[b]),
                      "distance": float(dist[a, b]),
                      "score": float(-dist[a, b])})
    adj = pd.DataFrame(0.0, index=cats, columns=cats)
    for e in edges:
        adj.loc[e["source"], e["target"]] = 1.0
        adj.loc[e["target"], e["source"]] = 1.0

    out: Dict[str, Any] = {
        "edges": pd.DataFrame(edges),
        "adjacency": adj,
        "n_edges": len(edges),
        "engine": "lib.paga.paga_tree",
        "caveat": ("这是**连通性**上的最小生成树，不是谱系树 —— 它只说「这两簇在数据里"
                   "挨着」，不说「谁分化成谁」。要给它方向，必须有独立证据"
                   "（拟时序 / RNA velocity / 采样时间点）；本函数不会替你猜方向"),
    }

    if root is not None:
        r = str(root)
        if r not in cats:
            raise ValueError(f"root={root!r} 不在簇标签里（可选：{cats}）")
        # 在 MST 上从根做 BFS，得到每个簇的层级 —— 只给"离根多远"，
        # 不给"谁先谁后"：后者需要时间/伪时间方向的外部证据
        nbrs: Dict[str, List[str]] = {c: [] for c in cats}
        for e in edges:
            nbrs[e["source"]].append(e["target"])
            nbrs[e["target"]].append(e["source"])
        depth: Dict[str, int] = {r: 0}
        queue = [r]
        while queue:
            cur = queue.pop(0)
            for nx in nbrs[cur]:
                if nx not in depth:
                    depth[nx] = depth[cur] + 1
                    queue.append(nx)
        missing = [c for c in cats if c not in depth]
        if missing:
            raise ValueError(f"以下簇未在生成树中连通到 root：{missing}")
        out["root"] = r
        out["depth"] = pd.Series(depth, name="depth_from_root").sort_values()
        out["root_rule"] = ("depth 是生成树上的跳数（ undirected hops），不是时间顺序；"
                            "把它当发育顺序读会得到完全错误的因果叙事")
    return out
