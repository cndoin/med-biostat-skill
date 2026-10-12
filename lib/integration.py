# -*- coding: utf-8 -*-
"""批次整合 + **整合保真诊断**。

这个模块存在的理由只有一句：**Harmony 只保证消批次，不保证留住任何生物学信号**。

已知的失效方式（都不是报错，而是静默给错结果）：
  1. 整合过度 —— θ 惩罚把"某细胞类型恰好富集于某个供体"当成批次效应抹掉，
     于是年轻/年老供体之间的真实差异被一并抹平，下游年龄趋势全部变平，
     Fig3 的失守点再也找不到，而你会误以为"本来就没有年龄效应"。
  2. 整合不足 —— 供体特异的偏移没去掉，下游差异表达首当其冲的全是供体差异，
     DE 表里排前十的全是个体差异，看上去"Noah县的亮点"其实是技术噪声。

所以**先整合，再实测 fidelity，然后才做下游分析**，中间那一步不能省。
本模块的 `fidelity` 就是把这一步变成可以写进论文方法学的数字。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vendor._harmony import core as _harmony

__all__ = ["harmonize", "batch_mixing", "covariate_variance", "fidelity"]


# ── 整合 ─────────────────────────────────────────────────────────────────────
def harmonize(X, batch: Sequence[Any], *, max_iter_harmony: int = 10,
              theta: float = 2.0, sigma: float = 0.1, ridge_lambda: float = 1.0,
              n_clusters: Optional[int] = None, random_state: int = 0,
              stop_on_increase: bool = False, **kw) -> Dict[str, Any]:
    """Harmony 整合的包装层。默认**关掉**上游那个会提前收工的早停（见下）。

    :param X: (细胞 × 维度) embedding，通常是 scale 过的 PCA
    :param batch: 批次标签，**通常就是供体 ID**；不要把年龄/分组塞进来
    :param theta: 多样性惩罚权重；**实测（2026-10-12）**：批次偏移接近生物群间距时
        上游默认 θ=2.0 会留下残留批次（三批次大偏移下 batch_mixing 的 normalized
        从 2.52 只降到 1.49），θ 提到 3 后降到 1.18，θ=4~6 不再改善（饱和）。
        所以**别指望默认值够用**，调完要用 ``batch_mixing`` 实测
    :param stop_on_increase: 目标上升时是否停止（**默认 False，与上游默认不同**）

    **为什么默认与上游不同（必须知道）**：
    上游 goal function 各轮算在不同的 embedding 上，跨轮不可比；批次偏移大时
    目标第二轮就上升，判据立刻认为"已收敛"→ 只跑 1 轮就收工。实测：2 批次
    大偏移下默认参数 mix 只降到 0.777（残留批次），跑满 10 轮降到 0.568，
    且生物团纯度反而从 0.947 升到 0.977。多跑几轮的代价远小于"没整干净"。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X 必须是二维 (细胞 × 维度)，收到 {X.shape}")
    b = np.asarray(list(batch))
    if b.shape[0] != X.shape[0]:
        raise ValueError(f"batch 长度 {b.shape[0]} 与细胞数 {X.shape[0]} 不一致")
    if max_iter_harmony < 1:
        raise ValueError(f"max_iter_harmony 必须 >= 1，收到 {max_iter_harmony}")

    Z, diag = _harmony.harmonize(
        X, batch, max_iter_harmony=int(max_iter_harmony), theta=float(theta),
        sigma=float(sigma), ridge_lambda=float(ridge_lambda),
        n_clusters=n_clusters, random_state=int(random_state),
        stop_on_increase=bool(stop_on_increase), return_diagnostics=True, **kw)

    return {
        "embedding": Z,
        "method": "harmony",
        "params": {
            "max_iter_harmony": int(max_iter_harmony),
            "iterations_run": int(diag["n_harmony_iter_run"]),
            "stopped_by": diag["stopped_by"],
            "theta": float(theta), "sigma": float(sigma),
            "ridge_lambda": float(ridge_lambda),
            "n_clusters": int(diag["n_clusters"]),
            "stop_on_increase": bool(stop_on_increase),
            "random_state": int(random_state),
        },
        "n_cells": int(X.shape[0]),
        "n_dims": int(X.shape[1]),
        "n_batches": int(diag["n_batches"]),
        "mean_shift": float(np.abs(Z - X).mean()),
        "max_shift": float(np.abs(Z - X).max()),
        "objectives": diag["objectives_harmony"],
        "soft_assignment": diag["soft_assignment"],
        "engine": "lib.integration.harmonize -> vendor._harmony.core",
        "source": "lilab-bcb/harmony-pytorch（BSD-3-Clause，去 torch/sklearn）",
        "caveat": ("整合是否成功**不能看这个函数的输出**，必须用 fidelity 实测；"
                   "Harmony 不保证保留任何协变量信号"),
    }


# ── 批次混合度 ───────────────────────────────────────────────────────────────
def batch_mixing(Z, batch: Sequence[Any], n_neighbors: int = 15) -> Dict[str, Any]:
    """kNN 邻居里"和自己是同一批次"的比例 —— **越低越好**。

    完美混合的期望值 = ``Σ_b p_b²``（随机取邻居落在同批次的概率）。所以报两个量：
    ``raw``（原始比例）与 ``normalized``（raw / 期望值，1.0 = 与随机无异，>1 = 仍分批聚集）。
    只报 raw 会误导：批次不平衡时 "同批次邻居占比" 天然偏高，那是构成不是整合质量。
    """
    Z = np.asarray(Z, dtype=float)
    b = np.asarray(list(batch))
    n = Z.shape[0]
    if b.shape[0] != n:
        raise ValueError("batch 长度与行数不一致")
    k = int(min(n_neighbors, n - 1))
    if k < 1:
        raise ValueError("细胞太少，无法构造近邻")

    G = Z @ Z.T
    sq = (Z ** 2).sum(axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0)
    np.fill_diagonal(D2, np.inf)
    idx = np.argsort(D2, axis=1)[:, :k]

    same = float((b[idx] == b[:, None]).mean())
    _, counts = np.unique(b, return_counts=True)
    expected = float(((counts / n) ** 2).sum())
    return {
        "raw": same,
        "expected_random": expected,
        "normalized": (same / expected) if expected > 0 else float("nan"),
        "n_neighbors": k,
        "n_cells": int(n),
        "batch_balance": {str(v): int(c) for v, c in zip(np.unique(b), counts)},
        "rule": "normalized ≈ 1 表示混合到随机水平；越小越好",
    }


# ── 协变量可解释方差 ─────────────────────────────────────────────────────────
def covariate_variance(Z, y: Sequence[float], kind: str = "continuous",
                       n_pcs: Optional[int] = None) -> Dict[str, Any]:
    """用 embedding 的前若干维做 OLS 解释某协变量，返回 R² 与 F 检验 p 值。

    这里的 R² 衡量的是**几何上的可解释性**：如果能用 embedding 的前几个主方向
    线性预测出年龄，就说明整合后的图里仍然携带着年龄信息。
    它**不是**因果，也不是下游任务精度 —— 只是一个便宜且直观的保真代理指标。
    """
    from scipy import stats

    Z = np.asarray(Z, dtype=float)
    y = np.asarray(list(y), dtype=float)
    if Z.shape[0] != y.shape[0]:
        raise ValueError("Z 的行数与协变量长度不一致")
    if Z.ndim != 2:
        raise ValueError("Z 必须是二维")
    n, d = Z.shape
    if n_pcs is None:
        n_pcs = int(min(d, 20))
    p = int(min(n_pcs, d))
    n_valid = int(np.isfinite(y).sum())
    if n_valid < p + 3:
        raise ValueError(f"有效样本 {n_valid} 不足以拟合 {p} 个主成分 + 截距")

    m = np.isfinite(y)
    A = np.hstack([np.ones((n, 1)), Z[:, :p]])[m]
    yy = y[m]
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ beta
    tss = float(((yy - yy.mean()) ** 2).sum())
    rss = float((resid ** 2).sum())
    r2 = float(1.0 - rss / tss) if tss > 0 else float("nan")
    df1, df2 = p, len(yy) - p - 1
    if df2 <= 0 or not np.isfinite(r2):
        f_stat, pval = float("nan"), float("nan")
    elif r2 >= 1.0 - 1e-12:
        # **完全可解释**必须判为极端显著，不能因为 F=∞ 就把 p 丢成 NaN。
        # 曾经的 bug：这里落进 NaN 后，下游 fidelity 用 "p < α" 判 informative，
        # 于是 R²=1（信号最强！）被判成"整合前就没有信息"，retention 直接失去定义
        # —— 信号越强越被当成没信号，方向完全反了。
        f_stat, pval = float("inf"), 0.0
    else:
        r2c = float(np.clip(r2, 0.0, 1.0 - 1e-15))
        f_stat = float((r2c / df1) / ((1.0 - r2c) / df2))
        pval = float(stats.f.sf(f_stat, df1, df2))
    return {
        "r2": r2, "f": f_stat, "pvalue": pval,
        "n_pcs": p, "n": int(len(yy)), "kind": kind,
        "model": "ols(y ~ 1 + PCs)",
        "engine": "lib.integration.covariate_variance",
    }


def fidelity(before, after, covariates: Mapping[str, Sequence[float]], *,
             batch: Optional[Sequence[Any]] = None,
             n_pcs: Optional[int] = None,
             min_before_r2: float = 0.01, alpha: float = 0.05) -> Dict[str, Any]:
    """**整合保真诊断**：逐个协变量比较整合前后"能被 embedding 解释的方差"。

    :param before: 整合前 embedding
    :param after: 整合后 embedding（必须与 before 同行、同批次顺序）
    :param covariates: ``{协变量名: 每个细胞的取值}``，如 ``{"age": ages}``
    :param batch: 可选，给出批次时一并报批次混合度
    :param min_before_r2: 整合前 R² 低于此值就标记 ``uninformative``
        —— 分母本身就是 0 时，"保留率"无论多少都没有意义
    :returns: per-covariate 表 + 汇总 verdict

    **判据（写死在这里，免得口头漂移）**
      - ``retention = r2_after / r2_before``，只在前 R² 达标且前后都可估时才给数；
      - 若某个协变量整合前 R² 就不达标或 F 检验不显著 → ``uninformative``，
        **不能**用来宣称"整合保住了它"（原本就没有信息可保）；
      - verdict 取所有 informative 协变量里**最差**的一个，这是保守口径：
        只要有一个关键协变量被抹掉，整张图就不可用于该协变量的下游分析。
    """
    A = np.asarray(before, dtype=float)
    B = np.asarray(after, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"before {A.shape} 与 after {B.shape} 形状必须一致")
    if not covariates:
        raise ValueError("covariates 为空 —— 没有协变量就没有保真可谈")

    rows = []
    for name, y in covariates.items():
        yv = np.asarray(list(y), dtype=float)
        cb = covariate_variance(A, yv, n_pcs=n_pcs)
        ca = covariate_variance(B, yv, n_pcs=n_pcs)
        before_ok = bool(np.isfinite(cb["r2"]) and cb["r2"] >= float(min_before_r2)
                         and np.isfinite(cb["pvalue"]) and cb["pvalue"] < float(alpha))
        if before_ok and np.isfinite(ca["r2"]) and cb["r2"] > 0:
            ret = float(ca["r2"] / cb["r2"])
        else:
            ret = None
        rows.append({
            "covariate": name,
            "r2_before": cb["r2"], "p_before": cb["pvalue"],
            "r2_after": ca["r2"], "p_after": ca["pvalue"],
            "retention": ret,
            "informative": bool(before_ok),
            "abs_change": (float(ca["r2"] - cb["r2"])
                           if np.isfinite(ca["r2"]) and np.isfinite(cb["r2"]) else None),
        })
    table = pd.DataFrame(rows)

    info = table[table["informative"] & table["retention"].notna()]
    if len(info):
        worst = info.loc[info["retention"].idxmin()]
        min_ret = float(worst["retention"])
        verdict = ("PASS" if min_ret >= 0.7 else
                   "WARN" if min_ret >= 0.4 else "FAIL")
        worst_name = str(worst["covariate"])
    else:
        min_ret, verdict, worst_name = None, "UNINFORMATIVE", None

    out: Dict[str, Any] = {
        "table": table,
        "min_retention": min_ret,
        "worst_covariate": worst_name,
        "verdict": verdict,
        "verdict_rule": ("PASS ≥ 0.7 / WARN ≥ 0.4 / FAIL < 0.4，取所有 informative "
                         "协变量中最差的那个；UNINFORMATIVE = 没有任何协变量在整合前"
                         "就显著可被解释（此时保留率无定义，不能宣称保住了信号）"),
        "n_pcs": int(min(n_pcs or 20, A.shape[1])),
        "engine": "lib.integration.fidelity",
        "caveat": ("R² 是**几何可解释性**的代理，不是下游任务精度，也不是因果；"
                   "它只能告诉你'信号还在图里'，不能告诉你'下游分析一定对'"),
    }
    if batch is not None:
        out["batch_mixing_before"] = batch_mixing(A, batch)
        out["batch_mixing_after"] = batch_mixing(B, batch)
    return out
