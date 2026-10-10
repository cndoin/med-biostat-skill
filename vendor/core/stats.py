# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/core/stats.py
# 许可: 自建/内部
# 说明: 统计基础：多重检验/效应量/缺失/重抽样/检验
# 源 sha256: 95cddc52e7d40dd2
# VENDORED-FILE-END
"""统计基础：多重检验、效应量、缺失值、重抽样。

所有函数均为纯函数，便于单测。默认不做任何"替用户做决定"的隐式处理——
缺失值的处理方式必须由调用方显式指定。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- 多重检验

def adjust_p(p: np.ndarray, method: str = "bh") -> np.ndarray:
    """p 值校正，返回 q 值。支持 bh（Benjamini-Hochberg）、by、bonferroni。

    注意：BH 控制 FDR，BY 更保守（在正相关检验下适用），Bonferroni 控制 FWER。
    输入含 NaN 时原样返回 NaN。
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    if method == "bonferroni":
        adj = np.minimum(ranked * m, 1.0)
    elif method == "by":
        denom = np.arange(1, m + 1) / (1.0 + np.sum(1.0 / np.arange(2, m + 2)))
        adj = np.minimum(ranked / denom, 1.0)
    else:  # bh
        adj = ranked * m / np.arange(1, m + 1)
    # 单调化：保证 q 随秩次不减
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


# ---------------------------------------------------------------- 效应量

def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d，合并标准差版本。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


def hedges_g(x: Sequence[float], y: Sequence[float]) -> float:
    """Hedges' g：对小样本偏差做校正的 d。"""
    d = cohens_d(x, y)
    n = len(x) + len(y)
    if np.isnan(d) or n <= 3:
        return d
    return float(d * (1 - 3.0 / (4 * n - 9)))


def odds_ratio(a: int, b: int, c: int, d: int) -> Tuple[float, float, float]:
    """2x2 表的比值比与 Woolf 对数置信区间。零单元格加 0.5 连续性校正。"""
    a, b, c, d = map(float, (a, b, c, d))
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = float(np.exp(np.log(or_) - 1.96 * se))
    hi = float(np.exp(np.log(or_) + 1.96 * se))
    return float(or_), lo, hi


def interpret_d(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "可忽略"
    if a < 0.5:
        return "小"
    if a < 0.8:
        return "中"
    return "大"


# ---------------------------------------------------------------- 缺失值

def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """逐列缺失报告，按缺失率降序。"""
    n = len(df)
    rows = []
    for c in df.columns:
        k = int(df[c].isna().sum())
        rows.append({"column": c, "n_missing": k,
                     "pct_missing": round(100.0 * k / n, 2) if n else np.nan})
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def impute_matrix(X: np.ndarray, strategy: str = "median",
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """简单插补。strategy: median / mean / zero / knn。

    注意：本函数只用于特征矩阵；临床协变量若缺失率高于 5%，
    应优先使用多重插补而非单点插补，并在论文中说明。
    """
    X = np.array(X, dtype=float, copy=True)
    if strategy in ("median", "mean"):
        stat = np.nanmedian(X, axis=0) if strategy == "median" else np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(stat, inds[1])
    elif strategy == "zero":
        X = np.nan_to_num(X, nan=0.0)
    elif strategy == "knn":
        X = _knn_impute(X, k=5)
    else:
        raise ValueError("未知插补策略：%s" % strategy)
    return X


def _knn_impute(X: np.ndarray, k: int = 5) -> np.ndarray:
    """按列相关最近的 k 个样本均值插补（简化版，不含迭代）。"""
    n, p = X.shape
    med = np.nanmedian(X, axis=0)
    X0 = np.where(np.isnan(X), med, X)
    for j in range(p):
        miss = np.where(np.isnan(X[:, j]))[0]
        if miss.size == 0:
            continue
        for i in miss:
            obs = ~np.isnan(X[:, j])
            if obs.sum() == 0:
                X[i, j] = med[j]
                continue
            d = np.sqrt(np.nansum((X0[obs] - X0[i]) ** 2, axis=1))
            idx = np.argsort(d)[:k]
            X[i, j] = np.nanmean(X[np.where(obs)[0][idx], j])
    return X


# ---------------------------------------------------------------- 重抽样

def bootstrap_ci(values: Sequence[float], func=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """对任意统计量做百分位 bootstrap 置信区间，返回 (点估计, lo, hi)。"""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(func(v))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(v, size=v.size, replace=True)
        draws[b] = func(s)
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return point, lo, hi


# ---------------------------------------------------------------- 假设检验

def welch_t(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Welch t 检验，附效应量。不假设等方差——这是两组比较的默认选择。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        return {"t": np.nan, "p": np.nan, "d": np.nan, "g": np.nan,
                "n1": x.size, "n2": y.size}
    t, p = stats.ttest_ind(x, y, equal_var=False)
    return {"t": float(t), "p": float(p), "d": cohens_d(x, y),
            "g": hedges_g(x, y), "n1": int(x.size), "n2": int(y.size)}


def normality_report(x: Sequence[float]) -> Dict[str, float]:
    """Shapiro-Wilk 正态性检验 + 偏度。n>5000 时 Shapiro 不可用，改用 D'Agostino。"""
    v = np.asarray(x, float)
    v = v[~np.isnan(v)]
    if v.size < 3:
        return {"n": v.size, "p": np.nan, "skew": np.nan, "method": "none"}
    if v.size <= 5000:
        stat, p = stats.shapiro(v)
        method = "shapiro"
    else:
        stat, p = stats.normaltest(v)
        method = "dagostino"
    return {"n": int(v.size), "stat": float(stat), "p": float(p),
            "skew": float(stats.skew(v)), "method": method}
