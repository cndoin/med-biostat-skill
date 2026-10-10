# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/engine/differential.py
# 许可: 自建/内部
# 说明: 差异表达 + 超几何富集
# 源 sha256: 78c7874b09b63d90
# VENDORED-FILE-END
"""差异分析与功能富集。

差异分析提供两条路线：
    t 检验路线  —— 两组，简单、可解释。
    线性模型路线 —— 任意设计（含协变量与批次），用 moderated t（limma 思想）。

moderated t 的作用：把每个特征的残差方差向全体方差的先验收缩，
在重复数少（n=3–6）时显著提升稳定性，这是蛋白组/转录组小样本的标准做法。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma, polygamma

from ..core.stats import adjust_p


def _fit_lm(X: np.ndarray, Y: np.ndarray) -> tuple:
    """对每个特征（Y 的列）拟合同一设计 X，返回系数、残差方差、自由度。"""
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ X.T @ Y                     # (p_features, k)
    resid = Y - X @ B
    df = max(X.shape[0] - np.linalg.matrix_rank(X), 1)
    s2 = (resid ** 2).sum(axis=0) / df        # (p_features,)
    return B, s2, df, XtX_inv


def moderated_t(Y: np.ndarray, X: np.ndarray, coef_index: int = 1,
                method: str = "fdr") -> pd.DataFrame:
    """线性模型 + 经验贝叶斯方差收缩的 moderated t 检验。

    Y: (n_samples, n_features) 定量矩阵
    X: (n_samples, k) 设计矩阵，第 0 列通常为截距
    coef_index: 待检验的系数下标

    返回 log2FC、t、p、q、以及收缩后的方差。
    """
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    B, s2, df, XtX_inv = _fit_lm(X, Y)
    beta = B[coef_index]
    vcoef = XtX_inv[coef_index, coef_index] * s2

    # 经验贝叶斯：用矩估计拟合逆卡方先验（Smyth 2004 的简化矩匹配）
    s2_pos = s2[s2 > 0]
    if s2_pos.size > 3:
        z = np.log(s2_pos)
        e = z - digamma(df / 2) + np.log(df / 2)
        var_e = np.var(e, ddof=1) - polygamma(1, df / 2)
        if var_e > 0:
            d0 = 2 * _trigamma_inverse(var_e)
            s0_sq = float(np.exp(np.mean(e) + digamma(d0 / 2) - np.log(d0 / 2)))
        else:
            d0, s0_sq = np.inf, float(np.mean(s2_pos))
    else:
        d0, s0_sq = np.inf, float(np.mean(s2_pos)) if s2_pos.size else 1.0

    if np.isinf(d0):
        s2_post = np.full_like(s2, s0_sq)
        df_total = 1e6
    else:
        s2_post = (d0 * s0_sq + df * s2) / (d0 + df)
        df_total = d0 + df

    vcoef_post = XtX_inv[coef_index, coef_index] * s2_post
    se = np.sqrt(np.maximum(vcoef_post, 1e-300))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df_total)
    return pd.DataFrame({
        "log2FC": beta / np.log(2),
        "beta": beta,
        "t": t,
        "p": p,
        "q": adjust_p(p, method),
        "s2": s2,
        "s2_post": s2_post,
    })


def _trigamma_inverse(x: float) -> float:
    """解 trigamma(y) = x 的牛顿迭代，用于经验贝叶斯先验自由度估计。"""
    y = 0.5 + 1.0 / max(x, 1e-12)
    for _ in range(50):
        tri = polygamma(1, y)
        d = tri * (1 - tri / x) / polygamma(2, y)
        y = y + d
        if y <= 0:
            y = 1e-6
        if abs(d) < 1e-10:
            break
    return float(y)


def differential(matrix: pd.DataFrame, metadata: pd.DataFrame,
                 group_col: str = "group",
                 covariates: Optional[Sequence[str]] = None,
                 lfc_thresh: float = 1.0,
                 alpha: float = 0.05,
                 log2_input: bool = False) -> pd.DataFrame:
    """端到端差异分析。

    matrix: 行=样本，列=特征（蛋白/基因/位点）
    log2_input=False 时，函数内部做 log2(x + 伪计数) 变换。
    """
    feats = list(matrix.columns)
    common = matrix.index.astype(str).intersection(metadata["sample_id"].astype(str))
    if len(common) < 6:
        raise ValueError("样本表与矩阵匹配样本数 %d < 6，无法做差异分析" % len(common))

    M = matrix.loc[matrix.index.astype(str).isin(common)]
    meta = metadata[metadata["sample_id"].astype(str).isin(common)].copy()
    meta = meta.set_index(meta["sample_id"].astype(str)).loc[M.index.astype(str)]

    Y = M[feats].values.astype(float)
    if not log2_input:
        Y = np.log2(np.clip(Y, 0, None) + 1.0)

    levels = sorted(meta[group_col].astype(str).unique())
    if len(levels) != 2:
        raise ValueError("differential 当前支持两组比较，检测到 %d 组" % len(levels))
    g = (meta[group_col].astype(str).values == levels[1]).astype(float)

    cols = [np.ones(len(g)), g]
    names = ["intercept", "group"]
    for cov in (covariates or []):
        if cov in meta.columns:
            v = meta[cov].values
            if pd.api.types.is_numeric_dtype(meta[cov]):
                vv = np.asarray(v, float)
                vv = np.nan_to_num(vv, nan=np.nanmean(vv))
                cols.append(vv)
            else:
                lv = sorted(set(map(str, v)))[1:]
                for L in lv:
                    cols.append((np.array([str(x) for x in v]) == L).astype(float))
                    names.append("%s=%s" % (cov, L))
                continue
            names.append(cov)
    X = np.column_stack(cols)

    res = moderated_t(Y, X, coef_index=1)
    res.insert(0, "feature", feats)
    res["significant"] = (res["q"] < alpha) & (res["log2FC"].abs() >= lfc_thresh)
    res["direction"] = np.where(res["significant"],
                                np.where(res["log2FC"] > 0, "up", "down"), "ns")
    return res.sort_values("p").reset_index(drop=True)


# ------------------------------------------------------------------ 富集

def enrichment(features: Sequence[str], gene_sets: Dict[str, Sequence[str]],
               universe: Sequence[str], alpha: float = 0.05) -> pd.DataFrame:
    """超几何检验（ORA）。

    universe 必须是"检测到的全部特征"，而不是全基因组——用错背景是富集分析
    最常见的错误，会系统性夸大通路显著性。
    """
    universe = set(map(str, universe))
    hits = set(map(str, features)) & universe
    N = len(universe)
    K = len(hits)
    rows = []
    for name, members in gene_sets.items():
        mset = set(map(str, members)) & universe
        n = len(mset)
        k = len(hits & mset)
        if n == 0 or K == 0:
            rows.append({"term": name, "k": k, "n": n, "p": np.nan, "fold": np.nan})
            continue
        p = float(stats.hypergeom.sf(k - 1, N, n, K))
        fold = (k / K) / (n / N) if n and K else np.nan
        rows.append({"term": name, "k": k, "n": n,
                     "expected": round(K * n / N, 2), "fold": fold, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["q"] = adjust_p(out["p"].values, method="bh")
        out["significant"] = out["q"] < alpha
        out = out.sort_values("p").reset_index(drop=True)
    return out
