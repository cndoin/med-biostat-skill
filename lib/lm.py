"""lm.py —— 经验贝叶斯 moderated t 线性模型引擎（**全项目唯一裁决点**）。

为什么单独抽一个模块
--------------------
同样一套 "OLS + Smyth 2004 `fitFDist` 方差收缩 → moderated t" 的引擎，
甲基化 EWAS（`lib.epigen`）与蛋白组差异（`lib.proteomics`）都要用。
若各写一份，两处的先验参数 d0 / s0²、自由度、p 值口径必然漂移，
同一个基因在两个组学里会得到**不可比的 p 值** —— 而"RNA-蛋白是否同向"
恰恰要靠两侧 p 值可比才能判。故抽成唯一裁决点（设计红线 P21）。

**本模块只做检验，不做组学语义**。beta→M 的转换、蛋白强度的 log2 转换
都是调用方的事，本文件一律认为 Y 已经是"适合建线性模型的尺度"。

口径（随数字一起输出，不可省略）
------------------------------
- ``d0`` / ``s0_sq``：Smyth 2004 的先验自由度与先验方差；``d0=inf`` 表示
  各特征方差完全一致（收缩最强），``d0=0`` 退化为普通 t 检验。
- ``dof``：检验自由度 = 残差自由度 + d0（moderated 时可能是小数）。
- ``method``：'limma-style moderated t (Smyth 2004)' 或 'ordinary t-test'。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma
from scipy import stats

__all__ = ["fit_f_dist", "moderated_lm", "bh_adjust"]


def _solve_trigamma(target: float, lo: float = 1e-8, hi: float = 1e8,
                    max_iter: int = 200) -> float:
    """解 ``trigamma(x) = target``（trigamma 单调递减，几何二分）。"""
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        if polygamma(1, mid) > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def fit_f_dist(s2: np.ndarray, df: float) -> Tuple[float, float]:
    """Smyth 2004 的 `fitFDist`：由残差方差的样本矩估计先验 (d0, s0²)。

    思路：``log(s2)`` 的期望与方差可写成 d0 与 s0² 的函数，反解即得先验。
    d0 → ∞ 表示「各方差一样」，收缩最强；d0 小则接近普通 t 检验。
    """
    s2 = np.asarray(s2, dtype=float)
    s2 = s2[np.isfinite(s2) & (s2 > 0)]
    if s2.size < 3:
        return 0.0, (float(np.mean(s2)) if s2.size else 1.0)
    z = np.log(s2)
    e = z - digamma(df / 2.0) + math.log(df / 2.0)
    emean, evar = float(e.mean()), float(e.var(ddof=1))
    target = evar - polygamma(1, df / 2.0)
    if target <= 1e-12:
        return float("inf"), math.exp(emean)      # 先验方差无限 → 完全收缩
    d0 = 2.0 / _solve_trigamma(target)
    return float(d0), float(math.exp(emean))


def moderated_lm(Y, X, coef_index: int = 1, moderated: bool = True,
                 feature_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """对 (样本 × 特征) 矩阵 Y 建线性模型，返回指定系数的 moderated t 检验。

    :param Y: (n_samples, n_features) 已转换到合适尺度的响应矩阵
    :param X: (n_samples, n_params) 设计矩阵（**含截距**）
    :param coef_index: 要检验的设计矩阵列下标（默认 1 = 第一个协变量）
    :param moderated: 经验贝叶斯方差收缩（默认 True；小样本强烈建议开）
    :return: dict
        - ``coef`` / ``se`` / ``t`` / ``pvalue``: 长度 n_features 的数组
        - ``d0`` / ``s0_sq`` / ``dof``: 先验与自由度（口径，必须一起报告）
        - ``s2``: 每特征的残差方差；``resid_dof``: 残差自由度
        - ``method`` / ``engine``
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y 必须是二维矩阵 (样本 × 特征)")
    if X.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"X 形状 {X.shape} 与 Y 行数 {Y.shape[0]} 不匹配")
    n, p = X.shape
    resid_dof = n - p
    if resid_dof <= 0:
        raise ValueError(f"样本数 {n} 不足以拟合 {p} 个参数（残差自由度 {resid_dof}）")
    if not (0 <= coef_index < p):
        raise ValueError(f"coef_index={coef_index} 越界（设计矩阵只有 {p} 列）")

    XtX_inv = np.linalg.pinv(X.T @ X)
    cj = float(XtX_inv[coef_index, coef_index])       # 该系数的杠杆
    beta_hat = XtX_inv @ X.T @ Y                      # (p, n_features)
    resid = Y - X @ beta_hat
    s2 = (resid ** 2).sum(axis=0) / resid_dof
    coef = beta_hat[coef_index]

    if moderated:
        d0, s0_sq = fit_f_dist(s2, resid_dof)
        if np.isinf(d0):
            s2_post = np.full_like(s2, s0_sq)
            dof_t = float("inf")
        else:
            s2_post = (d0 * s0_sq + resid_dof * s2) / (d0 + resid_dof)
            dof_t = float(resid_dof + d0)
        se = np.sqrt(s2_post * cj)
    else:
        d0, s0_sq, dof_t = 0.0, float("nan"), float(resid_dof)
        se = np.sqrt(s2 * cj)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, coef / se, np.nan)
    if np.isinf(dof_t):
        pvalue = 2 * stats.norm.sf(np.abs(t))
    else:
        pvalue = 2 * stats.t.sf(np.abs(t), dof_t)

    names = (list(feature_names) if feature_names is not None
             else [f"f{i}" for i in range(Y.shape[1])])
    return {
        "coef": coef,
        "se": se,
        "t": t,
        "pvalue": pvalue,
        "s2": s2,
        "d0": float(d0),
        "s0_sq": float(s0_sq) if np.isfinite(s0_sq) else float("inf"),
        "dof": dof_t,
        "resid_dof": int(resid_dof),
        "coef_index": int(coef_index),
        "feature_names": names,
        "moderated": bool(moderated),
        "method": ("limma-style moderated t (Smyth 2004)" if moderated
                   else "ordinary t-test"),
        "engine": "lib.lm.moderated_lm",
    }


def bh_adjust(pvalue: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正，返回 q 值（单调化的 BH）。

    放在这里而不是各处重写：`lib.epigen`、`lib.proteomics` 的 padj 必须
    是**同一套 BH**，否则两侧 q 值阈值不可比，"三层正交同向"就判不了。
    """
    p = np.asarray(pvalue, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    ok = np.isfinite(p)
    out = np.full(m, np.nan)
    pv = p[ok]
    k = pv.size
    order = np.argsort(pv)
    ranked = np.clip(pv[order] * k / (np.arange(k) + 1), 0, 1)
    q = np.empty(k)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    out[ok] = q
    return out
