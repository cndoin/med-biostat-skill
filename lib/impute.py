# SKILL-OWNED —— 本 skill 自有实现（可直接编辑）
"""impute.py —— 多重插补（MICE）+ Rubin 合并

LPDC 第 9.3 节明确要求：**缺失数据用多重插补（m = 20），并在敏感性分析中用完整病例复算**。
现有 `vendor/core/stats.impute_matrix` 只做单点插补（median/mean/knn），会低估方差 —— 这正是
「必须做多重插补」的理由：单点插补把缺失值当成已知，人为缩小了标准误。

本模块：
  * `mice()`        —— 链式方程多重插补（条件均值 + 残差扰动，迭代 max_iter 轮）
  * `pool_rubin()`  —— 按 Rubin's rules 合并 m 个估计（点估计、组内/组间方差、总方差、自由度）
  * `complete_cases()` —— 完整病例（敏感性分析的对照）

实现取向：**不引入 statsmodels/sklearn**，只用 numpy 的最小二乘，保证可审计、可复现。

依赖：numpy + pandas（Tier 1）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _regress_predict(X: np.ndarray, y: np.ndarray, y_missing: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """用观测部分拟合最小二乘，对有缺失处给「条件均值 + 残差扰动」的抽样值。"""
    obs = ~y_missing
    if obs.sum() < 2 or X.shape[1] == 0:
        return rng.normal(np.nanmean(y), max(np.nanstd(y), 1e-8), y_missing.sum())
    Xo, yo = X[obs], y[obs]
    Xo1 = np.column_stack([np.ones(len(Xo)), Xo])
    try:
        coef, *_ = np.linalg.lstsq(Xo1, yo, rcond=None)
    except np.linalg.LinAlgError:
        return rng.normal(np.nanmean(yo), max(np.nanstd(yo), 1e-8), y_missing.sum())
    resid = yo - Xo1 @ coef
    sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    Xm1 = np.column_stack([np.ones(int(y_missing.sum())), X[y_missing]])
    return Xm1 @ coef + rng.normal(0, sd if sd > 0 else 1e-8, int(y_missing.sum()))


def mice(df: pd.DataFrame, m: int = 20, max_iter: int = 10, seed: int = 0,
         numeric_only: bool = True) -> List[pd.DataFrame]:
    """链式方程多重插补，返回 m 个完整数据集。

    只对数值列插补；非数值列若有缺失会被原样保留并在结果中标注（调用方自行处理）。
    """
    rng = np.random.default_rng(seed)
    work = df.copy()
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_only:
        mask = {c: df[c].isna().to_numpy() for c in num_cols}
        for c in num_cols:                       # 先做一次中位数填充作为迭代起点
            work[c] = work[c].fillna(work[c].median())

    datasets = []
    for _ in range(m):
        cur = work.copy()
        for _it in range(max_iter):
            for c in num_cols:
                if not mask[c].any():
                    continue
                others = [o for o in num_cols if o != c]
                X = cur[others].to_numpy(dtype=float) if others else np.empty((len(cur), 0))
                y = cur[c].to_numpy(dtype=float)
                filled = _regress_predict(X, y, mask[c], rng)
                new = cur[c].to_numpy(dtype=float).copy()
                new[mask[c]] = filled
                cur[c] = new
        datasets.append(cur)
    return datasets


def pool_rubin(estimates: Sequence[float],
               variances: Sequence[float]) -> Dict[str, float]:
    """Rubin's rules：合并 m 个估计。

    返回合并点估计、组内方差 U、组间方差 B、总方差 T、自由度 df 与 95% CI。
    """
    q = np.asarray(estimates, float)
    u = np.asarray(variances, float)
    m = len(q)
    if m == 0:
        return {}
    qbar = float(q.mean())
    ubar = float(u.mean())
    if m > 1:
        b = float(np.var(q, ddof=1))
        t = ubar + (1 + 1 / m) * b
        df = ((m - 1) * (1 + ubar / ((1 + 1 / m) * b)) ** 2) if b > 0 else np.inf
    else:
        b, t, df = 0.0, ubar, np.inf
    se = float(np.sqrt(t))
    # 自由度有限时用 t 分位；此处用 1.96 近似并在 t 很小时标注
    crit = 1.96
    return {"estimate": qbar, "within_var": ubar, "between_var": b,
            "total_var": t, "se": se, "df": float(df),
            "ci_low": qbar - crit * se, "ci_high": qbar + crit * se,
            "m": m, "note": ("df 有限时宜用 t 分位（t_{df,0.975}）；此处给正态近似值")}


def complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    """完整病例（敏感性分析对照）。"""
    return df.dropna()
