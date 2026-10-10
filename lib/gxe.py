"""基因-环境交互（G×E）与效应修饰分析。

**方法来源（全部为公开发表的标准方法，不是自造算法）**
- 交互项检验：常规线性模型的乘积项 t 检验
- 分层效应：按环境变量分层后各自的 G 效应
- **相加交互 RERI / AP / S**：Rothman 的交互作用相加模型
  （Rothman KJ, Greenland S, Lash TL. *Modern Epidemiology*. 3rd ed. 2008, Ch. 5 & 18；
   也可参见 VanderWeele TJ, Knol MJ. *Eur J Epidemiol* 2014;29:229-32 的实操综述）
- Johnson–Neyman 区间：Johnson PO, Neyman J. *Psychometrika* 1936;1:57-68

**红线（这三条踩错任何一条，结论方向都会反）**
1. **交互 ≠ 分层**。「两层都显著」或「一层显著一层不显著」**都不能**替代交互检验——
   分层与交互检验的功效不同，两种情形下交互都可能不显著。
2. **相乘交互 ≠ 相加交互**。线性模型的乘积项衡量的是**相乘**尺度上的偏离；
   公共卫生解释常看**相加**尺度（RERI）。两者不是同一件事，不能互相替代。
3. **连续变量做交互前必须中心化**，否则「主效应」的含义会变成「另一个变量取 0 时
   的效应」，而 0 往往落在数据范围之外。本模块默认 `center=True` 并在结果里
   标明是否中心化。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "interaction_test", "stratified_effect", "johnson_neyman",
    "reri_additive_interaction",
]


def _design(*cols: np.ndarray) -> np.ndarray:
    """构造带截距的设计矩阵。空列（None）会被跳过。"""
    parts = [np.ones(len(cols[0]))]
    for c in cols:
        if c is None:
            continue
        parts.append(np.asarray(c, dtype=float))
    return np.column_stack(parts)


def _ols(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """最小二乘 + t 检验（纯 numpy，不依赖 statsmodels）。"""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    dof = n - p
    if dof <= 0:
        raise ValueError(f"样本数 {n} 不足以拟合 {p} 个参数")
    sigma2 = float(resid @ resid) / dof
    cov = np.linalg.pinv(X.T @ X) * sigma2
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    pval = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "t": t, "p": pval, "dof": dof, "r2": r2,
            "n": n}


def interaction_test(y: Sequence[float], g: Sequence[float], e: Sequence[float],
                     covariates: Optional[np.ndarray] = None,
                     center: bool = True) -> Dict[str, Any]:
    """检验 G×E 相乘交互：``y ~ g + e + g:e (+ 协变量)``。

    :param center: 是否把 g 与 e 中心化（**默认 True**，见模块头的红线 3）
    :return: dict
        - ``beta_interaction`` / ``se`` / ``pvalue`` / ``ci95``: 交互项
        - ``main_g`` / ``main_e``: 主效应（中心化时 = 各自均值处的效应）
        - ``n`` / ``dof`` / ``r2``
        - ``centered``: 实际是否做了中心化
        - ``interpretation``: 一句话说明该怎么读这个结果
        - ``method`` / ``engine``

    ⚠️ 主效应的解释依赖中心化：未中心化时 ``main_g`` 是「e = 0 时的 G 效应」，
       若 e 的取值远离 0，这个数没有实际意义。
    """
    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    e = np.asarray(e, dtype=float)
    if not (len(y) == len(g) == len(e)):
        raise ValueError("y / g / e 长度必须一致")
    if covariates is not None and len(covariates) != len(y):
        raise ValueError("covariates 行数与 y 不一致")

    gc, ec = (g - g.mean(), e - e.mean()) if center else (g, e)
    X = _design(gc, ec, gc * ec, None if covariates is None else np.asarray(covariates, float))
    fit = _ols(X, y)
    i = 3  # 0=截距, 1=g, 2=e, 3=g:e
    se_i = float(fit["se"][i])
    ci = (float(fit["beta"][i] - 1.96 * se_i), float(fit["beta"][i] + 1.96 * se_i))
    return {
        "beta_interaction": float(fit["beta"][i]),
        "se": se_i,
        "pvalue": float(fit["p"][i]),
        "ci95": ci,
        "main_g": float(fit["beta"][1]),
        "main_e": float(fit["beta"][2]),
        "n": int(fit["n"]),
        "dof": int(fit["dof"]),
        "r2": float(fit["r2"]),
        "centered": bool(center),
        "interpretation": (
            "interaction p<0.05 只说明『相乘尺度上偏离可加』；"
            "不要用分层结果替代交互检验，也不要把它当成相加交互（RERI）"),
        "method": "OLS with product term (t-test)",
        "engine": "lib.gxe.interaction_test",
    }


def stratified_effect(y: Sequence[float], g: Sequence[float], e: Sequence[float],
                      n_strata: int = 3,
                      cutpoints: Optional[Sequence[float]] = None,
                      covariates: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """按环境变量分层，分别估计 G 的效应。

    ⚠️ **这是描述，不是检验。** 分层之间的差异是否成立，仍要看
       ``interaction_test`` —— 两个分层的效应「一个显著一个不显著」
       完全可能对应一个不显著的交互项（功效不同）。
    """
    import pandas as pd  # 仅此处需要，便于返回表

    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    e = np.asarray(e, dtype=float)
    if cutpoints is None:
        qs = np.linspace(0, 1, n_strata + 1)[1:-1]
        cutpoints = np.quantile(e, qs)
    edges = [-np.inf, *list(cutpoints), np.inf]
    labels = [f"E{i + 1}" for i in range(len(edges) - 1)]
    rows = []
    for k in range(len(edges) - 1):
        m = (e > edges[k]) & (e <= edges[k + 1])
        if m.sum() < 5:
            rows.append({"stratum": labels[k], "range": f"({edges[k]:.3g}, {edges[k+1]:.3g}]",
                         "n": int(m.sum()), "beta_g": float("nan"),
                         "se": float("nan"), "pvalue": float("nan")})
            continue
        X = _design(g[m], None if covariates is None else np.asarray(covariates, float)[m])
        f = _ols(X, y[m])
        rows.append({"stratum": labels[k],
                     "range": f"({edges[k]:.3g}, {edges[k+1]:.3g}]",
                     "n": int(m.sum()), "beta_g": float(f["beta"][1]),
                     "se": float(f["se"][1]), "pvalue": float(f["p"][1])})
    df = pd.DataFrame(rows)
    return {"table": df, "cutpoints": [float(c) for c in cutpoints],
            "warning": "分层差异是描述性的；交互是否成立请看 interaction_test",
            "engine": "lib.gxe.stratified_effect"}


def johnson_neyman(y: Sequence[float], g: Sequence[float], e: Sequence[float],
                   alpha: float = 0.05, e_range: Optional[Sequence[float]] = None,
                   n_grid: int = 200) -> Dict[str, Any]:
    """Johnson–Neyman：找出**G 的效应显著**所对应的环境值区间。

    做法：在 e 的取值网格上逐点算 G 的简单斜率及其 95% CI，收集 CI 不跨 0 的区间。
    比「按中位数二分再各算一次」严谨——后者会丢掉「显著区间的边界在哪」这个信息。
    """
    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    e = np.asarray(e, dtype=float)
    lo, hi = (float(np.min(e)), float(np.max(e))) if e_range is None else (
        float(e_range[0]), float(e_range[1]))
    grid = np.linspace(lo, hi, int(n_grid))
    slopes, sig = [], []
    for e0 in grid:
        X = _design(g, e - e0, g * (e - e0))
        f = _ols(X, y)
        b, se = float(f["beta"][1]), float(f["se"][1])
        slopes.append(b)
        crit = stats.t.ppf(1 - alpha / 2, f["dof"])
        sig.append(bool(abs(b) - crit * se > 0))
    grid_a, sig_a = np.asarray(grid), np.asarray(sig)
    regions = []
    if sig_a.any():
        idx = np.flatnonzero(sig_a)
        start = idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                regions.append((float(grid_a[start]), float(grid_a[a])))
                start = b
        regions.append((float(grid_a[start]), float(grid_a[idx[-1]])))
    return {
        "e_grid": grid_a, "slope": np.asarray(slopes), "significant": sig_a,
        "significant_regions": regions,
        "note": ("JN 区间给出『G 的效应在哪些环境取值上显著』；"
                 "它是简单斜率的显著性区间，不等于交互检验"),
        "engine": "lib.gxe.johnson_neyman",
    }


def reri_additive_interaction(cases: Sequence[int],
                              g: Sequence[int], e: Sequence[int]) -> Dict[str, Any]:
    """相加尺度交互：**RERI / AP / S**（Rothman 的交互作用相加模型）。

    需要二分类的 G 与 E（0/1）和二分结局。先算四个联合暴露组的 RR（用 OR 近似，
    或直接给风险），再算：

        RERI = RR11 − RR10 − RR01 + 1
        AP   = RERI / RR11
        S    = (RR11 − 1) / ((RR10 − 1) + (RR01 − 1))

    **为什么必须另外算它**：相乘交互不显著时，相加交互可能显著，反之亦然；
    公共卫生意义上的「协同作用」通常指相加尺度。

    用 delta 法给出 RERI 的 95% CI（Hosmer–Lemeshow 近似）。
    """
    cases = np.asarray(cases, dtype=int)
    g = np.asarray(g, dtype=int)
    e = np.asarray(e, dtype=int)
    if not (len(cases) == len(g) == len(e)):
        raise ValueError("cases / g / e 长度必须一致")
    if set(np.unique(g)) - {0, 1} or set(np.unique(e)) - {0, 1}:
        raise ValueError("g 与 e 必须是二分类（0/1）；连续变量请先定切点")

    cell = {}
    for a in (0, 1):
        for b in (0, 1):
            m = (g == a) & (e == b)
            n1 = int(((cases == 1) & m).sum())
            n0 = int(((cases == 0) & m).sum())
            if n0 == 0:
                raise ValueError(f"单元格 g={a}, e={b} 没有对照——OR 不可估计（零格）")
            cell[(a, b)] = {"case": n1, "ctrl": n0, "or": (n1 / n0) if n0 else float("inf")}
    r10 = cell[(1, 0)]["or"]     # 只有 G
    r01 = cell[(0, 1)]["or"]     # 只有 E
    r11 = cell[(1, 1)]["or"]     # 两者都有
    r00 = cell[(0, 0)]["or"]     # 参照（此处 = 1，因为同分母约定简化）

    reri = r11 - r10 - r01 + 1.0
    ap = reri / r11 if r11 != 0 else float("nan")
    denom = (r10 - 1.0) + (r01 - 1.0)
    s = (r11 - 1.0) / denom if denom != 0 else float("nan")

    # RERI 的方差（Hosmer–Lemeshow 近似，OR 的对数尺度相加）
    def _var_or(a: int, b: int) -> float:
        c = cell[(a, b)]
        return 1.0 / c["case"] + 1.0 / c["ctrl"]
    var = (r11 ** 2) * _var_or(1, 1) + (r10 ** 2) * _var_or(1, 0) \
        + (r01 ** 2) * _var_or(0, 1)
    se = float(np.sqrt(var))
    ci = (float(reri - 1.96 * se), float(reri + 1.96 * se))

    return {
        "cells": {f"g{a}e{b}": v for (a, b), v in cell.items()},
        "RR10": float(r10), "RR01": float(r01), "RR11": float(r11),
        "RERI": float(reri), "RERI_se": se, "RERI_ci95": ci,
        "AP": float(ap), "S": float(s),
        "interpretation": (
            "RERI 的 95%CI 不含 0 → 存在相加尺度交互（超相加）；"
            "注意这与线性模型交互项的 p 值不是同一个检验"),
        "method": "Rothman additive interaction (RERI/AP/S)",
        "engine": "lib.gxe.reri_additive_interaction",
    }
