"""年龄失守点（age onset）—— **本模块是全项目「失守点」的唯一裁决点**。

同一逻辑只允许一个裁决点。任何脚本要「某个基因在几岁开始失守」这个数，
**必须**调用本模块，不得在别处另写一份实现（否则同一份数据会给出两个年龄）。

## 「失守点」到底指什么（口径先于数值）

它不是「表达最低的年龄」，也不是「变化最快的年龄」——这些直觉在噪声数据上
会给出完全不同的数。本模块固定两种**可复现**的口径，默认用第一种：

**口径 A · ``ci_departure``（默认，基于置信区间）**
  1. 基线 ``baseline`` = GAM 曲线在**最小年龄**处的拟合值（不是观测均值——
     观测均值受高年龄端噪声拉动）。
  2. 在年龄网格上求 95% 置信带。
  3. ``onset`` = 最早的年龄 t，使得 ``CI 上界(t) < baseline``（即曲线**显著低于**
     基线）**且从 t 到最大年龄持续成立**。
  「持续成立」是必要的：不加这条，单点噪声就会被当成失守。
  - 数学含义清晰、可检验；缺点是需要 95% 置信带（依赖 GAM 的 se）。

**口径 B · ``relative_drop``（相对降幅）**
  ``onset`` = 最早的年龄 t，使 ``μ(t) ≤ baseline × (1 − drop_fraction)`` 且此后持续成立。
  - 直观、不依赖置信带；但 ``drop_fraction`` 是人为阈值，**必须随结果输出**。

**方向**：``direction`` 记录曲线是「下降（down）」还是「上升（up）」。
只有 down 才是「失守」；up 是「开启」，语义不同，不得混用。

⚠️ **未检出**（曲线没显著偏离基线）时 ``onset_age = None``，**不是 0**
——把「没检出」写成 0 会让下游把"没有证据"当成"出生即失守"。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from vendor._gam import core as _gam

__all__ = ["ONSET_RULES", "fit_onset", "screen_age_effects", "summarize_screen"]

ONSET_RULES = ("ci_departure", "relative_drop")


def _norm_ppf(p: float) -> float:
    """标准正态分位数（去 scipy 依赖的轻量实现，用已有 vendor 统计量亦可）。

    这里直接用 scipy——它已在 Tier1 依赖里；写成独立函数便于替换。
    """
    from scipy import stats as _st
    return float(_st.norm.ppf(p))


def _persistent_from(mask: np.ndarray) -> Optional[int]:
    """返回 ``mask`` 中「从该位置起一直为 True」的第一个下标；不存在则 None。"""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    # 从后往前扫，找最后一个 False，其后的第一个 True 就是持续段起点
    false_idx = np.flatnonzero(~mask)
    if false_idx.size == 0:
        return int(0)
    last_false = int(false_idx[-1])
    if last_false + 1 >= mask.size:
        return None
    return int(last_false + 1)


def fit_onset(ages: Sequence[float], expr: Sequence[float],
              rule: str = "ci_departure", drop_fraction: float = 0.5,
              n_splines: int = 6, spline_order: int = 3,
              conf: float = 0.95, grid_n: int = 100,
              alpha: float = 0.05, baseline_frac: float = 0.10,
              lam: Optional[float] = None, lams: Optional[Sequence[float]] = None,
              gamma: float = 1.4) -> Dict[str, Any]:
    """**单基因**失守点（唯一裁决点的入口）。

    :param rule: ``ci_departure``（默认）或 ``relative_drop``
    :param drop_fraction: 口径 B 的降幅阈值（0<f<1），随结果输出
    :param n_splines: 样条数。默认 6（保守）——**年龄点少时样条数必须小**，
        否则会拟合到穿点而过。经验法则：每个样条至少 3-5 个观测。
    :param conf: 置信水平（口径 A 用）
    :param grid_n: 求 onset 的年龄网格密度
    :param alpha: **趋势显著性门槛**。趋势不显著时**一律不给失守点**
        （返回 ``onset_age=None``）——否则噪声的局部波动会被当成失守。
    :param baseline_frac: 基线取「年龄网格最年轻的这一段」的拟合值均值
        （默认前 10%）。**不用单点**：样条在端点上方差大，单点基线会让 onset 系统性偏早。
    :param lam: 固定平滑参数；None 则 GCV 选
    :return: dict；未检出趋势时 ``onset_age=None`` 且 ``detected=False``
    """
    if rule not in ONSET_RULES:
        raise ValueError(f"rule 必须是 {ONSET_RULES} 之一，收到 {rule!r}")
    ages = np.asarray(ages, dtype=float).ravel()
    expr = np.asarray(expr, dtype=float).ravel()
    if ages.shape[0] != expr.shape[0]:
        raise ValueError(f"ages 长度 {ages.shape[0]} != expr 长度 {expr.shape[0]}")
    ok = np.isfinite(ages) & np.isfinite(expr)
    ages, expr = ages[ok], expr[ok]
    if ages.shape[0] < 6:
        raise ValueError(f"有效样本 {ages.shape[0]} 太少（<6），无法拟合失守点")

    if not (0.0 < float(drop_fraction) < 1.0):
        raise ValueError("drop_fraction 必须在 (0,1)")
    if not (0.0 < float(baseline_frac) < 0.5):
        raise ValueError("baseline_frac 必须在 (0,0.5)")

    model = _gam.fit_gam(ages, expr, n_splines=int(n_splines),
                         spline_order=int(spline_order), lam=lam, lams=lams, gamma=gamma)

    # 常数表达（所有细胞完全相同）：不可能存在年龄效应，直接判「未检出」。
    # 不能放它进后面的流程——零宽置信带会把任何数值噪声判成显著偏离。
    if model.get("degenerate"):
        return {
            "onset_age": None, "direction": "flat", "rule": rule,
            "detected": False, "pvalue": 1.0,
            "degenerate": True,
            "degenerate_reason": "该基因在所有细胞里表达完全相同（方差为 0）——不可能有年龄效应",
            "edof": model["edof"], "lam": model["lam"],
            "lam_selection": model["lam_selection"],
            "r2": float("nan"), "gcv": model["gcv"],
            "delta_gcv_vs_intercept": float("nan"),
            "n": model["n"],
            "age_range": [float(np.min(ages)), float(np.max(ages))],
            "baseline": float(np.mean(expr)),
            "grid_ages": np.array([]), "grid_fitted": np.array([]),
            "grid_lo": np.array([]), "grid_hi": np.array([]),
            "n_splines": int(n_splines), "spline_order": int(spline_order),
            "gamma": float(gamma), "detail": {"rule_desc": "退化输入，未做失守点判定"},
            "engine": "lib.age_onset.fit_onset",
            "caveat": "退化输入（常数表达）",
        }

    # 年龄网格（严格落在数据范围内，不外推——外推区没有数据支撑）
    t_lo, t_hi = float(np.min(ages)), float(np.max(ages))
    t_grid = np.linspace(t_lo, t_hi, int(grid_n))
    B_grid = _gam.predict_basis(model, t_grid)
    mu_grid = B_grid @ model["coef"]

    # 拟合值标准误：Var(mu)=Σ(B·B̂)²·scale²（与 fit_once 同式）
    scale = model["scale"] if model["scale"] == model["scale"] else 0.0
    se_grid = _se_on_grid(model, B_grid, scale)

    z = _norm_ppf(1 - (1 - float(conf)) / 2.0)
    lo = mu_grid - z * se_grid
    hi = mu_grid + z * se_grid

    # 基线 = 最年轻那一段（默认前 10% 网格）的拟合值均值。
    # **不用单点**：样条在端点上方差最大，单点基线会让 onset 系统性偏早
    # （实测：单点基线把真值 40 岁的失守点判到 32.3 岁，偏差 8 年）。
    n_base = max(1, int(round(int(grid_n) * float(baseline_frac))))
    baseline = float(np.mean(mu_grid[:n_base]))

    # 趋势检验（GAM vs 截距模型）；用 F 近似 + GCV 差
    trend = _trend_test(model, ages, expr)
    detected = bool(trend["pvalue"] < float(alpha))

    direction = "flat"
    onset_age: Optional[float] = None
    onset_detail: Dict[str, Any] = {
        "baseline": baseline,
        "baseline_frac": float(baseline_frac),
        "baseline_desc": f"年龄网格最年轻的 {baseline_frac:.0%} 段（{n_base} 个网格点）的拟合值均值",
    }

    # ── 关键红线：趋势不显著 → 一律不给失守点 ──
    # 噪声数据在某个年龄段必然出现 CI 偏离基线，若不显著也报 onset，
    # 下游会把「本来没有的年龄效应」当成发现。
    if not detected:
        onset_detail["rule_desc"] = (
            f"趋势不显著（p={trend['pvalue']:.4g} >= alpha={alpha}）——不判定失守点")
        return {
            "onset_age": None, "direction": "flat", "rule": rule,
            "detected": False, "degenerate": False,
            "pvalue": trend["pvalue"], "alpha": float(alpha),
            "edof": model["edof"], "lam": model["lam"],
            "lam_selection": model["lam_selection"],
            "r2": trend["r2"], "gcv": model["gcv"],
            "delta_gcv_vs_intercept": trend["delta_gcv"],
            "n": model["n"],
            "age_range": [t_lo, t_hi],
            "baseline": baseline,
            "grid_ages": t_grid, "grid_fitted": mu_grid,
            "grid_lo": lo, "grid_hi": hi,
            "n_splines": int(n_splines), "spline_order": int(spline_order),
            "gamma": float(gamma),
            "detail": onset_detail,
            "engine": "lib.age_onset.fit_onset",
            "caveat": ("pvalue 为 F 近似（GAM 平滑项检验无唯一解，mgcv 另有一套）；"
                       "onset_age 是网格分辨率下的值（grid_n）；"
                       "人口细胞测得的年龄是供体年龄，非细胞年龄"),
        }

    if rule == "ci_departure":
        down_mask = hi < baseline
        up_mask = lo > baseline
        d_idx = _persistent_from(down_mask)
        u_idx = _persistent_from(up_mask)
        if d_idx is not None and (u_idx is None or d_idx <= u_idx):
            direction, onset_age = "down", float(t_grid[d_idx])
        elif u_idx is not None:
            direction, onset_age = "up", float(t_grid[u_idx])
        onset_detail.update({
            "conf": float(conf), "z": z,
            "rule_desc": (f"最早使 CI 上界 < 基线（{baseline:.6g}）且此后持续成立的年龄；"
                          f"置信水平 {conf}"),
        })
    else:  # relative_drop
        thresh = baseline * (1.0 - float(drop_fraction))
        down_mask = mu_grid <= thresh
        up_mask = mu_grid >= baseline * (1.0 + float(drop_fraction))
        d_idx = _persistent_from(down_mask)
        u_idx = _persistent_from(up_mask)
        if d_idx is not None and (u_idx is None or d_idx <= u_idx):
            direction, onset_age = "down", float(t_grid[d_idx])
        elif u_idx is not None:
            direction, onset_age = "up", float(t_grid[u_idx])
        onset_detail.update({
            "drop_fraction": float(drop_fraction),
            "threshold": float(thresh),
            "rule_desc": (f"最早使 μ(t) ≤ 基线×{1 - drop_fraction:.3f}（={thresh:.6g}）"
                          "且此后持续成立的年龄（相对降幅口径）"),
        })

    return {
        "onset_age": onset_age,
        "direction": direction,
        "rule": rule,
        "detected": True,
        "degenerate": False,
        "pvalue": trend["pvalue"],
        "alpha": float(alpha),
        "edof": model["edof"],
        "lam": model["lam"],
        "lam_selection": model["lam_selection"],
        "r2": trend["r2"],
        "gcv": model["gcv"],
        "delta_gcv_vs_intercept": trend["delta_gcv"],
        "n": model["n"],
        "age_range": [t_lo, t_hi],
        "baseline": baseline,
        "grid_ages": t_grid,
        "grid_fitted": mu_grid,
        "grid_lo": lo,
        "grid_hi": hi,
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "gamma": float(gamma),
        "detail": onset_detail,
        "engine": "lib.age_onset.fit_onset",
        "caveat": ("pvalue 为 F 近似（GAM 平滑项检验无唯一解，mgcv 另有一套）；"
                   "onset_age 是网格分辨率下的值（grid_n）；"
                   "人口细胞测得的年龄是供体年龄，非细胞年龄"),
    }


def _se_on_grid(model: Dict[str, Any], B_grid: np.ndarray, scale: float) -> np.ndarray:
    """在给定基矩阵上计算拟合值标准误。

    与 ``fit_once`` 内 ``se_fit`` **同式**：``Var(μ_i) = Σ_j (B_i · B̂[:, j])² · scale²``。
    直接复用已拟合的 ``B_hat``，不重复求解。
    """
    B_hat = model["B_hat"]
    s = scale if scale == scale else 0.0
    scaled = (B_grid @ B_hat) * s
    return np.sqrt(np.clip(np.sum(scaled ** 2, axis=1), 0, None))


def _trend_test(model: Dict[str, Any], ages: np.ndarray, expr: np.ndarray) -> Dict[str, Any]:
    """GAM 相对**截距模型**的增量检验。

    - ``pvalue``：F 近似。``F = ((RSS0−RSS1)/(edof1−1)) / (RSS1/(n−edof1))``，
      自由度 ``(edof1−1, n−edof1)``。⚠️ 是近似，不是精确检验。
    - ``delta_gcv``：截距模型 GCV − GAM GCV（>0 表示 GAM 更好）。
    - ``r2``：``1 − RSS1/RSS0``（偏离 1 的程度受 edof 影响，仅作参考）。
    """
    from scipy import stats as _st

    y = expr
    n = y.shape[0]
    mu0 = np.full(n, float(np.mean(y)))
    rss0 = float(np.sum((y - mu0) ** 2))
    rss1 = float(model["deviance"])
    edof1 = float(model["edof"])

    df1 = max(edof1 - 1.0, 1e-9)
    df2 = n - edof1
    if rss0 <= 0 or df2 <= 0:
        pval = 1.0
    else:
        f = ((rss0 - rss1) / df1) / (rss1 / df2) if rss1 > 0 else float("inf")
        f = max(f, 0.0)
        pval = float(_st.f.sf(f, df1, df2))

    # 截距模型的 GCV（edof=1，deviance=rss0）
    gcv0 = _gam.gcv_score(rss0, n, 1.0, gamma=model["gamma"])

    return {
        "pvalue": pval,
        "f_approx": True,
        "r2": float(1 - rss1 / rss0) if rss0 > 0 else float("nan"),
        "delta_gcv": float(gcv0 - model["gcv"]),
        "rss0": rss0,
        "rss1": rss1,
    }


def screen_age_effects(ages: Sequence[float], expr, gene_names: Optional[Sequence[str]] = None,
                       groups: Optional[Sequence[Any]] = None,
                       rule: str = "ci_departure", drop_fraction: float = 0.5,
                       fdr: float = 0.05, min_cells: int = 10,
                       n_splines: int = 6, spline_order: int = 3,
                       conf: float = 0.95, grid_n: int = 100,
                       alpha: float = 0.05, baseline_frac: float = 0.10,
                       gamma: float = 1.4,
                       lams: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """批量筛查：对每个（分组 × 基因）拟合 GAM，输出失守点表 + BH-FDR。

    对应判决性预检的判据「≥N 个基因呈年龄趋势（FDR<0.05）」。

    :param ages: 长度 = 细胞数 的年龄
    :param expr: (细胞 × 基因) 表达矩阵（DataFrame 或 ndarray）
    :param groups: 可选。给定时对每个分组**分别**拟合（如按细胞类型）
    :param min_cells: 该分组内有效细胞少于此数则跳过该组合（并记录跳过原因）
    :return: dict
        - ``table``: 长表（group, gene, onset_age, direction, pvalue, padj, ...）
        - ``n_significant`` / ``fdr`` / ``rule`` / ``genes_all_flat``
        - ``skipped``: ``{原因: [组合...]}``——**跳过必须可见**，不得静默丢弃
    """
    if rule not in ONSET_RULES:
        raise ValueError(f"rule 必须是 {ONSET_RULES} 之一，收到 {rule!r}")

    ages = np.asarray(ages, dtype=float).ravel()
    if hasattr(expr, "columns"):
        genes = [str(c) for c in expr.columns]
        M = expr.to_numpy(dtype=float)
    else:
        if gene_names is None:
            raise ValueError("ndarray 输入必须给 gene_names")
        genes = [str(g) for g in gene_names]
        M = np.asarray(expr, dtype=float)

    if M.shape[0] != ages.shape[0]:
        raise ValueError(f"expr 行数 {M.shape[0]} != ages 长度 {ages.shape[0]}")

    if groups is None:
        grp = np.array(["all"] * ages.shape[0])
    else:
        grp = np.array([str(g) for g in groups])
        if grp.shape[0] != ages.shape[0]:
            raise ValueError(f"groups 长度 {grp.shape[0]} != ages 长度 {ages.shape[0]}")

    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, List[str]] = {}

    def _skip(reason: str, tag: str):
        skipped.setdefault(reason, []).append(tag)

    for g in list(dict.fromkeys(grp.tolist())):
        m = grp == g
        n_g = int(m.sum())
        if n_g < min_cells:
            _skip(f"该分组有效细胞 {n_g} < min_cells={min_cells}", g)
            continue
        a_g = ages[m]
        if np.unique(a_g).size < 2:
            _skip("该分组内年龄只有 1 个取值", g)
            continue
        for j, gene in enumerate(genes):
            y = M[m, j]
            try:
                r = fit_onset(a_g, y, rule=rule, drop_fraction=drop_fraction,
                              n_splines=min(n_splines, max(3, n_g // 4)),
                              spline_order=spline_order, conf=conf, grid_n=grid_n,
                              alpha=alpha, baseline_frac=baseline_frac,
                              gamma=gamma, lams=lams)
            except ValueError as e:
                _skip(f"拟合失败：{str(e)[:60]}", f"{g}|{gene}")
                continue
            rows.append({
                "group": g, "gene": gene,
                "onset_age": r["onset_age"], "direction": r["direction"],
                "pvalue": r["pvalue"], "edof": r["edof"],
                "r2": r["r2"], "lam": r["lam"],
                "degenerate": bool(r.get("degenerate", False)),
                "delta_gcv_vs_intercept": r["delta_gcv_vs_intercept"],
                "n_cells": n_g, "age_min": r["age_range"][0], "age_max": r["age_range"][1],
            })

    table = pd.DataFrame(rows)
    if table.empty:
        return {
            "table": table, "n_rows": 0, "n_significant": 0, "fdr": float(fdr),
            "rule": rule, "skipped": skipped,
            "engine": "lib.age_onset.screen_age_effects",
        }

    # BH-FDR（对全部检验，不按分组分别校正——分组是分层而非独立族，
    # 若要按族校正请显式分组后多次调用）
    p = table["pvalue"].to_numpy(dtype=float)
    order = np.argsort(p)
    n = p.size
    padj = np.empty(n, dtype=float)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    padj[order] = np.clip(ranked, 0, 1)
    table["padj"] = padj
    table = table.sort_values(["pvalue"]).reset_index(drop=True)

    sig = table[table["padj"] < float(fdr)]
    sig_with_onset = sig[sig["onset_age"].notna()]

    return {
        "table": table,
        "n_rows": int(table.shape[0]),
        "n_significant": int(sig.shape[0]),
        "n_significant_with_onset": int(sig_with_onset.shape[0]),
        "fdr": float(fdr),
        "alpha": float(alpha),
        "rule": rule,
        "drop_fraction": float(drop_fraction) if rule == "relative_drop" else None,
        "baseline_frac": float(baseline_frac),
        "skipped": skipped,
        "engine": "lib.age_onset.screen_age_effects",
        "rule_desc": (f"逐个（分组×基因）拟合 GAM；失守点口径 = {rule}；"
                      f"趋势门槛 alpha={alpha}（不显著一律不给失守点）；"
                      f"BH-FDR = {fdr}；方向 down 才是「失守」，up 是「开启」"),
    }


def summarize_screen(res: Dict[str, Any], top: int = 20) -> pd.DataFrame:
    """把筛查结果压成可读摘要（按 FDR 排序）。"""
    t = res.get("table")
    if t is None or t.empty:
        return pd.DataFrame(columns=["group", "gene", "onset_age", "direction",
                                     "pvalue", "padj"])
    cols = ["group", "gene", "onset_age", "direction", "pvalue", "padj", "r2", "n_cells"]
    cols = [c for c in cols if c in t.columns]
    return t[cols].head(int(top)).reset_index(drop=True)
