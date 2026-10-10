"""分析法 ⑮：样本量与功效计算 + 诊断准确性指标。

功效分析（**必须在数据收集前做**，事后功效分析没有信息量）：
  · `power_two_means`   — 两组均值比较
  · `power_two_props`   — 两组率比较
  · `power_survival`    — 生存结局（按事件数而非样本数计算）
  · `power_correlation` — 相关系数
  · `sample_size_*`     — 反解所需样本量

诊断准确性（**不能只报 AUC**）：
  · `diagnostic_accuracy` — 敏感性/特异性/PPV/NPV/似然比/DOR/Youden
  · `optimal_cutoff`      — Youden 最优截断值
  · `roc_curve`           — ROC 曲线点

⚠️ 事后功效（observed power）是**循环论证**：它完全由 p 值决定，
   p 显著 → 功效一定高，p 不显著 → 功效一定低，不提供任何新信息。
  阴性结果的正确做法是报告**效应量的置信区间**，而不是「功效只有 40%」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import stats as _st


def power_two_means(delta: float, sd: float, n1: int, n2: Optional[int] = None,
                    alpha: float = 0.05, alternative: str = "two-sided") -> Dict[str, Any]:
    """两组均值比较的功效（双样本 t 检验）。"""
    n2 = n1 if n2 is None else n2
    d = abs(float(delta)) / float(sd) if sd > 0 else 0.0
    df = n1 + n2 - 2
    se = np.sqrt(1.0 / n1 + 1.0 / n2)
    ncp = d / se                                  # 非中心参数
    if alternative == "two-sided":
        crit = _st.t.ppf(1 - alpha / 2, df)
        power = float(_st.nct.sf(crit, df, ncp) + _st.nct.cdf(-crit, df, ncp))
    else:
        crit = _st.t.ppf(1 - alpha, df)
        power = float(_st.nct.sf(crit, df, ncp))
    return {"power": power, "effect_size_cohens_d": d, "n1": int(n1), "n2": int(n2),
            "alpha": alpha, "alternative": alternative}


def sample_size_two_means(delta: float, sd: float, power: float = 0.8,
                          alpha: float = 0.05, ratio: float = 1.0,
                          alternative: str = "two-sided") -> Dict[str, Any]:
    """反解每组所需样本量（迭代求解 power = 目标值）。"""
    d = abs(float(delta)) / float(sd) if sd > 0 else 0.0
    lo, hi = 2, 100000
    z_a = _st.norm.ppf(1 - alpha / 2) if alternative == "two-sided" else _st.norm.ppf(1 - alpha)
    # 正态近似作初值
    z_b = _st.norm.ppf(power)
    n_apx = int(np.ceil(((z_a + z_b) ** 2) * (1 + 1 / ratio) / (d ** 2))) if d > 0 else hi
    lo = max(2, n_apx // 4)
    hi = max(n_apx * 4, 100)
    for _ in range(200):
        mid = (lo + hi) // 2
        p = power_two_means(delta, sd, mid, int(np.ceil(mid * ratio)),
                            alpha=alpha, alternative=alternative)["power"]
        if p < power:
            lo = mid + 1
        else:
            hi = mid
        if lo >= hi:
            break
    n1 = int(lo)
    return {"n_per_group": n1, "n_group2": int(np.ceil(n1 * ratio)),
            "total": int(n1 + np.ceil(n1 * ratio)),
            "effect_size_cohens_d": d, "target_power": power, "alpha": alpha}


def power_two_props(p1: float, p2: float, n1: int, n2: Optional[int] = None,
                    alpha: float = 0.05) -> Dict[str, Any]:
    """两组率比较的功效（正态近似，两组独立）。"""
    n2 = n1 if n2 is None else n2
    p1, p2 = float(p1), float(p2)
    pbar = (p1 * n1 + p2 * n2) / (n1 + n2)
    se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    se1 = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se0 == 0 or se1 == 0:
        return {"power": float("nan"), "note": "率退化（0 或 1），正态近似失效"}
    z_a = _st.norm.ppf(1 - alpha / 2)
    power = float(_st.norm.sf((z_a * se0 - abs(p1 - p2)) / se1)
                  + _st.norm.cdf((-z_a * se0 - abs(p1 - p2)) / se1))
    return {"power": power, "p1": p1, "p2": p2, "n1": int(n1), "n2": int(n2), "alpha": alpha}


def sample_size_two_props(p1: float, p2: float, power: float = 0.8,
                          alpha: float = 0.05, ratio: float = 1.0) -> Dict[str, Any]:
    """两组率比较的样本量（正态近似）。"""
    p1, p2 = float(p1), float(p2)
    z_a = _st.norm.ppf(1 - alpha / 2)
    z_b = _st.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2
    d = abs(p1 - p2)
    n1 = int(np.ceil(num / (d ** 2))) if d > 0 else -1
    return {"n_per_group": n1, "n_group2": int(np.ceil(n1 * ratio)),
            "total": int(n1 + np.ceil(n1 * ratio)), "target_power": power, "alpha": alpha}


def power_survival(hazard_ratio: float, n_events: int, alpha: float = 0.05,
                   allocation: float = 0.5) -> Dict[str, Any]:
    """生存结局的功效（Schoenfeld 公式：**按事件数**，不是样本数）。

    ⚠️ 生存分析的功效由**事件数**决定，不是入组人数。
       入组 1000 人但只随访到 20 个事件，功效远低于入组 200 人随访到 100 个事件。
    """
    hr = float(hazard_ratio)
    if hr <= 0:
        raise ValueError("hazard_ratio 必须为正")
    z_a = _st.norm.ppf(1 - alpha / 2)
    theta = np.log(hr)
    p = float(allocation)
    power = float(_st.norm.cdf(abs(theta) * np.sqrt(n_events * p * (1 - p)) - z_a))
    return {"power": power, "hazard_ratio": hr, "n_events": int(n_events),
            "alpha": alpha, "allocation": p,
            "note": "Schoenfeld：功效由**事件数**决定，不是入组数"}


def sample_size_survival(hazard_ratio: float, power: float = 0.8, alpha: float = 0.05,
                         allocation: float = 0.5) -> Dict[str, Any]:
    """生存分析所需**事件数**（再按预期事件率换算成入组人数）。"""
    hr = float(hazard_ratio)
    z_a = _st.norm.ppf(1 - alpha / 2)
    z_b = _st.norm.ppf(power)
    p = float(allocation)
    n_events = int(np.ceil(((z_a + z_b) / np.log(hr)) ** 2 / (p * (1 - p))))
    return {"n_events_required": n_events, "hazard_ratio": hr,
            "target_power": power, "alpha": alpha,
            "note": "得到的是事件数；入组数 = 事件数 / 预期事件率"}


def power_correlation(r: float, n: int, alpha: float = 0.05) -> Dict[str, Any]:
    """相关系数 r 的显著性检验功效（Fisher z 变换）。"""
    r = float(np.clip(r, -0.999999, 0.999999))
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z_a = _st.norm.ppf(1 - alpha / 2)
    power = float(_st.norm.sf(z_a - abs(z_r) / se) + _st.norm.cdf(-z_a - abs(z_r) / se))
    return {"power": power, "r": r, "n": int(n), "alpha": alpha}


def diagnostic_accuracy(y_true: Sequence[int], y_score: Sequence[float],
                        cutoff: Optional[float] = None) -> Dict[str, Any]:
    """诊断准确性全套指标（**不能只报 AUC**）。

    未给 cutoff 时用 Youden 最优截断。
    指标：敏感性、特异性、PPV、NPV、+LR、−LR、DOR、Youden J、准确率。
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if cutoff is None:
        cutoff = float(optimal_cutoff(y, s)["cutoff"])
    pred = (s >= cutoff).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    lr_plus = sens / (1 - spec) if (spec is not None and spec < 1 and np.isfinite(sens)) else float("nan")
    lr_minus = (1 - sens) / spec if (np.isfinite(sens) and spec and spec > 0) else float("nan")
    dor = (lr_plus / lr_minus) if (np.isfinite(lr_plus) and np.isfinite(lr_minus) and lr_minus) else float("nan")
    return {
        "cutoff": float(cutoff), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec,
        "ppv": ppv, "npv": npv,
        "lr_plus": lr_plus, "lr_minus": lr_minus, "dor": dor,
        "youden_j": sens + spec - 1 if np.isfinite(sens) and np.isfinite(spec) else float("nan"),
        "accuracy": (tp + tn) / len(y),
        "prevalence": float(np.mean(y)),
        "note": "PPV/NPV 依赖患病率，跨人群不可直接比较；LR 才是可迁移的指标",
    }


def optimal_cutoff(y_true: Sequence[int], y_score: Sequence[float]) -> Dict[str, Any]:
    """Youden J 最优截断值（sens + spec - 1 最大）。"""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    cand = np.unique(s)
    best_j, best_c = -np.inf, float(np.median(s))
    for c in cand:
        pred = (s >= c).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_c = j, float(c)
    return {"cutoff": best_c, "youden_j": float(best_j)}


def roc_curve(y_true: Sequence[int], y_score: Sequence[float],
              n_points: Optional[int] = None) -> pd.DataFrame:
    """ROC 曲线点（FPR, TPR）。"""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    order = np.argsort(-s)
    ss = s[order]
    yy = y[order]
    # 折点：每个不同分数组的**最后一个**（累计到组末才完整）
    keep = np.r_[np.diff(ss) != 0, True]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    n_pos = int(yy.sum())
    n_neg = int(len(yy) - n_pos)
    fpr = np.r_[0.0, (fp / n_neg)[keep]]
    tpr = np.r_[0.0, (tp / n_pos)[keep]]
    df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": np.r_[np.inf, ss[keep]]})
    if n_points and len(df) > n_points:
        idx = np.linspace(0, len(df) - 1, n_points).astype(int)
        df = df.iloc[idx].reset_index(drop=True)
    # 梯形法 AUC
    auc = float(np.trapezoid(df["tpr"].to_numpy(), df["fpr"].to_numpy()))
    df.attrs["auc"] = auc
    return df
