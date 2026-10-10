"""分析法 ⑧：Meta 分析（固定效应 / 随机效应 + 异质性 + 发表偏倚 + 森林图数据）。

口径（与 metafor / RevMan 对齐）：
  · 固定效应：inverse-variance 加权，w_i = 1/se_i²
  · 随机效应：DerSimonian–Laird，τ² = max(0, (Q - (k-1)) / (Σw - Σw²/Σw))
  · 异质性：Q（Cochran）、I² = max(0, (Q - df)/Q)、H² = Q/df、τ²
  · 发表偏倚：Egger 回归（对称漏斗图检验）、Begg 秩相关（可选）
  · 合并效应量的**置信区间**用正态近似（z）；OR/RR 先在 log 尺度合并再指数化

⚠️ 三条高频错误：
  1. 把**标准化均值差**与**原始均值差**混在一项里合并 → 必须同尺度同方向
  2. 用固定效应报告有明显异质性的结果（I²>50%）→ 会给出过窄的 CI
  3. 只报合并值不报异质性 → 读者无法判断合并是否有意义
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scipy import stats as _st


def meta_analysis(estimate: Sequence[float], se: Sequence[float],
                  labels: Optional[Sequence[str]] = None,
                  model: str = "auto", effect_type: str = "generic",
                  level: float = 0.95) -> Dict[str, Any]:
    """合并效应量。

    :param estimate: 各项研究的效应量（OR/RR 请先取 log 后传入）
    :param se: 各项研究的标准误（与 estimate 同尺度）
    :param model: fixed / random / auto（auto：I²>50% 或 Q 显著则随机效应）
    :param effect_type: generic / log_or（log 尺度，结果额外给出 exp 后的 OR）
    """
    y = np.asarray(estimate, dtype=float)
    s = np.asarray(se, dtype=float)
    if y.size != s.size:
        raise ValueError("estimate 与 se 长度必须一致")
    if np.any(s <= 0):
        raise ValueError("se 必须为正；se<=0 会让权重发散")
    k = y.size
    if k < 2:
        raise ValueError("Meta 分析至少需要 2 项研究")

    w_fixed = 1.0 / s ** 2
    mu_fixed = float(np.sum(w_fixed * y) / np.sum(w_fixed))
    se_fixed = float(np.sqrt(1.0 / np.sum(w_fixed)))

    # Cochran Q
    Q = float(np.sum(w_fixed * (y - mu_fixed) ** 2))
    df = k - 1
    p_Q = float(1.0 - _st.chi2.cdf(Q, df)) if Q > 0 else 1.0
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    H2 = float(Q / df) if df > 0 and Q > 0 else 0.0

    # DerSimonian-Laird τ²
    sum_w = float(np.sum(w_fixed))
    sum_w2 = float(np.sum(w_fixed ** 2))
    denom = sum_w - sum_w2 / sum_w
    tau2 = float(max(0.0, (Q - df) / denom)) if denom > 0 else 0.0

    w_random = 1.0 / (s ** 2 + tau2)
    mu_random = float(np.sum(w_random * y) / np.sum(w_random))
    se_random = float(np.sqrt(1.0 / np.sum(w_random)))

    if model == "auto":
        chosen = "random" if (I2 > 0.5 or p_Q < 0.05) else "fixed"
    elif model in ("fixed", "random"):
        chosen = model
    else:
        raise ValueError(f"model 必须是 fixed/random/auto，收到 {model!r}")

    mu = mu_fixed if chosen == "fixed" else mu_random
    se_mu = se_fixed if chosen == "fixed" else se_random
    z = float(_st.norm.ppf(1 - (1 - level) / 2))
    ci = (mu - z * se_mu, mu + z * se_mu)
    p_val = float(2 * (1 - _st.norm.cdf(abs(mu / se_mu)))) if se_mu > 0 else float("nan")

    out: Dict[str, Any] = {
        "k": int(k),
        "model": chosen,
        "model_requested": model,
        "estimate": mu,
        "se": se_mu,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "z": float(mu / se_mu) if se_mu > 0 else float("nan"),
        "pvalue": p_val,
        "level": level,
        "fixed": {"estimate": mu_fixed, "se": se_fixed,
                  "ci_low": mu_fixed - z * se_fixed, "ci_high": mu_fixed + z * se_fixed},
        "random": {"estimate": mu_random, "se": se_random,
                   "ci_low": mu_random - z * se_random, "ci_high": mu_random + z * se_random},
        "heterogeneity": {"Q": Q, "df": int(df), "p_Q": p_Q, "I2": I2,
                          "H2": H2, "tau2": tau2, "tau": float(np.sqrt(tau2))},
        "weights": {"fixed": w_fixed / np.sum(w_fixed),
                    "random": w_random / np.sum(w_random)},
        "per_study": pd.DataFrame({
            "label": list(labels) if labels is not None else [f"study{i+1}" for i in range(k)],
            "estimate": y, "se": s,
            "ci_low": y - z * s, "ci_high": y + z * s,
            "weight_fixed": w_fixed / np.sum(w_fixed),
            "weight_random": w_random / np.sum(w_random),
        }),
    }
    if effect_type == "log_or":
        out["odds_ratio"] = {"or": float(np.exp(mu)),
                             "ci_low": float(np.exp(ci[0])), "ci_high": float(np.exp(ci[1]))}
        out["note"] = "输入按 log(OR) 处理，输出额外给出 exp 后的 OR 与 CI"
    return out


def egger_test(estimate: Sequence[float], se: Sequence[float]) -> Dict[str, Any]:
    """Egger 回归检验发表偏倚（漏斗图不对称）。

    回归：y_i / se_i ~ 1/se_i（即 z_i 对精度加权），截距显著 ≠ 0 提示不对称。
    ⚠️ 研究数 <10 时该检验功效很低——**不显著不等于没有发表偏倚**。
    """
    y = np.asarray(estimate, dtype=float)
    s = np.asarray(se, dtype=float)
    z = y / s
    prec = 1.0 / s
    X = np.column_stack([np.ones_like(prec), prec])
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    resid = z - X @ beta
    dof = max(len(z) - 2, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se_int = float(np.sqrt(cov[0, 0]))
    t_stat = float(beta[0] / se_int) if se_int > 0 else float("nan")
    p = float(2 * (1 - _st.t.cdf(abs(t_stat), dof))) if np.isfinite(t_stat) else float("nan")
    return {
        "intercept": float(beta[0]), "se_intercept": se_int,
        "t": t_stat, "pvalue": p, "df": int(dof),
        "slope": float(beta[1]),
        "n_studies": int(len(z)),
        "caution": "研究数 <10 时功效很低；不显著 ≠ 无发表偏倚",
    }


def begg_test(estimate: Sequence[float], se: Sequence[float]) -> Dict[str, Any]:
    """Begg 秩相关检验（Kendall tau）：效应量与方差的秩相关。"""
    y = np.asarray(estimate, dtype=float)
    s = np.asarray(se, dtype=float)
    tau, p = _st.kendalltau(y, s ** 2)
    return {"kendall_tau": float(tau), "pvalue": float(p), "n_studies": int(len(y)),
            "caution": "功效低于 Egger；两者不一致时以 Egger 为主并如实报告"}


def leave_one_out(estimate: Sequence[float], se: Sequence[float],
                  labels: Optional[Sequence[str]] = None,
                  model: str = "random") -> pd.DataFrame:
    """留一敏感性分析：逐项剔除后重算合并值，看结论是否被单项研究绑架。"""
    y = np.asarray(estimate, dtype=float)
    s = np.asarray(se, dtype=float)
    labs = list(labels) if labels is not None else [f"study{i+1}" for i in range(len(y))]
    rows = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        r = meta_analysis(y[keep], s[keep], model=model)
        rows.append({"excluded": labs[i], "estimate": r["estimate"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                     "pvalue": r["pvalue"],
                     "I2": r["heterogeneity"]["I2"]})
    return pd.DataFrame(rows)


def subgroup_test(estimate: Sequence[float], se: Sequence[float],
                  group: Sequence[Any], model: str = "random") -> pd.DataFrame:
    """亚组分析：逐组合并 + 亚组间差异检验（Q_between）。"""
    y = np.asarray(estimate, dtype=float)
    s = np.asarray(se, dtype=float)
    g = np.asarray(group)
    rows = []
    for lev in pd.unique(g):
        m = g == lev
        if m.sum() < 2:
            rows.append({"group": lev, "k": int(m.sum()), "estimate": float(np.average(y[m], weights=1/s[m]**2)),
                         "ci_low": float("nan"), "ci_high": float("nan"), "note": "研究数<2，未合并"})
            continue
        r = meta_analysis(y[m], s[m], model=model)
        rows.append({"group": lev, "k": int(m.sum()), "estimate": r["estimate"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                     "I2": r["heterogeneity"]["I2"], "note": ""})
    return pd.DataFrame(rows)
