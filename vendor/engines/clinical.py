# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/engine/clinical.py
# 许可: 自建/内部
# 说明: 基线表 Table1 / 两组比较 / 风险表
# 源 sha256: dbf340700d5e5244
# VENDORED-FILE-END
"""临床统计：基线特征表（Table 1）、组间比较、相关性。

设计原则：不替用户选检验，而是把选择规则显式化并写进输出——
每一列都附带"用了什么检验、为什么"。审稿人能直接看到判据。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.stats import (adjust_p, hedges_g, interpret_d, normality_report,
                          odds_ratio, welch_t)


def _fmt_mean_sd(v: np.ndarray) -> str:
    return "%.2f ± %.2f" % (np.nanmean(v), np.nanstd(v, ddof=1))


def _fmt_median_iqr(v: np.ndarray) -> str:
    q1, q3 = np.nanpercentile(v, [25, 75])
    return "%.2f (%.2f–%.2f)" % (np.nanmedian(v), q1, q3)


def _fmt_n_pct(k: int, n: int) -> str:
    return "%d (%.1f%%)" % (k, 100.0 * k / n) if n else "—"


def table_one(metadata: pd.DataFrame,
              variables: Optional[Sequence[str]] = None,
              group_col: str = "group",
              categorical: Optional[Sequence[str]] = None,
              alpha: float = 0.05) -> pd.DataFrame:
    """生成基线特征表。

    连续变量：正态（Shapiro p > alpha 且 |偏度| < 1）用均值±标准差 + Welch t，
    否则用中位数（四分位间距） + Mann-Whitney U。
    分类变量：期望频数全部 ≥5 用卡方，否则用 Fisher 精确检验。

    返回表含 test 列，说明每个变量实际使用的检验。
    """
    categorical = set(categorical or [])
    if variables is None:
        variables = [c for c in metadata.columns if c not in (group_col, "sample_id")]

    groups = [str(g) for g in metadata[group_col].dropna().unique()]
    groups = sorted(groups)
    rows: List[Dict] = []

    for var in variables:
        if var not in metadata.columns:
            continue
        sub = metadata[[group_col, var]].copy()
        sub[group_col] = sub[group_col].astype(str)

        if var in categorical or not pd.api.types.is_numeric_dtype(sub[var]):
            # 分类
            tab = pd.crosstab(sub[var], sub[group_col])
            expected = stats.contingency.expected_freq(tab.values)
            if expected.min() >= 5:
                chi2, p, dof, _ = stats.chi2_contingency(tab.values)
                test = "卡方检验"
                stat = float(chi2)
            else:
                if tab.shape == (2, 2):
                    orr, p = stats.fisher_exact(tab.values)
                    test = "Fisher 精确检验"
                    stat = float(orr) if orr is not None else np.nan
                else:
                    chi2, p, dof, _ = stats.chi2_contingency(tab.values)
                    test = "卡方检验（期望频数<5，结果不稳）"
                    stat = float(chi2)
            row = {"variable": var, "type": "分类", "test": test,
                   "statistic": stat, "p": float(p)}
            counts = sub[var].value_counts()
            for g in groups:
                col = sub.loc[sub[group_col] == g, var]
                row["%s (n=%d)" % (g, len(col))] = (
                    "; ".join("%s: %s" % (k, _fmt_n_pct(int((col == k).sum()), len(col)))
                              for k in counts.index[:6]))
            row["总体 (n=%d)" % len(sub)] = "; ".join(
                "%s: %s" % (k, _fmt_n_pct(int((sub[var] == k).sum()), len(sub)))
                for k in counts.index[:6])
        else:
            # 连续
            vals = [sub.loc[sub[group_col] == g, var].dropna().values for g in groups]
            allv = sub[var].dropna().values
            norm = normality_report(allv)
            normal = (norm.get("p", 0) > alpha) and abs(norm.get("skew", 99)) < 1
            if normal and len(groups) == 2:
                r = welch_t(vals[0], vals[1])
                test, stat, p = "Welch t 检验", r["t"], r["p"]
                effect = "d = %.3f (%s)" % (r["d"], interpret_d(r["d"]))
                fmt = _fmt_mean_sd
            else:
                if len(groups) == 2:
                    u, p = stats.mannwhitneyu(vals[0], vals[1], alternative="two-sided")
                    test = "Mann-Whitney U"
                else:
                    h, p = stats.kruskal(*vals)
                    test = "Kruskal-Wallis H"
                    u = h
                stat = float(u)
                effect = "g = %.3f (%s)" % (hedges_g(vals[0], vals[-1]),
                                            interpret_d(hedges_g(vals[0], vals[-1])))
                fmt = _fmt_median_iqr
            row = {"variable": var, "type": "连续", "test": test,
                   "statistic": stat, "p": float(p), "effect": effect}
            for g, v in zip(groups, vals):
                row["%s (n=%d)" % (g, len(v))] = fmt(v) if len(v) else "—"
            row["总体 (n=%d)" % len(sub)] = fmt(allv) if allv.size else "—"

        rows.append(row)

    out = pd.DataFrame(rows)
    if "p" in out.columns and len(out):
        out["q"] = adjust_p(out["p"].values, method="bh")
    return out


def compare_two_groups(metadata: pd.DataFrame, value_col: str,
                       group_col: str = "group") -> Dict:
    """两组比较，返回检验、效应量与置信区间。"""
    sub = metadata[[group_col, value_col]].dropna()
    sub[group_col] = sub[group_col].astype(str)
    groups = sorted(sub[group_col].unique())
    if len(groups) != 2:
        raise ValueError("compare_two_groups 需要恰好 2 个分组，当前 %d 个" % len(groups))
    a = sub.loc[sub[group_col] == groups[0], value_col].values.astype(float)
    b = sub.loc[sub[group_col] == groups[1], value_col].values.astype(float)
    r = welch_t(a, b)
    lo, hi = stats.t.interval(
        0.95, len(a) + len(b) - 2,
        loc=np.mean(a) - np.mean(b),
        scale=np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    r.update({
        "group_a": groups[0], "group_b": groups[1],
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(a) - np.mean(b)),
        "ci_low": float(lo), "ci_high": float(hi),
        "normal_a": normality_report(a), "normal_b": normality_report(b),
    })
    return r


def risk_table(exposed_cases: int, exposed_controls: int,
               nonexposed_cases: int, nonexposed_controls: int) -> Dict[str, float]:
    """2x2 风险表：RR、OR、RD 与置信区间。"""
    a, b, c, d = exposed_cases, exposed_controls, nonexposed_cases, nonexposed_controls
    r1 = a / (a + b) if (a + b) else np.nan
    r2 = c / (c + d) if (c + d) else np.nan
    rr = r1 / r2 if r2 else np.nan
    se_log_rr = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)) if min(a, c) else np.nan
    rr_lo = float(np.exp(np.log(rr) - 1.96 * se_log_rr)) if rr and se_log_rr == se_log_rr else np.nan
    rr_hi = float(np.exp(np.log(rr) + 1.96 * se_log_rr)) if rr and se_log_rr == se_log_rr else np.nan
    orr, or_lo, or_hi = odds_ratio(a, b, c, d)
    return {"risk_exposed": r1, "risk_unexposed": r2,
            "risk_difference": r1 - r2,
            "relative_risk": rr, "rr_ci_low": rr_lo, "rr_ci_high": rr_hi,
            "odds_ratio": orr, "or_ci_low": or_lo, "or_ci_high": or_hi}
