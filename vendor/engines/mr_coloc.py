# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/engine/causal.py
# 许可: 自建/内部
# 说明: 孟德尔随机化 + 简化共定位
# 源 sha256: 2981f0f125fb1246
# VENDORED-FILE-END
"""孟德尔随机化与共定位。

提供药物靶点 MR 的标准工具链：IVW、加权中位数、MR-Egger（含截距检验）、
留一法、异质性（Cochran Q 与 I²）、以及简化共定位。

一个刻意的设计：所有方法一起报告，不做"挑好看的报"。
各法方向一致才支持因果；Egger 截距显著或 Q 显著时必须明确标注为不确定。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _ivw(bx: np.ndarray, by: np.ndarray, se_by: np.ndarray) -> Dict[str, float]:
    w = 1.0 / np.maximum(se_by ** 2, 1e-18)
    beta = float(np.sum(w * bx * by) / np.sum(w * bx ** 2))
    se = float(np.sqrt(1.0 / np.sum(w * bx ** 2)))
    return {"beta": beta, "se": se,
            "p": float(2 * stats.norm.sf(abs(beta / se))) if se > 0 else np.nan,
            "method": "IVW"}


def _egger(bx: np.ndarray, by: np.ndarray, se_by: np.ndarray) -> Dict[str, float]:
    """MR-Egger：允许截距非零，截距显著提示定向多效性。"""
    n = len(bx)
    if n < 3:
        return {"beta": np.nan, "se": np.nan, "p": np.nan,
                "intercept": np.nan, "intercept_p": np.nan, "method": "MR-Egger"}
    X = np.column_stack([np.ones(n), bx])
    W = np.diag(1.0 / np.maximum(se_by ** 2, 1e-18))
    XtWX = X.T @ W @ X
    try:
        cov = np.linalg.pinv(XtWX)
        coef = cov @ (X.T @ W @ by)
    except Exception:
        return {"beta": np.nan, "se": np.nan, "p": np.nan,
                "intercept": np.nan, "intercept_p": np.nan, "method": "MR-Egger"}
    b0, b1 = float(coef[0]), float(coef[1])
    se1 = float(np.sqrt(max(cov[1, 1], 1e-18)))
    se0 = float(np.sqrt(max(cov[0, 0], 1e-18)))
    return {"beta": b1, "se": se1,
            "p": float(2 * stats.norm.sf(abs(b1 / se1))) if se1 > 0 else np.nan,
            "intercept": b0,
            "intercept_p": float(2 * stats.norm.sf(abs(b0 / se0))) if se0 > 0 else np.nan,
            "method": "MR-Egger"}


def _weighted_median(bx: np.ndarray, by: np.ndarray, se_by: np.ndarray) -> Dict[str, float]:
    """加权中位数：单个无效工具不超过半数时稳健。"""
    order = np.argsort(by / np.maximum(bx, 1e-18))
    bx, by, se = bx[order], by[order], se_by[order]
    ratios = by / np.where(bx == 0, np.nan, bx)
    w = 1.0 / np.maximum(se ** 2, 1e-18)
    w = w / w.sum()
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, 0.5))
    idx = min(idx, len(ratios) - 1)
    beta = float(ratios[idx])
    # bootstrap 标准误
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(500):
        s = rng.integers(0, len(ratios), len(ratios))
        rr = ratios[s]
        ww = w[s] / w[s].sum()
        o = np.argsort(rr)
        cc = np.cumsum(ww[o])
        draws.append(rr[o][min(int(np.searchsorted(cc, 0.5)), len(rr) - 1)])
    se = float(np.nanstd(draws, ddof=1)) if draws else np.nan
    return {"beta": beta, "se": se,
            "p": float(2 * stats.norm.sf(abs(beta / se))) if se and se > 0 else np.nan,
            "method": "加权中位数"}


def mr(instruments: pd.DataFrame,
       beta_exposure_col: str = "beta_exposure",
       beta_outcome_col: str = "beta_outcome",
       se_outcome_col: str = "se_outcome") -> pd.DataFrame:
    """主分析：三种方法并列 + 异质性 + 留一法敏感性。全部返回，不做挑选。"""
    bx = instruments[beta_exposure_col].values.astype(float)
    by = instruments[beta_outcome_col].values.astype(float)
    se = instruments[se_outcome_col].values.astype(float)
    ok = ~(np.isnan(bx) | np.isnan(by) | np.isnan(se)) & (np.abs(bx) > 0)
    bx, by, se = bx[ok], by[ok], se[ok]

    rows = []
    if len(bx) == 0:
        return pd.DataFrame([{"method": "无可用工具", "beta": np.nan}])

    rows.append(_ivw(bx, by, se))
    if len(bx) >= 3:
        rows.append(_egger(bx, by, se))
    if len(bx) >= 3:
        rows.append(_weighted_median(bx, by, se))
    out = pd.DataFrame(rows)

    # 异质性
    ivw_beta = float(out.loc[out["method"] == "IVW", "beta"].iloc[0])
    resid = (by - ivw_beta * bx) / np.maximum(se, 1e-12)
    q = float(np.sum(resid ** 2))
    dfree = max(len(bx) - 1, 1)
    q_p = float(stats.chi2.sf(q, dfree))
    i2 = float(max(0.0, (q - dfree) / q)) if q > 0 else 0.0

    out["n_instruments"] = len(bx)
    out["cochran_q"] = q
    out["q_p"] = q_p
    out["i2"] = i2
    out["or"] = np.exp(out["beta"])
    out["or_ci_low"] = np.exp(out["beta"] - 1.96 * out["se"])
    out["or_ci_high"] = np.exp(out["beta"] + 1.96 * out["se"])

    eg = out[out["method"] == "MR-Egger"]
    consistent = True
    if len(eg) and not np.isnan(eg["intercept_p"].iloc[0]):
        consistent = eg["intercept_p"].iloc[0] > 0.05
    out["verdict"] = np.where(
        ~consistent, "Egger 截距显著，存在定向多效性，结论不确定",
        np.where(q_p < 0.05, "异质性显著（Q p<0.05），需检查离群工具", "方向一致，无显著异质性"))
    return out


def leave_one_out(instruments: pd.DataFrame,
                  beta_exposure_col: str = "beta_exposure",
                  beta_outcome_col: str = "beta_outcome",
                  se_outcome_col: str = "se_outcome") -> pd.DataFrame:
    """留一法：逐个剔除工具后重估，检验是否由单一 SNP 驱动。"""
    rows = []
    for i in range(len(instruments)):
        sub = instruments.drop(instruments.index[i])
        r = mr(sub, beta_exposure_col, beta_outcome_col, se_outcome_col)
        r = r[r["method"] == "IVW"]
        if len(r):
            rows.append({"dropped": i, "beta": float(r["beta"].iloc[0]),
                         "or": float(r["or"].iloc[0]),
                         "p": float(r["p"].iloc[0])})
    return pd.DataFrame(rows)


def coloc_approx(p1: np.ndarray, p2: np.ndarray,
                 prior_1: float = 1e-4, prior_2: float = 1e-4,
                 prior_12: float = 1e-5) -> Dict[str, float]:
    """简化共定位（单因果假设，近似 Bayes 因子）。

    完整实现需要区域的连锁不平衡矩阵与效应量；此处给出量级判断，
    正式发表应使用 coloc.susie() 并提供 LD 矩阵。
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    m = max(len(p1), len(p2), 1)
    # 近似：区域内最小 p 值对应的 z 与 Bayes 因子量级
    def _bf(p):
        p = np.clip(np.nanmin(p) if p.size else 1.0, 1e-300, 1.0)
        z = stats.norm.isf(p / 2)
        return float(np.exp(z ** 2 / 2) / np.sqrt(max(np.log(m), 2.0)))

    bf1, bf2 = _bf(p1), _bf(p2)
    # H4  posterior（四假设近似）
    h = {
        "H0": (1 - prior_1) * (1 - prior_2),
        "H1": prior_1 * (1 - prior_2) * bf1,
        "H2": (1 - prior_1) * prior_2 * bf2,
        "H3": prior_1 * prior_2 * bf1,
        "H4": prior_12 * bf1 * bf2,
    }
    tot = sum(h.values())
    post = {k: v / tot for k, v in h.items()}
    return {"PP.H0": post["H0"], "PP.H1": post["H1"], "PP.H2": post["H2"],
            "PP.H3": post["H3"], "PP.H4": post["H4"],
            "verdict": ("强共定位" if post["H4"] >= 0.8 else
                        "中等证据" if post["H4"] >= 0.5 else
                        "无共定位（信号可能由不同变异驱动）")}
