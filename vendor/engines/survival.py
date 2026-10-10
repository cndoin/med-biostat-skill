# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/engine/survival.py
# 许可: 自建/内部
# 说明: 生存分析：KM / log-rank / Cox / PH 检验
# 源 sha256: f281f53952814030
# VENDORED-FILE-END
"""生存分析：KM 估计、log-rank 检验、Cox 比例风险、比例风险假定检验。

Cox 用 statsmodels 的 PHReg（Breslow 处理结）。
比例风险假定用基于 Schoenfeld 残差秩相关的简化检验。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def km_estimate(time: Sequence[float], event: Sequence[int]) -> pd.DataFrame:
    """Kaplan-Meier 乘积极限估计，含风险人数与逐点置信区间（Greenwood）。"""
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    order = np.argsort(t)
    t, e = t[order], e[order]
    n = t.size

    times: List[float] = [0.0]
    surv: List[float] = [1.0]
    se: List[float] = [0.0]
    at_risk: List[int] = [n]
    n_event: List[int] = [0]
    n_censor: List[int] = [0]

    s = 1.0
    var = 0.0
    alive = n
    i = 0
    while i < n:
        tt = t[i]
        # 同一时点的事件数与删失数
        block = (t == tt)
        d = int(np.sum(e[block] == 1))
        c = int(np.sum(e[block] == 0))
        if d > 0:
            var += d / (alive * max(alive - d, 1))
            s *= (1 - d / alive)
            times.append(float(tt))
            surv.append(float(s))
            se.append(float(np.sqrt(var) * s) if var > 0 else 0.0)
            at_risk.append(int(alive))
            n_event.append(d)
            n_censor.append(c)
        alive -= (d + c)
        i += int(block.sum())

    out = pd.DataFrame({
        "time": times, "survival": surv, "se": se,
        "at_risk": at_risk, "n_event": n_event, "n_censor": n_censor,
    })
    # Greenwood 的对数变换置信区间（保证落在 [0,1]）
    with np.errstate(divide="ignore", invalid="ignore"):
        log_s = np.log(np.clip(out["survival"].values, 1e-12, 1.0))
        se_log = np.where(out["survival"].values > 0,
                          out["se"].values / np.clip(out["survival"].values, 1e-12, None), np.nan)
    out["ci_low"] = np.exp(log_s - 1.96 * se_log)
    out["ci_high"] = np.exp(log_s + 1.96 * se_log)
    out.loc[out["time"] == 0, ["ci_low", "ci_high"]] = 1.0
    return out


def logrank(time_a: Sequence[float], event_a: Sequence[int],
            time_b: Sequence[float], event_b: Sequence[int]) -> Dict[str, float]:
    """两组 log-rank 检验。"""
    ta = np.asarray(time_a, float)
    ea = np.asarray(event_a, int)
    tb = np.asarray(time_b, float)
    eb = np.asarray(event_b, int)
    all_t = np.unique(np.concatenate([ta, tb]))
    o_minus_e = 0.0
    var = 0.0
    for tt in all_t:
        na = int(np.sum(ta >= tt))
        nb = int(np.sum(tb >= tt))
        n = na + nb
        da = int(np.sum((ta == tt) & (ea == 1)))
        db = int(np.sum((tb == tt) & (eb == 1)))
        d = da + db
        if n <= 1 or d == 0:
            continue
        o_minus_e += da - d * na / n
        var += d * (na / n) * (1 - na / n) * (n - d) / (n - 1)
    if var <= 0:
        return {"chi2": np.nan, "p": np.nan, "hr_approx": np.nan}
    chi2 = o_minus_e ** 2 / var
    p = float(stats.chi2.sf(chi2, 1))
    # 近似 HR（基于观测/期望事件数比）
    ea_exp = float(np.sum([(np.sum((np.concatenate([ta, tb]) == tt) & (np.concatenate([ea, eb]) == 1)))
                           * np.sum(ta >= tt) / max(np.sum(ta >= tt) + np.sum(tb >= tt), 1)
                           for tt in all_t]))
    hr = float(np.sum(ea == 1) / ea_exp) if ea_exp > 0 else np.nan
    return {"chi2": float(chi2), "p": p, "hr_approx": hr}


def cox_fit(metadata: pd.DataFrame, covariates: Sequence[str],
            time_col: str = "time", event_col: str = "event") -> pd.DataFrame:
    """Cox 比例风险回归（单变量逐个拟合，输出 HR 与 95% CI）。

    返回每个协变量一行：HR、CI、p、样本量。
    """
    from statsmodels.duration.hazard_regression import PHReg

    sub = metadata[[time_col, event_col] + list(covariates)].dropna()
    rows = []
    for cov in covariates:
        d = sub[[time_col, event_col, cov]].dropna()
        if len(d) < 10 or d[event_col].sum() < 5:
            rows.append({"covariate": cov, "hr": np.nan, "ci_low": np.nan,
                         "ci_high": np.nan, "p": np.nan, "n": len(d),
                         "events": int(d[event_col].sum()) if len(d) else 0,
                         "note": "样本量或事件数不足，未拟合"})
            continue
        X = d[[cov]].values.astype(float)
        if np.std(X) == 0:
            rows.append({"covariate": cov, "hr": np.nan, "ci_low": np.nan,
                         "ci_high": np.nan, "p": np.nan, "n": len(d),
                         "events": int(d[event_col].sum()), "note": "协变量无变异"})
            continue
        try:
            mod = PHReg(d[time_col].values.astype(float), X,
                        status=d[event_col].values.astype(int))
            res = mod.fit()
            beta = float(res.params[0])
            se = float(res.bse[0])
            hr = float(np.exp(beta))
            lo = float(np.exp(beta - 1.96 * se))
            hi = float(np.exp(beta + 1.96 * se))
            p = float(res.pvalues[0])
            note = ""
        except Exception as exc:  # 收敛失败等
            hr = lo = hi = p = np.nan
            note = "拟合失败：%s" % type(exc).__name__
        rows.append({"covariate": cov, "hr": hr, "ci_low": lo, "ci_high": hi,
                     "p": p, "n": len(d), "events": int(d[event_col].sum()), "note": note})
    out = pd.DataFrame(rows)
    if len(out):
        from ..core.stats import adjust_p
        out["q"] = adjust_p(out["p"].values, method="bh")
    return out


def proportional_hazards_check(metadata: pd.DataFrame, covariate: str,
                               time_col: str = "time",
                               event_col: str = "event") -> Dict[str, float]:
    """比例风险假定检验（简化 Schoenfeld 残差秩相关）。

    返回秩相关系数与 p；p < 0.05 提示该协变量违反比例风险假定，
    需要考虑时间交互项或改用限制性平均生存时间（RMST）。
    """
    from statsmodels.duration.hazard_regression import PHReg

    d = metadata[[time_col, event_col, covariate]].dropna()
    if len(d) < 20 or d[event_col].sum() < 5:
        return {"rho": np.nan, "p": np.nan, "n": len(d), "verdict": "事件数不足，无法检验"}
    X = d[[covariate]].values.astype(float)
    mod = PHReg(d[time_col].values.astype(float), X, status=d[event_col].values.astype(int))
    res = mod.fit()
    sch = np.asarray(res.schoenfeld_residuals, dtype=float).ravel()
    ev = d[event_col].values.astype(int) == 1
    sch = sch[ev]
    tt = d[time_col].values.astype(float)[ev]
    if sch.size < 5:
        return {"rho": np.nan, "p": np.nan, "n": len(d), "verdict": "事件数不足"}
    rho, p = stats.spearmanr(tt, sch)
    return {"rho": float(rho), "p": float(p), "n": int(len(d)),
            "verdict": "未发现违反" if p > 0.05 else "可能违反比例风险假定"}
