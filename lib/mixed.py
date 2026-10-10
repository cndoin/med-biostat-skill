# SKILL-OWNED —— 本 skill 自有实现（可直接编辑）
"""mixed.py —— 线性混合效应模型（认知斜率等重复测量）

补齐 LPDC 第 9.1 节的**主要分析**：
    MoCA ~ PDR × time + 协变量 + (1 + time | subject)
报告 **暴露×时间交互项**的估计值与 95% CI（斜率差异）。

依赖：numpy + pandas + statsmodels（Tier 1，惰性 import）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def mixed_slope(df: pd.DataFrame, outcome: str, time_col: str, subject_col: str,
                exposure: str, covariates: Optional[Sequence[str]] = None,
                random_slope: bool = True) -> Dict:
    """拟合 `outcome ~ exposure*time + covariates`，随机截距(+斜率)按 subject。

    返回 dict，含交互项（暴露×时间）的 beta / se / p / 95% CI，以及拟合摘要。
    """
    covariates = list(covariates or [])
    need = [outcome, time_col, subject_col, exposure] + covariates
    missing = [c for c in need if c not in df.columns]
    if missing:
        return {"error": f"缺少列：{missing}"}

    # 依赖守卫必须包住 import 本身：statsmodels 缺失时应返回 error，而不是抛 ModuleNotFoundError
    try:
        import statsmodels.formula.api as smf
    except ImportError as e:  # noqa: BLE001
        return {"error": f"缺少 statsmodels（{e}）；安装：pip install statsmodels"}

    d = df[need].dropna()
    if d[subject_col].nunique() < 3:
        return {"error": f"受试者数不足（{d[subject_col].nunique()} < 3），无法估计随机效应"}

    rhs = " + ".join([f"{exposure} * {time_col}"] + covariates)
    formula = f"{outcome} ~ {rhs}"
    re_formula = f"~{time_col}" if random_slope else "~1"

    try:
        model = smf.mixedlm(formula, d, groups=d[subject_col], re_formula=re_formula)
        fit = model.fit(reml=True, method="lbfgs")
    except Exception as e:  # noqa: BLE001
        return {"error": f"拟合失败：{type(e).__name__}: {e}"}

    inter = f"{exposure}:{time_col}"
    # statsmodels 的交互项命名可能是 a:b 或 b:a，两种都找
    alt = f"{time_col}:{exposure}"
    key = inter if inter in fit.params.index else (alt if alt in fit.params.index else None)
    if key is None:
        return {"error": f"未找到交互项 {inter}；参数={list(fit.params.index)}"}

    b = float(fit.params[key])
    se = float(fit.bse[key])
    ci = fit.conf_int().loc[key]
    out = {
        "formula": formula,
        "re_formula": re_formula,
        "interaction_name": key,
        "interaction_beta": b,
        "interaction_se": se,
        "interaction_p": float(fit.pvalues[key]),
        "interaction_ci_low": float(ci[0]),
        "interaction_ci_high": float(ci[1]),
        "n_obs": int(fit.nobs),
        "n_subjects": int(d[subject_col].nunique()),
        "converged": bool(fit.converged),
        "interpretation": ("暴露×时间交互项显著，提示暴露随时间的斜率存在组间差异"
                           if float(fit.pvalues[key]) < 0.05 else
                           "交互项不显著：未检出暴露对变化斜率的证据"),
    }
    # 各固定效应的估计（便于审计）
    out["fixed_effects"] = {k: float(v) for k, v in fit.params.items()}
    return out
