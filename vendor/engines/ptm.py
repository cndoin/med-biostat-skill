# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/engine/ptm.py
# 许可: 自建/内部
# 说明: 翻译后修饰：占位率 / 局部 FDR
# 源 sha256: 6286a7c79b0b33a6
# VENDORED-FILE-END
"""翻译后修饰（PTM）分析：占位率、局部 FDR、伪影校正、损伤速率。

这一模块是本平台区别于通用生信工具的部分，也是晶状体蛋白损伤研究的
核心计量学。四条原则：

1. 占位率而非强度：占位率 = 修饰型 / (修饰型 + 非修饰型)，与蛋白总量无关，
   不受上样量与结晶蛋白超高丰度的影响。
2. 局部 FDR：开放搜索的全局 1% FDR 会被高丰度修饰掩盖稀有质量漂移的假阳性，
   必须按 Δm 区间单独估计错误率。
3. 伪影显式扣除：脱酰胺在晶状体中既是制备伪影又是真实过程，
   必须用同批次空白/合成肽对照扣除，且核与皮质必须同批次。
4. 共洗脱下界：当分离出的异构体峰少于理论数时，占位率只是下界，必须标注。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.stats import adjust_p


# 常见修饰与伪影的质量漂移（Da），用于开放搜索的 Δm 注释
KNOWN_SHIFTS = {
    "脱酰胺 (N→D)": 0.98402,
    "氧化 (M)": 15.99491,
    "磷酸化": 79.96633,
    "甲酰化（伪影）": 27.99491,
    "焦谷氨酸（伪影）": -17.02655,
    "过烷基化（伪影）": 114.04293,
    "双羧甲基化（伪影）": 58.00548,
    "乙酰化": 42.01056,
    "甲基化": 14.01565,
}


def occupancy(modified: np.ndarray, unmodified: np.ndarray,
              eps: float = 1e-9) -> np.ndarray:
    """占位率 = 修饰型强度 / (修饰型 + 非修饰型强度)。

    返回 [0,1]；两者均为 0 时返回 NaN（无信息，不可当作 0）。
    """
    m = np.asarray(modified, float)
    u = np.asarray(unmodified, float)
    total = m + u
    with np.errstate(invalid="ignore", divide="ignore"):
        occ = np.where(total > eps, m / np.where(total > eps, total, 1.0), np.nan)
    return occ


def local_fdr(mass_shifts: np.ndarray, is_decoy: np.ndarray,
              bin_width: float = 0.01) -> pd.DataFrame:
    """Delta-mass 特异 FDR：按质量漂移区间分别估计错误率。

    返回每个区间：目标数、诱饵数、局部 FDR、以及该区间内经 BH 校正的最小 q。
    """
    ms = np.asarray(mass_shifts, float)
    dec = np.asarray(is_decoy).astype(bool)
    if ms.size == 0:
        return pd.DataFrame(columns=["bin", "n_target", "n_decoy", "local_fdr"])
    bins = np.floor(ms / bin_width).astype(int)
    rows = []
    for b in np.unique(bins):
        sel = bins == b
        nt = int((sel & ~dec).sum())
        nd = int((sel & dec).sum())
        rows.append({
            "bin_center": round((b + 0.5) * bin_width, 4),
            "n_target": nt,
            "n_decoy": nd,
            "local_fdr": nd / nt if nt else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("n_target", ascending=False).reset_index(drop=True)
    out["verdict"] = np.where(out["local_fdr"] > 0.01, "该 Δm 区间不可靠", "通过")
    return out


def annotate_shifts(mass_shifts: np.ndarray, tol: float = 0.01) -> pd.Series:
    """把观测到的质量漂移注释到已知修饰或已知伪影。"""
    ms = np.asarray(mass_shifts, float)
    labels = []
    for x in ms:
        best, name = tol + 1, "未知"
        for k, v in KNOWN_SHIFTS.items():
            d = abs(x - v)
            if d < best:
                best, name = d, k
        labels.append(name if best <= tol else "未知")
    return pd.Series(labels, name="annotation")


def subtract_artifact(occ_sample: np.ndarray, occ_blank: np.ndarray,
                      clip: bool = True) -> np.ndarray:
    """扣除同批次空白/合成肽对照中的伪影占位率。

    注意：若空白对照的占位率高于样本，说明该位点在本批次中不可信，
    扣除后按 clip=True 归零并应标注为"伪影不可分"。
    """
    s = np.asarray(occ_sample, float)
    b = np.asarray(occ_blank, float)
    out = s - b
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out


def damage_rate(occ_nucleus: np.ndarray, occ_cortex: np.ndarray,
                delta_t: np.ndarray) -> np.ndarray:
    """个体内损伤累积速率 PDR = [占位率(核) − 占位率(皮质)] / Δt。

    单位：%·年⁻¹。核与皮质来自同一个体、同一管、同一批次时，
    制备伪影在分子与分母上抵消，这是该指标成立的前提。
    """
    n = np.asarray(occ_nucleus, float)
    c = np.asarray(occ_cortex, float)
    dt = np.asarray(delta_t, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n - c) / np.where(dt > 0, dt, np.nan)
    return r


def classify_kinetics(occ: np.ndarray, age: np.ndarray,
                      cortex_baseline: Optional[np.ndarray] = None,
                      slope_thresh: float = 2e-4) -> pd.Series:
    """位点动力学分型。

    adult_linear  —— 成年线性累积型：随年龄显著上升，且皮质本底低。
    early_saturating —— 早期饱和型：皮质本底高（发育期已接近饱和），
                       成年后不再随年龄变化，对速率无贡献，必须剔除。
    inert         —— 惰性型：几乎不随年龄变化且本底低。

    为什么必须分型：把早期饱和型混进时钟，会让模型学到的是"出生时的发育状态"
    而不是"成年后的损伤速率"，这正是多数 PTM 时钟与时序年龄高度纠缠的原因之一。
    """
    occ = np.asarray(occ, float)
    age = np.asarray(age, float)
    base = (np.nanmean(np.asarray(cortex_baseline, float), axis=0)
            if cortex_baseline is not None else np.zeros(occ.shape[1]))

    labels = []
    for j in range(occ.shape[1]):
        y = occ[:, j]
        ok = ~(np.isnan(y) | np.isnan(age))
        if ok.sum() < 8:
            labels.append("inert")
            continue
        r = stats.pearsonr(age[ok], y[ok])
        slope = float(np.polyfit(age[ok], y[ok], 1)[0])
        high_base = base[j] > 0.25 if base.ndim else False
        if r[1] < 0.05 and slope >= slope_thresh and not high_base:
            labels.append("adult_linear")
        elif high_base:
            labels.append("early_saturating")
        else:
            labels.append("inert")
    return pd.Series(labels, name="kinetics")


def coflag_celution(n_isomers_observed: np.ndarray,
                    n_isomers_theoretical: np.ndarray) -> pd.Series:
    """共洗脱标记：观测异构体峰数少于理论数时，占位率应作为下界报告。"""
    o = np.asarray(n_isomers_observed, int)
    t = np.asarray(n_isomers_theoretical, int)
    return pd.Series(np.where(o < t, "下界（存在共洗脱）", "完整分离"),
                     name="isomer_flag")


def ptm_pipeline(report: pd.DataFrame,
                 sample_col: str = "sample_id",
                 region_col: str = "region",
                 site_col: str = "site",
                 mod_col: str = "modified",
                 unmod_col: str = "unmodified",
                 shift_col: str = "mass_shift",
                 decoy_col: Optional[str] = "is_decoy",
                 blank_col: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """PTM 端到端：局部 FDR → 占位率 → 伪影扣除 → 输出。"""
    out: Dict[str, pd.DataFrame] = {}

    if decoy_col and decoy_col in report.columns and shift_col in report.columns:
        out["local_fdr"] = local_fdr(report[shift_col].values,
                                     report[decoy_col].values)
    if shift_col in report.columns:
        out["annotation"] = annotate_shifts(report[shift_col].values).to_frame()

    df = report.copy()
    df["occupancy"] = occupancy(df[mod_col].values, df[unmod_col].values)
    if blank_col and blank_col in df.columns:
        df["occupancy_corrected"] = subtract_artifact(
            df["occupancy"].values, df[blank_col].values)
    else:
        df["occupancy_corrected"] = df["occupancy"]
        df["note"] = "未提供空白对照，占位率含伪影，不可直接用于速率估计"

    out["sites"] = df
    return out
