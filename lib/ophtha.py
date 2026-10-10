# -*- coding: utf-8 -*-
"""ophtha.py —— 眼科 / 白内障生信与临床计算

**定位**：包装层。光学与屈光的核心数学来自 vendored 开源项目
**PEARL-DGS**（`vendor/_ophtha/pearldgs.py`，MIT，来源 gdebel/pearldgs_toolbox），
本模块只做两件事：① 把物理单位（米）换成临床单位（mm / D）；
② 补上眼科临床与流行病学的标准指标（这些是公开定义的标准量，不是自造算法）。

**红线**：这里没有任何一步依赖模型判断。全部是给定输入 → 确定性数值。

**经典 IOL 度数计算（第三代薄透镜公式）**：SRK/T · Hoffer Q · Holladay 1 · Haigis
按**公开发表的原始论文**独立实现，见文件末尾那一节的来源与验证级别说明。
⚠️ 这四个公式**不是**从任何开源仓库复制的——可搜到的实现要么是 AGPL-3.0，
要么没有 LICENSE，**没有一个许可干净到可以内嵌**。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

from vendor._ophtha import pearldgs as _pg

__all__ = [
    "REFRACTIVE_INDEX", "tilp_from_refraction", "refraction_from_tilp",
    "corneal_power", "predicted_posterior_radius", "segmented_axial_length",
    "logmar_from_decimal", "decimal_from_logmar", "logmar_from_snellen",
    "refractive_outcome", "iop_stats", "CATARACT_GENE_SETS",
    "lens_gene_overlap",
    # 经典 IOL 度数计算（第三代薄透镜公式）
    "K_from_r", "r_from_K",
    "srkt_refraction", "srkt_power",
    "hofferq_refraction", "hofferq_power",
    "holladay1_refraction", "holladay1_power",
    "haigis_refraction", "haigis_power", "haigis_solve_d",
    "acd_const_from_a_constant", "a_constant_from_acd_const",
    "haigis_a0_from_acd_const", "iol_power_comparison",
]

# Gullstrand 简化眼的标准折射率（PEARL-DGS 使用的取值）
REFRACTIVE_INDEX = {
    "cornea": 1.376, "iol": 1.46, "vitreous": 1.336,
    "air": 1.0, "aqueous": 1.336,
}

_M2MM = 1000.0


# ── 一、光学核心（委托 PEARL-DGS）────────────────────────────────────
def tilp_from_refraction(se: float, al_mm: float, rco1_mm: float,
                         rco2_mm: float, eco_mm: float = 0.55,
                         riol1_mm: float = 10.8, riol2_mm: float = -10.8,
                         iol_t_mm: float = 1.0, vertex_mm: float = 12.0,
                         index: Dict[str, float] | None = None) -> float:
    """由术后等效球镜反解 IOL 理论位置（TILP，单位 mm）。

    TILP = 角膜后表面到 IOL 前表面的距离，厚透镜公式能精确复现该术后屈光的距离。
    这是 IOL 计算公式（PEARL-DGS / Barrett 等）的**共同中间量**。

    参数全部用临床单位（mm、D）；内部转成米后调用开源实现。
    """
    n = dict(REFRACTIVE_INDEX if index is None else index)
    out = _pg.calcTILP(
        n["cornea"], n["iol"], n["vitreous"], n["air"], n["aqueous"],
        rco1_mm / _M2MM, rco2_mm / _M2MM, eco_mm / _M2MM,
        riol1_mm / _M2MM, riol2_mm / _M2MM, iol_t_mm / _M2MM,
        float(se), al_mm / _M2MM, vertex_mm / _M2MM,
    )
    return float(out) * _M2MM


def refraction_from_tilp(tilp_mm: float, al_mm: float, rco1_mm: float,
                         rco2_mm: float, eco_mm: float = 0.55,
                         riol1_mm: float = 10.8, riol2_mm: float = -10.8,
                         iol_t_mm: float = 1.0, vertex_mm: float = 12.0,
                         index: Dict[str, float] | None = None) -> float:
    """由 IOL 位置正算术后等效球镜（D，眼镜平面）。`tilp_from_refraction` 的逆运算。"""
    n = dict(REFRACTIVE_INDEX if index is None else index)
    out = _pg.calcSE(
        n["cornea"], n["iol"], n["vitreous"], n["air"], n["aqueous"],
        rco1_mm / _M2MM, rco2_mm / _M2MM, eco_mm / _M2MM,
        riol1_mm / _M2MM, riol2_mm / _M2MM, iol_t_mm / _M2MM,
        tilp_mm / _M2MM, al_mm / _M2MM, vertex_mm / _M2MM,
    )
    return float(out)


def corneal_power(r1_mm: float, r2_mm: float) -> float:
    """平均角膜曲率（D）：由前后表面曲率半径（mm）算平均半径。"""
    return float(_pg.calcARC(r1_mm, r2_mm))


def predicted_posterior_radius(arc: float) -> float:
    """由前角膜曲率（D）预测后角膜曲率半径（mm）——PEARL-DGS 的 PRC 预测式。"""
    return float(_pg.predPRC(arc))


def segmented_axial_length(al_mm: float, lt_mm: float) -> float:
    """分段眼轴（考虑晶状体厚度的等效长度）。"""
    return float(_pg.calculateSegmentedAL(al_mm, lt_mm))


# ── 二、视力换算（国际标准定义，非自造）──────────────────────────────
def logmar_from_decimal(decimal: float) -> float:
    """小数视力 → logMAR：logMAR = -log10(decimal)。0.0 表示非法（对数无定义）。"""
    d = float(decimal)
    if d <= 0:
        raise ValueError(f"小数视力必须 > 0，得到 {d}")
    return float(-np.log10(d))


def decimal_from_logmar(logmar: float) -> float:
    """logMAR → 小数视力：decimal = 10^(-logMAR)。"""
    return float(10.0 ** (-float(logmar)))


def logmar_from_snellen(numerator: float, denominator: float) -> float:
    """Snellen 分数（如 6/12、20/40）→ logMAR。"""
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Snellen 分子分母都必须 > 0")
    return logmar_from_decimal(float(numerator) / float(denominator))


# ── 三、白内障手术屈光结局（标准报告指标）────────────────────────────
def refractive_outcome(predicted: Sequence[float],
                       actual: Sequence[float]) -> Dict[str, Any]:
    """屈光预测误差的标准评价集。

    PE = 实际等效球镜 − 预测等效球镜（**符号约定必须输出**，否则无法解读方向）。
    指标：ME（均值误差，含 ±1.96SD 的 LoA 用区间）、MAE、MedAE、SD、
    以及落在 ±0.25 / ±0.5 / ±1.0 D 内的比例（JCRS 标准报告项）。
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    if p.shape != a.shape:
        raise ValueError(f"predicted/actual 长度不一致：{p.shape} vs {a.shape}")
    if p.size == 0:
        raise ValueError("至少需要 1 例")

    pe = a - p
    abs_pe = np.abs(pe)
    n = int(pe.size)
    return {
        "n": n,
        "sign_convention": "PE = actual - predicted（正=术后比预测更偏近视/欠矫）",
        "ME": float(np.mean(pe)),
        "SD": float(np.std(pe, ddof=1)) if n > 1 else 0.0,
        "MAE": float(np.mean(abs_pe)),
        "MedAE": float(np.median(abs_pe)),
        "min": float(np.min(pe)),
        "max": float(np.max(pe)),
        "within_0.25D": float(np.mean(abs_pe <= 0.25)),
        "within_0.50D": float(np.mean(abs_pe <= 0.50)),
        "within_1.00D": float(np.mean(abs_pe <= 1.00)),
        "loa_low": float(np.mean(pe) - 1.96 * (np.std(pe, ddof=1) if n > 1 else 0.0)),
        "loa_high": float(np.mean(pe) + 1.96 * (np.std(pe, ddof=1) if n > 1 else 0.0)),
    }


def iop_stats(iop: Sequence[float]) -> Dict[str, Any]:
    """眼压（mmHg）的描述统计 + 眼科常用分层（≤21 mmHg 为常规上限）。"""
    x = np.asarray(iop, dtype=float)
    if x.size == 0:
        raise ValueError("空的眼压序列")
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "above_21": int(np.sum(x > 21)),
        "above_21_pct": float(np.mean(x > 21)),
    }


# ── 四、白内障 / 晶状体基因集（用于富集与候选基因筛选）────────────────
# 来源：公开发表的白内障遗传学研究（Choquet 2021 Nat Commun 的多民族 GWAS、
# iSyTE 晶状体表达数据库、先天性白内障基因 panel）。此处仅为**基因列表常量**，
# 不含任何判断逻辑；富集检验本身由 vendor/_enrich/stats.py 的超几何完成。
CATARACT_GENE_SETS: Dict[str, tuple] = {
    "crystallin_alpha": ("CRYAA", "CRYAB"),
    "crystallin_beta": ("CRYBA1", "CRYBA2", "CRYBA4", "CRYBB1", "CRYBB2", "CRYBB3"),
    "crystallin_gamma": ("CRYGA", "CRYGB", "CRYGC", "CRYGD", "CRYGS"),
    "lens_membrane_channel": ("MIP", "GJA3", "GJA8", "LIM2", "TMEM114"),
    "lens_development_tf": ("PAX6", "SOX1", "SOX2", "PITX3", "FOXE3", "MAF", "HSF4", "PROX1"),
    "age_related_cataract_gwas": (
        "EPHA2", "CRYAA", "CRYAB", "CRYBB1", "CRYBB2", "CRYGC", "CRYGD", "CRYGS",
        "GJA8", "MIP", "LIM2", "HSF4", "MAF", "PITX3", "FOXE3", "PAX6",
        "CASP7", "FTL", "TFAP2A", "TDRD7", "SLC16A12", "CHMP4B", "BFSP1", "BFSP2",
        "VIM", "CPAMD8", "ADAMTSL4", "COL4A1", "SIL1", "GCNT2", "ABCA1", "APOE",
    ),
    "congenital_cataract_panel": (
        "CRYAA", "CRYAB", "CRYBA1", "CRYBB1", "CRYBB2", "CRYBB3", "CRYGC", "CRYGD",
        "CRYGS", "GJA3", "GJA8", "MIP", "LIM2", "BFSP1", "BFSP2", "HSF4", "MAF",
        "PITX3", "FOXE3", "PAX6", "VIM", "TDRD7", "SIL1", "CHMP4B", "FYCO1",
        "EPHA2", "NHS", "BCOR", "OCRL", "GCNT2", "AGK", "TMEM114",
    ),
    "lens_epithelium_uv_damage": (
        "CRYAA", "CRYAB", "HSPB1", "HSPA1A", "HSPA1B", "DNAJB1", "HMOX1",
        "SOD1", "SOD2", "CAT", "GPX1", "NRF2", "NFE2L2", "KEAP1", "TP53",
    ),
}


def lens_gene_overlap(genes: Sequence[str], set_name: str = "age_related_cataract_gwas"):
    """给定基因列表与某个眼科基因集的交集。

    只做集合运算；**是否富集**要交给 `vendor/_enrich/stats.py::calc_pvalues`
    做超几何检验——直接数交集个数会随基因集大小系统性偏大，那是常见错误。
    """
    if set_name not in CATARACT_GENE_SETS:
        raise KeyError(f"未知基因集 {set_name!r}；可用：{sorted(CATARACT_GENE_SETS)}")
    ref = set(CATARACT_GENE_SETS[set_name])
    query = {str(g).upper() for g in genes}
    hit = sorted(query & ref)
    return {
        "set": set_name,
        "set_size": len(ref),
        "query_size": len(query),
        "overlap": hit,
        "n_overlap": len(hit),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 经典 IOL 度数计算（第三代薄透镜公式）：SRK/T · Hoffer Q · Holladay 1 · Haigis
# ══════════════════════════════════════════════════════════════════════════════
# **来源（公开发表的原始论文，逐条引用）**
#   · SRK/T       Retzlaff JA, Sanders DR, Kraff MC. Development of the SRK/T
#                 intraocular lens implant power calculation formula.
#                 J Cataract Refract Surg. 1990;16(3):333-40
#                 （勘误：J Cataract Refract Surg. 1990;16(4):528）
#   · Hoffer Q    Hoffer KJ. The Hoffer Q formula: a comparison of theoretic and
#                 regression formulas. J Cataract Refract Surg. 1993;19(6):700-12
#                 （勘误：Zuberbuhler B, Morrell AJ. JCRS. 2007;33(1):2）
#   · Holladay 1  Holladay JT, Musgrove KH, Prager TC, et al. A three-part system
#                 for refining intraocular lens power calculations.
#                 J Cataract Refract Surg. 1988;14(1):17-24
#   · Haigis      Haigis W, Lege B, Miller N, Schneider B. Comparison of immersion
#                 ultrasound biometry and partial coherence interferometry for
#                 intraocular lens calculation according to Haigis.
#                 Graefes Arch Clin Exp Ophthalmol. 2000;238(9):765-73
#
# **常数与函数结构的核对来源（公开技术说明）**
#   · Debellemanière G, et al. "The (not so) secret mechanisms behind classical
#     thin lens IOL formulas". IOL Solver, 2022-10-23.
#     https://news.iolsolver.com/the-not-so-secret-mechanisms-of-cla/
#   · "How to easily perform the Haigis triple optimization". IOL Solver.
#     https://news.iolsolver.com/how-to-easily-perform-the-haigis-triple-optimization/
#   · Haigis W. "The Haigis formula". In: Shammas HJ (ed). Intraocular Lens Power
#     Calculation. Slack Inc; 2004.（另见 http://ocusoft.de/serv/ioltxt/haie.htm）
#
# ⚠️ **为什么这一段不是「照抄开源项目」**：2026-10-11 实测遍历了能搜到的实现——
#    `enbrown/iol-calculations` 是 **AGPL-3.0**（传染性最强）；`goldengrape/
#    Python-for-ophthalmologist`、`GiusepeG/iol-calc`、`junedanube/IOL_Calculator_AI`
#    均**无 LICENSE**；`rdimitr/iolcalc` 是 NOASSERTION。**没有一个许可干净到可以内嵌**。
#    故本段按**公开发表的公式**独立编写，不是任何仓库代码的复制。
#
# ⚠️ **验证级别（如实标注，别当临床计算器用）**
#    已验证：① 正算 → 反算往返一致；② 度数对眼轴单调（短眼 → 高度数）；
#            ③ 四公式在典型眼上的互相差异落在文献报告范围内；
#            ④ Haigis 的 d 反解对上公开算例（24 mm / 7.5 mm / 21 D / −2 D）。
#    未验证：与各家在线计算器（ASCRS / Barrett 等）逐例对拍 —— 那需要厂商常数。
#    → 用于**科研统计**（反推公式表现、常数优化）合适；
#      用于**临床决策**前，请用你所在机构的计算器复核。

# 顶点距离（mm）：三个经典公式的标准取值均为 12
_VERTEX_MM = 12.0
# 房水 / 玻璃体折射率
_N_AQUEOUS = 1.336


def K_from_r(r_mm: float, nk: float = 1.3375) -> float:
    """角膜曲率半径（mm）→ 角膜屈光力 K（D）：K = (nk − 1)·1000 / r。

    **角膜折射指数 nk 各公式不同，不可混用**：
    SRK/T 与 Hoffer Q 取 1.3375；Holladay 1 取 4/3；Haigis 取 1.3315。
    """
    return (float(nk) - 1.0) * 1000.0 / float(r_mm)


def r_from_K(K_d: float, nk: float = 1.3375) -> float:
    """角膜屈光力 K（D）→ 曲率半径（mm）。``K_from_r`` 的逆运算。"""
    return (nk - 1.0) * 1000.0 / float(K_d)


def _bisect_power(f_se, target: float, lo: float = -10.0, hi: float = 60.0,
                  tol: float = 1e-10, max_iter: int = 300) -> float:
    """解 ``f_se(power) = target``。IOL 度数越大 → 术后越近视化 → 等效球镜越小，
    故 f_se 关于 power 单调递减，直接二分。

    两端同号说明输入不物理（如曲率半径填 0），**报错而不是给出假值**。
    """
    flo = f_se(lo) - target
    fhi = f_se(hi) - target
    if flo * fhi > 0:
        raise ValueError(
            f"在 [{lo}, {hi}] D 区间内无解（两端同号）——请检查输入是否物理合理"
            f"（眼轴 mm、角膜半径 mm、常数是否为该公式的单位）")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f_se(mid) - target
        if abs(fm) < tol or (hi - lo) < 1e-9:
            return mid
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── SRK/T ─────────────────────────────────────────────────────────────────────
def _srkt_terms(la_mm: float, r_mm: float, a_const: float):
    """SRK/T 的几何量（正算/反算共用）。"""
    la, r = float(la_mm), float(r_mm)
    K = 337.5 / r                                    # 算 ACD 用 nk = 1.3375
    ACDconst = 0.62467 * float(a_const) - 68.747     # A 常数 → ACD 常数
    offset = ACDconst - 3.336
    LCOR = la if la <= 24.2 else -3.446 + 1.715 * la - 0.0237 * la ** 2
    Cw = -5.40948 + 0.58412 * LCOR + 0.098 * K       # 角膜宽度
    x = r ** 2 - (Cw ** 2) / 4.0
    H = r - math.sqrt(x if x > 0 else 0.0)           # 平方根内取 0（Retzlaff 原式）
    ACDest = H + offset
    LOPT = la + (0.65696 - 0.02029 * la)             # 光学眼轴 = 眼轴 + 视网膜厚度
    return r, ACDest, LOPT


def srkt_refraction(la_mm: float, r_mm: float, a_const: float,
                    power_d: float) -> float:
    """SRK/T 正算：给定 IOL 度数 → 术后等效球镜（D，眼镜平面）。"""
    r, ACDest, LOPT = _srkt_terms(la_mm, r_mm, a_const)
    na, ncml = _N_AQUEOUS, 0.333
    p, V = float(power_d), _VERTEX_MM
    num = 1000 * na * (na * r - ncml * LOPT) \
        - p * (LOPT - ACDest) * (na * r - ncml * ACDest)
    den = na * (V * (na * r - ncml * LOPT) + LOPT * r) \
        - 0.001 * p * (LOPT - ACDest) * (V * (na * r - ncml * ACDest) + ACDest * r)
    return num / den


def srkt_power(la_mm: float, r_mm: float, a_const: float,
               target_refraction_d: float = 0.0) -> float:
    """SRK/T 反算：给定目标术后屈光 → 所需 IOL 度数（D）。**解析解**。

    光学式对 power 是线性的：``SE = (A1 − p·A2)/(B1 − p·B2)``
    → ``p = (A1 − SE·B1)/(A2 − SE·B2)``。
    """
    r, ACDest, LOPT = _srkt_terms(la_mm, r_mm, a_const)
    na, ncml, V = _N_AQUEOUS, 0.333, _VERTEX_MM
    a1 = 1000 * na * (na * r - ncml * LOPT)
    a2 = (LOPT - ACDest) * (na * r - ncml * ACDest)
    b1 = na * (V * (na * r - ncml * LOPT) + LOPT * r)
    b2 = 0.001 * (LOPT - ACDest) * (V * (na * r - ncml * ACDest) + ACDest * r)
    se = float(target_refraction_d)
    return (a1 - se * b1) / (a2 - se * b2)


# ── Hoffer Q（第三代中最难敲对的一个）────────────────────────────────────────
def _hofferq_acd(la_mm: float, r_mm: float, pacd: float):
    la = float(la_mm)
    K = 337.5 / float(r_mm)                          # nk = 1.3375
    if la <= 23.0:
        M, G = 1.0, 28.0
    else:
        M, G = -1.0, 23.5
    A = 31.0 if la > 31.0 else (18.5 if la < 18.5 else la)
    p2 = 0.3 * (A - 23.5)
    p3 = math.tan(math.radians(K)) ** 2
    p4 = 0.1 * M * ((23.5 - A) ** 2)
    p5 = math.tan(math.radians(0.1 * (G - A) ** 2))
    return K, float(pacd) + p2 + p3 + p4 * p5 - 0.99166


def hofferq_refraction(la_mm: float, r_mm: float, pacd: float,
                       power_d: float) -> float:
    """Hoffer Q 正算：给定 IOL 度数 → 术后等效球镜（D）。"""
    la = float(la_mm)
    K, ACD = _hofferq_acd(la_mm, r_mm, pacd)
    p = float(power_d)
    R = (1.336 / (1.336 / (1336.0 / (la - ACD - 0.05) - p)
                  + (ACD + 0.05) / 1000.0)) - K
    return R / (1.0 + 0.012 * R)


def hofferq_power(la_mm: float, r_mm: float, pacd: float,
                  target_refraction_d: float = 0.0) -> float:
    """Hoffer Q 反算（数值求根 —— 该式嵌套，无简洁解析解）。"""
    return _bisect_power(lambda p: hofferq_refraction(la_mm, r_mm, pacd, p),
                         float(target_refraction_d))


# ── Holladay 1 ────────────────────────────────────────────────────────────────
def _holladay1_terms(la_mm: float, r_mm: float, sf: float):
    la, R = float(la_mm), float(r_mm)
    Rag = R if R >= 7.0 else 7.0                     # 半径下限 7 mm
    AG = min(12.5 * la / 23.45, 13.5)                # 上限 13.5（≈ 眼轴 25.32 mm）
    ACD = 0.56 + Rag - math.sqrt(max(Rag ** 2 - AG ** 2 / 4.0, 0.0))
    return R, ACD, la + 0.2, float(sf)               # +0.2 mm 视网膜厚度


def holladay1_refraction(la_mm: float, r_mm: float, sf: float,
                         power_d: float) -> float:
    """Holladay 1 正算（角膜折射指数 4/3）。"""
    R, ACD, ALm, sf = _holladay1_terms(la_mm, r_mm, sf)
    na, nc, V = _N_AQUEOUS, 4.0 / 3.0, _VERTEX_MM
    p = float(power_d)
    a1 = 1000 * na * (na * R - (nc - 1) * ALm)
    a2 = (ALm - ACD - sf) * (na * R - (nc - 1) * (ACD + sf))
    b1 = na * (V * (na * R - (nc - 1) * ALm) + ALm * R)
    b2 = 0.001 * (ALm - ACD - sf) * (V * (na * R - (nc - 1) * (ACD + sf))
                                     + (ACD + sf) * R)
    return (a1 - p * a2) / (b1 - p * b2)


def holladay1_power(la_mm: float, r_mm: float, sf: float,
                    target_refraction_d: float = 0.0) -> float:
    """Holladay 1 反算（解析解，结构同 SRK/T）。"""
    R, ACD, ALm, sf = _holladay1_terms(la_mm, r_mm, sf)
    na, nc, V = _N_AQUEOUS, 4.0 / 3.0, _VERTEX_MM
    a1 = 1000 * na * (na * R - (nc - 1) * ALm)
    a2 = (ALm - ACD - sf) * (na * R - (nc - 1) * (ACD + sf))
    b1 = na * (V * (na * R - (nc - 1) * ALm) + ALm * R)
    b2 = 0.001 * (ALm - ACD - sf) * (V * (na * R - (nc - 1) * (ACD + sf))
                                     + (ACD + sf) * R)
    se = float(target_refraction_d)
    return (a1 - se * b1) / (a2 - se * b2)


# ── Haigis（第四代，三常数）────────────────────────────────────────────────────
def haigis_refraction(la_mm: float, r_mm: float, acd_mm: float,
                      a0: float, a1: float, a2: float, power_d: float) -> float:
    """Haigis 正算。``d = a0 + a1·ACD + a2·AL``（多元回归），角膜指数 1.3315。"""
    n, nc, dx = _N_AQUEOUS, 1.3315, 0.012
    d_mm = float(a0) + float(a1) * float(acd_mm) + float(a2) * float(la_mm)
    Rm, ALm, dm = float(r_mm) / 1000.0, float(la_mm) / 1000.0, d_mm / 1000.0
    Dc = (nc - 1.0) / Rm
    p = float(power_d)
    q_num = n * (n - p * (ALm - dm))
    q_den = n * (ALm - dm) + dm * (n - p * (ALm - dm))
    q = q_num / q_den
    return (q - Dc) / (1.0 + dx * (q - Dc))


def haigis_power(la_mm: float, r_mm: float, acd_mm: float,
                 a0: float, a1: float, a2: float,
                 target_refraction_d: float = 0.0) -> float:
    """Haigis 反算（数值求根）。"""
    return _bisect_power(
        lambda p: haigis_refraction(la_mm, r_mm, acd_mm, a0, a1, a2, p),
        float(target_refraction_d))


def haigis_solve_d(la_mm: float, r_mm: float, power_d: float,
                   refraction_d: float) -> float:
    """由**术后实际屈光**反推 Haigis 的 d（光学前房深度），单位 mm。

    用途：IOL 常数优化 —— 对每个病例反解 d，再对 (ACD, AL) 做多元回归，
    即得三重优化后的 a0 / a1 / a2。

    公开算例（IOL Solver）：``solve_d(AL=24 mm, R=7.5 mm, P=21 D, Rx=−2 D)``
    → d ≈ 5.4615 mm。
    """
    n, nc, dx = _N_AQUEOUS, 1.3315, 0.012
    Rm, ALm = float(r_mm) / 1000.0, float(la_mm) / 1000.0
    Dc = (nc - 1.0) / Rm
    Rx, p = float(refraction_d), float(power_d)
    z = Dc + Rx / (1.0 - Rx * dx)
    b = -p * (ALm + n / z)
    c = n * (ALm - n / z) + p * ALm * n / z
    disc = b * b - 4.0 * p * c
    if disc < 0:
        raise ValueError("判别式为负 —— 输入组合不物理（检查 power 与屈光的符号）")
    d = (1.0 / (2.0 * p)) * (-b - math.sqrt(disc))
    return d * 1000.0


# ── IOL 常数换算（各公式常数不可互换，只提供有文献依据的换算）──────────────
def acd_const_from_a_constant(a_const: float) -> float:
    """A 常数 → ACD 常数：``ACD = 0.62467·A − 68.747``（Retzlaff 1990）。"""
    return 0.62467 * float(a_const) - 68.747


def a_constant_from_acd_const(acd_const: float) -> float:
    """ACD 常数 → A 常数。``acd_const_from_a_constant`` 的逆运算。"""
    return (float(acd_const) + 68.747) / 0.62467


def haigis_a0_from_acd_const(acd_const: float, a1: float = 0.4, a2: float = 0.1,
                             mean_acd_mm: float = 3.37,
                             mean_al_mm: float = 23.39) -> float:
    """由厂商 ACD 常数导出 Haigis 的 a0（a1 / a2 用默认 0.4 / 0.1）。

    ``a0 = ACD常数 − a1·mean(ACD) − a2·mean(AL)``，均值取 3.37 mm 与 23.39 mm
    （Haigis 原始定义中的群体均值）。
    """
    return (float(acd_const) - float(a1) * float(mean_acd_mm)
            - float(a2) * float(mean_al_mm))


# ── 多公式横向比较 ────────────────────────────────────────────────────────────
def iol_power_comparison(la_mm: float, r_mm: float, *,
                         a_const: float | None = None,
                         pacd: float | None = None,
                         sf: float | None = None,
                         haigis_abc: Sequence[float] | None = None,
                         acd_mm: float | None = None,
                         target_refraction_d: float = 0.0) -> Dict[str, Any]:
    """同一只眼跑多个公式，返回各公式度数、极差与一致性提示。

    **各公式用各自的常数，彼此不可互换**（厂商分别公布）：
    SRK/T 用 A 常数；Hoffer Q 用 pACD；Holladay 1 用 SF；Haigis 用 (a0, a1, a2)。
    缺哪个常数就跳过哪个公式，并在 ``skipped`` 里写明原因。

    临床上本来就是几个公式一起跑再拍板，所以这里给的是**横向比较**，不是
    一个「正确答案」。极差大只说明该眼对公式选择敏感（常见于极端眼轴），
    不说明哪个更「对」。
    """
    out: Dict[str, Any] = {"values": {}, "skipped": {},
                           "target_refraction_d": float(target_refraction_d)}
    if a_const is not None:
        out["values"]["SRK/T"] = srkt_power(la_mm, r_mm, a_const, target_refraction_d)
    else:
        out["skipped"]["SRK/T"] = "缺 A 常数"
    if pacd is not None:
        out["values"]["Hoffer Q"] = hofferq_power(la_mm, r_mm, pacd, target_refraction_d)
    else:
        out["skipped"]["Hoffer Q"] = "缺 pACD"
    if sf is not None:
        out["values"]["Holladay 1"] = holladay1_power(la_mm, r_mm, sf, target_refraction_d)
    else:
        out["skipped"]["Holladay 1"] = "缺 SF"
    if haigis_abc is not None:
        if acd_mm is None:
            out["skipped"]["Haigis"] = "缺术前 ACD —— Haigis 必须实测 ACD"
        else:
            a0, a1, a2 = (float(x) for x in haigis_abc)
            out["values"]["Haigis"] = haigis_power(la_mm, r_mm, acd_mm,
                                                   a0, a1, a2, target_refraction_d)
    else:
        out["skipped"]["Haigis"] = "缺 a0/a1/a2"
    if len(out["values"]) >= 2:
        vals = list(out["values"].values())
        out["spread_d"] = max(vals) - min(vals)
        out["mean_d"] = sum(vals) / len(vals)
    out["engine"] = "lib.ophtha.iol_power_comparison"
    out["note"] = ("各公式常数不可互换；极差大只说明该眼对公式选择敏感，"
                   "不代表哪个结果『对』")
    return out
