# -*- coding: utf-8 -*-
"""ophtha.py —— 眼科 / 白内障生信与临床计算

**定位**：包装层。光学与屈光的核心数学来自 vendored 开源项目
**PEARL-DGS**（`vendor/_ophtha/pearldgs.py`，MIT，来源 gdebel/pearldgs_toolbox），
本模块只做两件事：① 把物理单位（米）换成临床单位（mm / D）；
② 补上眼科临床与流行病学的标准指标（这些是公开定义的标准量，不是自造算法）。

**红线**：这里没有任何一步依赖模型判断。全部是给定输入 → 确定性数值。
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from vendor._ophtha import pearldgs as _pg

__all__ = [
    "REFRACTIVE_INDEX", "tilp_from_refraction", "refraction_from_tilp",
    "corneal_power", "predicted_posterior_radius", "segmented_axial_length",
    "logmar_from_decimal", "decimal_from_logmar", "logmar_from_snellen",
    "refractive_outcome", "iop_stats", "CATARACT_GENE_SETS",
    "lens_gene_overlap",
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
