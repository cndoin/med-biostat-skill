"""proteomics.py —— 蛋白组差异分析 + RNA-蛋白断层分析。

**为什么没有内嵌 limma / MSstats**
----------------------------------
- ``limma``：GPL-2/GPL-3（Bioconductor），**不能内嵌**
- ``MSstats``：**GPL-3**，不能内嵌

故本模块的差异检验走 `lib.lm.moderated_lm` —— 那是按 Smyth 2004 论文
**独立实现**的 moderated t（limma 的核心思想），与 EWAS（`lib.epigen`）
共用**同一个引擎**，保证 RNA 侧与蛋白侧的 p 值**口径一致、可直接比**。
这一点是"RNA-蛋白是否同向"能判得出来的前提：两侧 p 值若来自两套实现，
"同向"就只是两个数字的偶然。

**RNA-蛋白断层：当发现写，不当噪声**
------------------------------------
转录本与蛋白不一致是**有生物学含义的**（翻译后调控、蛋白降解、分泌），
不是技术噪声。本模块把断层**量化**出来并单独成表，供下游功能富集。

**唯一裁决点**
--------------
`rna_protein_discordance` 是全项目**唯一**定义"一致/断层"的地方。
别处不许再写一套 sign(logFC_rna) == sign(logFC_prot) 的判断——
那种散落各处的判断一定会漂移（有人用 p 值、有人用 padj、有人不设阈值），
最后同一批基因在不同表里结论不同。

**红线**
--------
1. **一致性率的分母必须是"两侧都显著"的基因**。若把"两侧都没测到"的基因
   也算进分母，一致性率会被大量零信息基因稀释到接近随机。
   本函数**同时**报两个率（严格 / 宽松），并把口径写进结果。
2. **方向相反但只有一侧显著 ≠ 断层**，单列为 `weak_discordant`，不混进主计数。
3. 强度矩阵 log2 前必须为正；含 0/负值一律**报错**，不静默加伪计数
   （加多少都会改变低丰度蛋白的 log2FC，是隐藏的口径漂移）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import lm as _lm

__all__ = ["differential", "dep_summary", "rna_protein_discordance"]


def differential(intensity, group: Sequence[Any],
                 protein_names: Optional[Sequence[str]] = None,
                 covariates: Optional[np.ndarray] = None,
                 log2_input: bool = True,
                 moderated: bool = True) -> Dict[str, Any]:
    """两组间蛋白差异分析（moderated t）。

    :param intensity: (样本 × 蛋白) 强度矩阵。``log2_input=True`` 时认为**已经是
        log2 域**；为 False 时本函数做 log2（此时必须全为正）
    :param group: 长度 = 样本数的二分类标签
    :param covariates: 需调整的协变量（年龄/性别/批次），**强烈建议给**
    :return: dict
        - ``table``: DataFrame(protein, log2FC, t, pvalue, padj)
        - ``d0`` / ``dof`` / ``method``: 口径，必须一起报告
        - ``n_case`` / ``n_ctrl``
    """
    I = np.asarray(intensity, dtype=float)
    if I.ndim != 2:
        raise ValueError("intensity 必须是二维矩阵 (样本 × 蛋白)")
    group = np.asarray(group)
    if group.shape[0] != I.shape[0]:
        raise ValueError(f"group 长度 {group.shape[0]} != 样本数 {I.shape[0]}")
    levels = list(pd.unique(group))
    if len(levels) != 2:
        raise ValueError(f"group 必须是二分类，当前有 {len(levels)} 类：{levels}")
    if np.any(~np.isfinite(I)):
        raise ValueError("intensity 含 NaN/Inf。**蛋白组的缺失不是随机缺失（MNAR）**："
                         "低于检测限的蛋白会成片缺失，直接填 0 会把「没测到」当成"
                         "「不表达」。请调用方先显式决定填补或过滤策略")

    if not log2_input:
        if np.any(I <= 0):
            raise ValueError(
                "log2_input=False 时强度必须全为正（含 0 或负值无法取 log2）。"
                "请勿自行加伪计数——伪计数大小会系统性改变低丰度蛋白的 log2FC")
        I = np.log2(I)

    n_prot = I.shape[1]
    names = (list(protein_names) if protein_names is not None
             else [f"p{i}" for i in range(n_prot)])
    if len(names) != n_prot:
        raise ValueError(f"protein_names 长度 {len(names)} != 蛋白数 {n_prot}")

    is_case = (group == levels[1]).astype(float)
    cols = [np.ones(len(is_case)), is_case]
    if covariates is not None:
        C = np.asarray(covariates, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != len(is_case):
            raise ValueError("covariates 行数与样本数不一致")
        cols.append(C)
    X = np.column_stack(cols)

    fit = _lm.moderated_lm(I, X, coef_index=1, moderated=moderated)
    grp1 = (group == levels[1])
    log2fc = np.nanmean(I[grp1], axis=0) - np.nanmean(I[~grp1], axis=0)

    table = pd.DataFrame({
        "protein": names,
        "log2FC": log2fc,
        "t": fit["t"],
        "pvalue": fit["pvalue"],
        "padj": _lm.bh_adjust(fit["pvalue"]),
    })
    return {
        "table": table,
        "d0": fit["d0"],
        "s0_sq": fit["s0_sq"],
        "dof": fit["dof"],
        "resid_dof": fit["resid_dof"],
        "n_case": int(grp1.sum()),
        "n_ctrl": int((~grp1).sum()),
        "log2_input": bool(log2_input),
        "moderated": bool(moderated),
        "method": fit["method"],
        "engine": "lib.proteomics.differential",
    }


def dep_summary(result: Dict[str, Any], fdr_cut: float = 0.05,
                log2fc_cut: float = 0.0) -> Dict[str, Any]:
    """差异蛋白（DEP）汇总。

    **必须同时设 FDR 与 log2FC 两条阈值**：蛋白组样本量通常不大，
    单看 FDR 会放进大量"统计显著但效应极小"的蛋白。
    """
    t = result["table"]
    sig = t[(t["padj"] < fdr_cut) & (t["log2FC"].abs() >= log2fc_cut)]
    return {
        "n_tested": int(len(t)),
        "n_dep": int(len(sig)),
        "n_up": int((sig["log2FC"] > 0).sum()),
        "n_down": int((sig["log2FC"] < 0).sum()),
        "fdr_cut": float(fdr_cut),
        "log2fc_cut": float(log2fc_cut),
        "top": sig.reindex(sig["padj"].sort_values().index).head(10),
        "engine": "lib.proteomics.dep_summary",
    }


def rna_protein_discordance(rna: Any, prot: Any,
                            key_rna: str = "gene",
                            key_prot: str = "protein",
                            fdr_cut: float = 0.05,
                            logfc_cut: float = 0.0) -> Dict[str, Any]:
    """RNA-蛋白一致性 / 断层分析（**唯一裁决点**）。

    :param rna: RNA 侧差异结果——`dict`（含 ``table``）或 DataFrame
    :param prot: 蛋白侧差异结果——同上
    :return: dict
        - ``table``: 逐基因分类表（class / logFC_rna / logFC_prot / 两侧 padj）
        - ``n_concordant`` / ``n_discordant``（**断层**）/ ``n_weak_discordant``
        - ``strict_rate``: 一致性率，分母 = 两侧都显著的基因（**主指标**）
        - ``loose_rate``: 宽松率，分母 = 至少一侧显著的基因（辅助，口径不同）
        - ``binomial_p``: 对 strict 分母做二项检验（H0: 一致率 = 0.5）
        - ``caliber``: 阈值口径

    分类规则（写死，不得事后调整）：
        - ``concordant``：两侧**都** padj<cut 且 logFC 同向
        - ``discordant``（**断层**）：两侧**都** padj<cut 且 logFC 反向
        - ``weak_discordant``：反向但**只有一侧**显著 → 单独计数，**不进主率**
        - ``rna_only`` / ``prot_only``：同向但只有一侧显著
        - ``null``：两侧都不显著
    """
    def _as_table(x, key):
        if isinstance(x, dict):
            t = x["table"]
        else:
            t = pd.DataFrame(x)
        if key not in t.columns:
            raise ValueError(f"表缺少列 {key!r}；现有列 {list(t.columns)[:8]}")
        return t[[key, "log2FC" if "log2FC" in t.columns else "logFC",
                  "padj" if "padj" in t.columns else "pvalue"]].copy()

    a = _as_table(rna, key_rna)
    b = _as_table(prot, key_prot)
    a.columns = ["gene", "logFC", "padj"]
    b.columns = ["gene", "logFC", "padj"]
    m = a.merge(b, on="gene", suffixes=("_rna", "_prot"), how="inner")
    if len(m) == 0:
        raise ValueError("两侧基因**一个都匹配不上**——通常是命名体系不一致"
                         "（RNA 用 Symbol、蛋白用 UniProt）。不报错会让下游"
                         "拿着空表算出「一致性率 0」这种假结论")

    sa = (m["padj_rna"] < fdr_cut) & (m["logFC_rna"].abs() >= logfc_cut)
    sb = (m["padj_prot"] < fdr_cut) & (m["logFC_prot"].abs() >= logfc_cut)
    same = np.sign(m["logFC_rna"]) == np.sign(m["logFC_prot"])
    opp = np.sign(m["logFC_rna"]) * np.sign(m["logFC_prot"]) < 0

    cls = np.where(sa & sb & same, "concordant",
          np.where(sa & sb & opp, "discordant",
          np.where(opp & (sa | sb), "weak_discordant",
          np.where(sa & ~sb & same, "rna_only",
          np.where(sb & ~sa & same, "prot_only", "null")))))
    m["class"] = cls

    n_conc = int((cls == "concordant").sum())
    n_disc = int((cls == "discordant").sum())
    n_weak = int((cls == "weak_discordant").sum())
    denom_strict = n_conc + n_disc
    denom_loose = denom_strict + n_weak

    strict_rate = (n_conc / denom_strict) if denom_strict else float("nan")
    loose_rate = (n_conc / denom_loose) if denom_loose else float("nan")
    if denom_strict > 0:
        bt = stats.binomtest(n_conc, denom_strict, 0.5)
        binom_p = float(bt.pvalue)
    else:
        binom_p = float("nan")

    return {
        "table": m,
        "n_tested": int(len(m)),
        "n_concordant": n_conc,
        "n_discordant": n_disc,
        "n_weak_discordant": n_weak,
        "n_rna_only": int((cls == "rna_only").sum()),
        "n_prot_only": int((cls == "prot_only").sum()),
        "n_null": int((cls == "null").sum()),
        "strict_rate": float(strict_rate),
        "loose_rate": float(loose_rate),
        "denom_strict": denom_strict,
        "denom_loose": denom_loose,
        "binomial_p": binom_p,
        "discordant_genes": m.loc[m["class"] == "discordant", "gene"].tolist(),
        "caliber": {"fdr_cut": float(fdr_cut), "logfc_cut": float(logfc_cut),
                    "denominator": "strict = 两侧都显著（主指标）；"
                                   "loose = 至少一侧显著（辅助）",
                    "test": "二项检验 H0: 一致率=0.5"},
        "engine": "lib.proteomics.rna_protein_discordance",
    }
