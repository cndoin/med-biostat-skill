"""composition.py —— 细胞类型组成（亚群比例）随协变量的**供体层级**检验。

算法在 `vendor/_sccoda/core.py`（移植自 theislab/scCODA，**BSD-3-Clause**；
推断引擎从 HMC 换成惩罚 MAP + 供体 bootstrap，差异已在该文件头如实标注）。

**什么时候必须用这个，而不是对各类别分别做 t 检验**
--------------------------------------------------
1. **细胞不是独立样本**。几千个细胞来自十几个供体，拿细胞当 n 会把样本量
   虚增两三个数量级，p 值必然假阳性。检验单位是**供体**。
2. **比例天生和为 1**。某一细胞类型真的变多，其余类型的比例**必然**下降。
   对各类别独立做检验会同时犯两个错：把"被稀释"误判成"自己也变了"，
   以及忽略约束后方差估计是错的。成分模型（Dirichlet-Multinomial +
   log-ratio 效应）是这类问题的正解。

**miloR 为什么没有**
--------------------
Milo 的 R 实现 `MarioniLab/miloR` 是 **GPL-3**，按红线不能内嵌也不能照抄。
scCODA 与 Milo 回答的是**不同问题**：scCODA 问"某个细胞类型的比例是否随
协变量变"，Milo 问"在 KNN 邻域里哪些局部区域变了"。
缺了 Milo 意味着**不能做邻域级的局部检验**——这一点要如实写进局限。

**口径（必须随数字一起报告）**
------------------------------
- ``estimate`` 是**相对参考细胞类型的 log 丰度比**随每单位协变量的变化，
  **不是**比例的变化量。结果里带 ``reference`` 字段，写论文时必须写明。
- ``p_boot`` / ``ci`` 来自**供体 bootstrap**（固定 seed，可复现），
  **不是**后验概率；``sel_freq`` **不是**上游 scCODA 的后验包含概率。
- 效应量经过"L1 选变量 → 支持集上无惩罚重拟合"，所以**点估计没有 L1 收缩偏差**，
  但**选择这一步本身带来的乐观性由 bootstrap 承担** —— 因此必须看
  ``padj`` 而不是只看点估计。

**红线**
--------
1. 输入的计数矩阵必须是**供体 × 细胞类型**，不是细胞 × 基因。
   若你手上是细胞级数据，先用 `lib.pseudobulk` 或本模块的 `counts_from_labels`。
2. 供体数太少（<3）直接报错，不"尽力算一个"—— 成分模型的自由度是 (K−1)×P，
   供体不够时任何结果都是噪声。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from vendor._sccoda import core as _core

__all__ = ["counts_from_labels", "test", "effect_table", "summary"]


def counts_from_labels(donor: Sequence[Any], cell_type: Sequence[Any]) -> pd.DataFrame:
    """由细胞级标签生成 (供体 × 细胞类型) 计数矩阵（成分检验的输入）。"""
    d = np.asarray(donor)
    c = np.asarray(cell_type)
    if d.shape[0] != c.shape[0]:
        raise ValueError(f"donor 长度 {d.shape[0]} != cell_type 长度 {c.shape[0]}")
    if d.size == 0:
        raise ValueError("标签为空")
    tab = pd.crosstab(pd.Series(d, name="donor"), pd.Series(c, name="cell_type"))
    if tab.shape[1] < 2:
        raise ValueError(f"只有 {tab.shape[1]} 个细胞类型；成分检验至少需要 2 类")
    return tab


def test(counts, X, cell_types: Optional[Sequence[str]] = None,
         covariate_names: Optional[Sequence[str]] = None,
         reference: Optional[int] = None,
         B: int = 200, seed: int = 0, alpha: float = 0.05,
         intercept: bool = True) -> Dict[str, Any]:
    """供体层级的成分差异检验（scCODA 模型族）。

    :param counts: (供体 × 细胞类型) **非负整数计数**
    :param X: (供体 × 协变量) 设计矩阵。`intercept=True` 时第一列视为截距
        （不参与 L1 选择，因为基线组成不该被压向均匀）
    :param reference: 参考细胞类型下标（默认最后一类）
    :param B: bootstrap 次数（默认 200，固定 seed 可复现）
    :return: dict，含 ``table``（整洁表）/ ``fit``（原始结果）/
        ``bootstrap`` / ``reference`` / ``inference``
    """
    if isinstance(counts, pd.DataFrame):
        cts = list(counts.columns)
        Y = counts.to_numpy(dtype=float)
    else:
        Y = np.asarray(counts, dtype=float)
        cts = list(cell_types) if cell_types is not None else [f"ct{i}" for i in range(Y.shape[1])]
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"counts {Y.shape} 与 X {X.shape} 行数不一致")
    if len(cts) != Y.shape[1]:
        raise ValueError(f"cell_types 长度 {len(cts)} != 列数 {Y.shape[1]}")
    cov = (list(covariate_names) if covariate_names is not None
           else [f"x{j}" for j in range(X.shape[1])])
    if len(cov) != X.shape[1]:
        raise ValueError(f"covariate_names 长度 {len(cov)} != X 列数 {X.shape[1]}")

    fit = _core.fit(Y, X, reference=reference, intercept=intercept)
    boot = _core.bootstrap(Y, X, reference=fit["reference"],
                           lam=fit["lambda"], B=B, seed=seed,
                           alpha=alpha, intercept=intercept)

    tab = effect_table(boot, cts, cov)
    return {
        "table": tab,
        "fit": fit,
        "bootstrap": boot,
        "reference": cts[fit["reference"]],
        "reference_index": int(fit["reference"]),
        "n_donors": int(Y.shape[0]),
        "n_cell_types": int(Y.shape[1]),
        "lambda": float(fit["lambda"]),
        "theta": float(fit["theta"]),
        "intercept": bool(intercept),
        "intercept_covariate": (str(cov[0]) if (intercept and cov) else None),
        "B": int(boot["B"]),
        "seed": int(seed),
        "alpha": float(alpha),
        "upstream": "theislab/scCODA (BSD-3-Clause)",
        "inference": boot["inference"],
        "estimation": fit["estimation"],
        "engine": "lib.composition.test",
    }


def effect_table(boot: Dict[str, Any], cell_types: Sequence[str],
                 covariate_names: Sequence[str]) -> pd.DataFrame:
    """把 bootstrap 结果摊成整洁表（每行 = 细胞类型 × 协变量）。

    参考细胞类型那一行的效应**恒为 0 且无检验**（它是可辨识性约束，不是发现），
    表里用 ``is_reference=True`` 标出，避免被当成"该类型没有变化"。
    """
    K, P = boot["est"].shape
    ref = boot["reference"]
    rows = []
    for k in range(K):
        for j in range(P):
            rows.append({
                "cell_type": str(cell_types[k]),
                "covariate": str(covariate_names[j]),
                "estimate": float(boot["est"][k, j]),
                "ci_low": float(boot["ci_low"][k, j]),
                "ci_high": float(boot["ci_high"][k, j]),
                "p_boot": float(boot["p_boot"][k, j]),
                "padj": float(boot["padj"][k, j]) if np.isfinite(boot["padj"][k, j]) else np.nan,
                "sel_freq": float(boot["sel_freq"][k, j]),
                "selected": bool(boot["selected"][k, j]),
                "is_reference": bool(k == ref),
            })
    return pd.DataFrame(rows)


def summary(res: Dict[str, Any], padj_cut: float = 0.05) -> Dict[str, Any]:
    """显著变化汇总（只看非参考行的被惩罚协变量）。"""
    t = res["table"]
    t = t[~t["is_reference"]]
    # 截距不是发现：它描述基线组成，"截距显著"是废话（基线当然不为 0），
    # 若把它算进显著数，会把 n_significant 虚增 (K−1) 个。
    if res.get("intercept_covariate"):
        t = t[t["covariate"] != res["intercept_covariate"]]
    sig = t[(t["padj"] < padj_cut) & (t["sel_freq"] >= 0.5)]
    return {
        "n_tests": int(len(t)),
        "n_significant": int(len(sig)),
        "reference": res["reference"],
        "padj_cut": float(padj_cut),
        "n_donors": res["n_donors"],
        "lambda": res["lambda"],
        "B": res["B"],
        "significant": sig.sort_values("padj"),
        "note": ("estimate 是相对参考细胞类型 " + str(res["reference"]) +
                 " 的 log 丰度比变化；sel_freq 与 p 来自供体 bootstrap，"
                 "**不是** scCODA 上游的 HMC 后验包含概率"),
        "engine": "lib.composition.summary",
    }
