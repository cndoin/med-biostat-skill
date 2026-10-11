"""specificity.py —— 细胞类型表达特异性（**全项目唯一裁决点**）。

**为什么是"独立实现"而不是内嵌**
--------------------------------
细胞类型特异性的通行工具是 CELLEX / CELLECT。2026-10-12 查 GitHub 官方
SPDX：**两者都是 GPL-3.0**，按项目红线（GPL 一律不内嵌）**不能用**，
也不能照抄其代码。故本模块按**公开描述的特异性度量思路**独立编写，
并**明确声明**：这是 ESμ 风格的等价实现，**不是** CELLEX 官方产物，
数值不可与 CELLECT 输出逐位对比，只能与论文报告的定性结论比对。

（同一次核查还发现 harmonypy 也是 GPL-3、CELLEX 曾被误记为 MIT。
  许可一律查官方 SPDX，不凭印象。）

**度量定义（写到代码里，免得口头上漂移）**
----------------------------------------
设 ``E[g,c]`` 为基因 g 在细胞类型 c 的平均表达（非负值），``K`` 为细胞类型数。

- ``share``（**默认**）：``ES[g,c] = log2( (E[g,c]+eps) / (mean_c E[g,·]+eps) )``
  即"该细胞类型分到的表达份额，是均匀分配的多少倍"。
  - 只在一种细胞表达 → ``log2(K)``（最大）
  - 各细胞均等 → ``0``
  - 管家基因（哪儿都高）→ ``0`` —— **绝对表达量不进分子分母之外，天然免疫表达量伪影**
- ``max_norm``：``ES[g,c] = (E[g,c]+eps) / (max_c E[g,c]+eps)``，取值 (0,1]，
  以该基因自身的峰值为 1。与 Yanai 2005 的 τ 特异度指数同源思路（τ 是基因级标量，
  这里是逐细胞类型版）。

两条口径**不可混用**，结果里必带 ``caliber`` 字段。

⚠️ **一次实测教训（2026-10-12）**：初版把基线写成"以表达量自身为权重"
（``Σ E² / Σ E``），结果**单调性反了** —— 只在一种细胞表达的基因
（E=[10,0,0,0]）基线正好等于最大值，ES=0；而 [10,1,1,1] 反而得 +0.34。
也就是"越特异分越低"。这类 bug 单看代码看不出来，只有把
"极端特异"和"完全均匀"两个方向**都算一遍**才暴露。故本模块的测试
必须包含**方向性断言**（特异基因 ES > 均匀基因 ES），而不能只断言"能跑通"。

**红线**
--------
1. **特异性 ≠ 表达量**。高表达不等于特异。审稿人第一打击点就是"你的细胞定位
   是不是表达量伪影"，故必须与**表达率（检出率）**双口径并用 —— 见 `dual_evidence`。
2. **全零基因没有特异性**。表达全为 0 的基因 ES 一律 NaN 并计入 ``n_all_zero``，
   不填 0（填 0 会让它在下游被当成"极度不特异"而参与排序，是假的）。
3. **输入必须非负**。负表达（如已做 scale/中心化）会让 log2 无定义，直接报错，
   不静默取绝对值。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import lm as _lm

__all__ = ["specificity", "assign", "gene_set_test", "dual_evidence"]


def specificity(expr, gene_names: Optional[Sequence[str]] = None,
                cell_types: Optional[Sequence[str]] = None,
                caliber: str = "share",
                eps: float = 1e-12) -> Dict[str, Any]:
    """由 (基因 × 细胞类型) 平均表达矩阵计算特异性 ES。

    :param expr: (基因 × 细胞类型) 矩阵，**非负**（平均表达或 pseudobulk 均值）
    :param caliber: ``share``（默认，log2 份额富集）或 ``max_norm``（对自身峰值归一）
    :return: dict
        - ``es``: DataFrame (基因 × 细胞类型) 的 ES 值
        - ``base``: 每个基因的基线值
        - ``caliber``: 口径（**必须随数字一起报告**）
        - ``n_all_zero``: 全零基因数（ES 为 NaN）
    """
    E = np.asarray(expr, dtype=float)
    if E.ndim != 2:
        raise ValueError("expr 必须是二维矩阵 (基因 × 细胞类型)")
    if caliber not in ("share", "max_norm"):
        raise ValueError(f"caliber 只能是 share / max_norm，当前 {caliber!r}")
    if np.any(~np.isfinite(E)):
        raise ValueError("expr 含 NaN/Inf；缺失要在聚合阶段处理，这里不能静默当 0")
    if np.any(E < 0):
        raise ValueError(
            "expr 含负值——ES 建立在『表达比例的 log』上，负值无定义。"
            "若你的矩阵已 scale/中心化，请换用未中心化的平均表达")

    n_gene, n_ct = E.shape
    genes = (list(gene_names) if gene_names is not None
             else [f"g{i}" for i in range(n_gene)])
    cts = (list(cell_types) if cell_types is not None
           else [f"ct{i}" for i in range(n_ct)])
    if len(genes) != n_gene or len(cts) != n_ct:
        raise ValueError("gene_names / cell_types 长度与矩阵维度不匹配")

    rowsum = E.sum(axis=1)
    all_zero = rowsum <= 0
    if caliber == "share":
        base = E.mean(axis=1)                       # 均匀分配下的份额
        with np.errstate(divide="ignore", invalid="ignore"):
            es = np.log2((E + eps) / (base[:, None] + eps))
    else:
        peak = E.max(axis=1)                        # 该基因自身的峰值
        base = peak
        with np.errstate(divide="ignore", invalid="ignore"):
            es = (E + eps) / (peak[:, None] + eps)
    es[all_zero, :] = np.nan

    return {
        "es": pd.DataFrame(es, index=genes, columns=cts),
        "base": pd.Series(base, index=genes),
        "caliber": caliber,
        "eps": float(eps),
        "n_genes": int(n_gene),
        "n_cell_types": int(n_ct),
        "n_all_zero": int(all_zero.sum()),
        "note": ("ESμ 风格独立实现（CELLEX/CELLECT 为 GPL-3，不可内嵌也不可照抄）；"
                 "数值不可与 CELLECT 官方产物逐位对比"),
        "engine": "lib.specificity.specificity",
    }


def assign(res: Dict[str, Any], min_es: float = 0.0,
           ambiguity_gap: float = 0.5) -> pd.DataFrame:
    """给每个基因指定最特异的细胞类型，并标出"拿不准"的。

    :param min_es: ES 低于此值视为**没有明确归属**（返回 ``None`` 而非硬选一个）
    :param ambiguity_gap: 第一名与第二名的 ES 差小于此值 → ``ambiguous=True``
    :return: DataFrame(gene, top_cell_type, es, runner_up, es_runner_up,
                       gap, ambiguous, unassigned)
    """
    es = res["es"]
    M = es.to_numpy(dtype=float)
    genes = list(es.index)
    cts = list(es.columns)
    n_ct = M.shape[1]

    rows = []
    for i, g in enumerate(genes):
        row = M[i]
        if np.all(~np.isfinite(row)):
            rows.append({"gene": g, "top_cell_type": None, "es": np.nan,
                         "runner_up": None, "es_runner_up": np.nan,
                         "gap": np.nan, "ambiguous": True, "unassigned": True})
            continue
        order = np.argsort(-np.where(np.isfinite(row), row, -np.inf))
        top = int(order[0])
        top_es = float(row[top])
        if n_ct > 1 and np.isfinite(row[order[1]]):
            ru, ru_es = int(order[1]), float(row[order[1]])
            gap = top_es - ru_es
        else:
            ru, ru_es, gap = None, np.nan, np.inf
        # 用 <=：均匀基因 ES 恰为 0，硬给它指定一个"最特异类型"就是把噪声当发现
        unassigned = (not np.isfinite(top_es)) or (top_es <= min_es)
        rows.append({
            "gene": g,
            "top_cell_type": None if unassigned else cts[top],
            "es": top_es,
            "runner_up": cts[ru] if ru is not None else None,
            "es_runner_up": ru_es,
            "gap": float(gap) if np.isfinite(gap) else np.nan,
            "ambiguous": bool(np.isfinite(gap) and gap < ambiguity_gap),
            "unassigned": bool(unassigned),
        })
    return pd.DataFrame(rows)


def gene_set_test(res: Dict[str, Any], gene_set: Sequence[str],
                  background: Optional[Sequence[str]] = None,
                  alternative: str = "greater") -> Dict[str, Any]:
    """某基因集（如 78 个 GWAS 风险基因）在特定细胞类型上是否**更特异**。

    用 Mann-Whitney U（秩和）：比较基因集内与背景基因在该细胞类型的 ES 分布。
    为什么不用 t 检验：ES 分布长尾且非正态，秩检验更稳。

    :param background: 背景基因（默认 = 矩阵里除基因集外的全部基因）。
        **背景必须显式意识到**：不给背景就等于拿"全集减自己"当背景，
        若基因集占了全集大半，检验会失去分辨力。
    :return: dict，含每个细胞类型的 U / p / padj(BH) / 均值差
    """
    es = res["es"]
    in_set = [str(g) for g in gene_set]
    present = [g for g in in_set if g in es.index]
    missing = sorted(set(in_set) - set(es.index))
    if not present:
        raise ValueError(
            f"基因集有 {len(in_set)} 个基因，与表达矩阵**一个都匹配不上**"
            f"（示例：{in_set[:3]}）。这通常是基因命名体系不一致（Ensembl vs Symbol），"
            f"不报错会让下游静默算出空结果")

    if background is None:
        bg = [g for g in es.index if g not in set(present)]
    else:
        bg = [str(g) for g in background if str(g) in es.index and str(g) not in set(present)]
    if len(bg) < 5:
        raise ValueError(f"背景基因太少（{len(bg)} 个），秩检验没有分辨力")

    rows = []
    for c in es.columns:
        a = es.loc[present, c].to_numpy(dtype=float)
        b = es.loc[bg, c].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 3 or b.size < 3:
            rows.append({"cell_type": c, "mean_es_set": np.nan, "mean_es_bg": np.nan,
                         "delta_es": np.nan, "U": np.nan, "pvalue": np.nan})
            continue
        U, p = stats.mannwhitneyu(a, b, alternative=alternative)
        rows.append({"cell_type": c, "mean_es_set": float(a.mean()),
                     "mean_es_bg": float(b.mean()),
                     "delta_es": float(a.mean() - b.mean()),
                     "U": float(U), "pvalue": float(p)})
    tab = pd.DataFrame(rows)
    tab["padj"] = _lm.bh_adjust(tab["pvalue"].to_numpy(dtype=float))
    return {
        "table": tab,
        "n_in_set": len(present),
        "n_missing": len(missing),
        "missing_examples": missing[:10],
        "n_background": len(bg),
        "caliber": res["caliber"],
        "test": "Mann-Whitney U (rank-sum)",
        "alternative": alternative,
        "engine": "lib.specificity.gene_set_test",
    }


def dual_evidence(res: Dict[str, Any], rates,
                  min_rate: float = 0.1) -> pd.DataFrame:
    """**表达率 + 特异性双口径**互证（排除"细胞定位是表达量伪影"）。

    :param rates: (基因 × 细胞类型) 检出率矩阵，取值 0–1
    :param min_rate: 低于此检出率的细胞类型不参与归属（测都测不到，谈何定位）
    :return: DataFrame(gene, top_by_es, top_by_rate, agree, es_top, rate_top,
                       flagged)
        ``flagged=True`` 表示两口径不一致 → **必须人工看**，
        不能只报 ES 口径的结论。
    """
    es = res["es"]
    R = pd.DataFrame(rates)
    if not R.index.equals(es.index) or not R.columns.equals(es.columns):
        raise ValueError("rates 的行列必须与 es 完全一致（基因 × 细胞类型）")
    if (R.to_numpy(dtype=float) < 0).any() or (R.to_numpy(dtype=float) > 1).any():
        raise ValueError("rates 必须在 0–1 之间（检出率）")

    asg = assign(res).set_index("gene")
    out = []
    for g in es.index:
        r = R.loc[g].to_numpy(dtype=float)
        cand = [c for c, v in zip(es.columns, r) if np.isfinite(v) and v >= min_rate]
        top_rate = None
        if cand:
            sub = R.loc[g, cand]
            top_rate = str(sub.idxmax())
        top_es = asg.loc[g, "top_cell_type"]
        out.append({
            "gene": g,
            "top_by_es": None if (isinstance(top_es, float) and np.isnan(top_es)) else top_es,
            "top_by_rate": top_rate,
            "agree": bool(top_rate is not None and top_es == top_rate),
            "es_top": float(asg.loc[g, "es"]),
            "rate_top": float(R.loc[g, top_rate]) if top_rate else np.nan,
            "n_detected_types": int(np.sum(r >= min_rate)),
            "flagged": bool(top_rate is None or
                            (top_es is not None and top_es != top_rate)),
        })
    return pd.DataFrame(out)
