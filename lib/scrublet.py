"""双胞体检测（Scrublet 全流程）—— `lib` 层包装。

算法在 `vendor/_scrublet/core.py`（移植自 swolock/scrublet，MIT）。
本模块只做：参数编排、按供体分批、结果整理、口径输出。

**为什么这一步不能省**：双胞体在轨迹/伪时间推断里会落在两个真实细胞群之间，
被算法当成"过渡态细胞"。不清掉，年龄轨迹与失守曲线都会被它污染
——而且污染方向是系统性的（伪过渡态集中在采样密度高的年龄段附近）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from vendor._scrublet import core as _core

__all__ = ["detect_doublets", "summarize_doublets", "doublet_score_summary"]


def _one_batch(E: np.ndarray, sim_doublet_ratio: float, expected_doublet_rate: float,
               stdev_doublet_rate: float, n_neighbors: int, gene_pctl: float,
               n_prin_comps: int, min_counts: float, min_cells: int,
               threshold: Optional[float], seed: int,
               min_genes_after_filter: int = 20) -> Dict[str, Any]:
    """单个批次（供体）的完整 scrublet 流程（照抄上游 scrub_doublets 的步骤顺序）。"""
    E = np.asarray(E, dtype=float)
    n_cells, n_genes_in = E.shape

    # ① 按细胞总计数归一化（目标 = 各细胞总计数的均值，上游默认口径）
    E_norm = _core.total_counts_normalize(E)

    # ② 基因过滤：表达门槛 + v-score 变异性
    gene_ix = _core.filter_genes(E_norm, min_vscore_pctl=gene_pctl,
                                 min_counts=min_counts, min_cells=min_cells)
    if gene_ix.size == 0:
        raise ValueError("基因过滤后一个基因都不剩——放宽 min_counts / min_cells / gene_pctl")

    # ⚠️ 关键门禁：保留基因太少时**不能继续**。
    # v-score 筛的是「高于泊松噪声的变异」；若数据几乎无过散布（例如纯模拟计数），
    # 它会把真正的信号基因一起滤掉，只剩少量低表达背景基因。此时 PCA 建立在噪声上，
    # 双胞体分数**看起来仍在 0-1 之间**（完全不像出错），却毫无判别力——典型静默失败。
    # 实测：400 基因里信号基因被全部滤掉、只剩 32 个背景基因，AUC 从预期 >0.9 掉到 0.68。
    if gene_ix.size < int(min_genes_after_filter):
        raise ValueError(
            f"基因过滤后仅剩 {gene_ix.size}/{n_genes_in} 个基因（< min_genes_after_filter="
            f"{min_genes_after_filter}）——v-score 可能把真实信号基因一并滤掉了。"
            "请降低 gene_pctl（如 50）或检查数据是否有过散布（真实 UMI 数据通常有）")

    E_f = E[:, gene_ix]

    # ③ 模拟双胞体
    tots = E_f.sum(axis=1)
    E_sim, tots_sim, _parents = _core.simulate_doublets(
        E_f, tots, sim_doublet_ratio=sim_doublet_ratio, seed=seed)

    # ④ 归一化到 1e6（上游第二次归一化的目标值固定为 1e6）
    E_f_norm = _core.total_counts_normalize(E_f, target_total=1e6, total_counts=tots)
    E_s_norm = _core.total_counts_normalize(E_sim, target_total=1e6, total_counts=tots_sim)

    # ⑤ z-score（用观测细胞的均值/标准差，模拟细胞套用同一组参数）
    gm = E_f_norm.mean(axis=0)
    gs = E_f_norm.std(axis=0, ddof=1)
    Z_obs = _core.gene_zscore(E_f_norm, gm, gs)
    Z_sim = _core.gene_zscore(E_s_norm, gm, gs)

    # ⑥ PCA（只用观测细胞拟合基，模拟细胞投影上去）
    n_pc = int(min(n_prin_comps, min(Z_obs.shape) - 1, Z_obs.shape[0] - 1))
    if n_pc < 2:
        raise ValueError(f"细胞数 {Z_obs.shape[0]} 太少，无法做 PCA")
    _scores, components, _sv = _core.pca_scores(Z_obs, n_pc)
    manifold_obs = (Z_obs - Z_obs.mean(axis=0)) @ components.T
    manifold_sim = (Z_sim - Z_obs.mean(axis=0)) @ components.T

    # ⑦ 打分与判定
    k = int(n_neighbors if n_neighbors else max(2, round(0.5 * np.sqrt(Z_obs.shape[0]))))
    sc = _core.doublet_scores(
        manifold_obs, manifold_sim, k=k,
        exp_doublet_rate=expected_doublet_rate,
        stdev_doublet_rate=stdev_doublet_rate)
    call = _core.call_doublets(sc["scores_obs"], sc["scores_sim"],
                              sc["errors_obs"], threshold=threshold)

    return {
        "scores_obs": sc["scores_obs"],
        "predicted": call["predicted_doublets"],
        "z_scores": call["z_scores"],
        "threshold": call["threshold"],
        "k_used": k,
        "k_adj": sc["k_adj"],
        "n_genes_used": int(gene_ix.size),
        "n_cells": int(n_cells),
        "n_sim": int(sc["scores_sim"].size),
        "detected_doublet_rate": call["detected_doublet_rate"],
        "detectable_doublet_fraction": call["detectable_doublet_fraction"],
        "overall_doublet_rate": call["overall_doublet_rate"],
        "gene_index_kept": gene_ix,
    }


def detect_doublets(counts, sim_doublet_ratio: float = 2.0,
                    expected_doublet_rate: float = 0.1,
                    stdev_doublet_rate: float = 0.03,
                    n_neighbors: Optional[int] = None,
                    gene_variability_pctl: float = 85.0,
                    n_prin_comps: int = 30, min_counts: float = 3,
                    min_cells: int = 3, threshold: Optional[float] = None,
                    seed: int = 0, batch: Optional[Sequence[Any]] = None,
                    batch_seed_offset: int = 1000,
                    min_genes_after_filter: int = 20) -> Dict[str, Any]:
    """检测双胞体（整块或按批次分别检测）。

    :param counts: (细胞 × 基因) 原始 **counts** 矩阵（不要传已归一化的数据
        ——归一化会破坏"计数相加=双胞体"这一模拟前提）
    :param batch: 供体/样本标签。**给了就按样本分别检测**：不同样本的捕获效率
        与细胞组成不同，混在一起检测会把"样本间差异"当成双胞体信号。
    :param threshold: ``None`` 时自动定阈（双峰谷）；**强烈建议人工核对分布后固定**
        ——自动阈值是 advisory，scrublet 官方也这么说。
    :param batch_seed_offset: 每批的随机种子偏移，保证批次间可复现且不互相干扰
    :return: dict
        - ``predicted``: 布尔数组（True = 判为双胞体），长度 = 细胞数
        - ``scores`` / ``z_scores``: 每细胞的 doublet score 与 z
        - ``per_batch`` / ``thresholds``: 分批结果与各自阈值
        - ``detected_doublet_rate`` / ``overall_doublet_rate``: 全局比例
        - ``method`` / ``caveat``: 口径与已知差异
    """
    if hasattr(counts, "toarray"):
        E = counts.toarray().astype(float)
    else:
        E = np.asarray(counts, dtype=float)
    if E.ndim != 2:
        raise ValueError("counts 必须是二维（细胞 × 基因）")
    if E.min() < 0:
        raise ValueError("counts 含负值——scrublet 要求非负计数矩阵")
    n_cells = E.shape[0]

    if batch is None:
        r = _one_batch(E, sim_doublet_ratio, expected_doublet_rate, stdev_doublet_rate,
                       n_neighbors, gene_variability_pctl, n_prin_comps,
                       min_counts, min_cells, threshold, seed,
                       min_genes_after_filter=min_genes_after_filter)
        return {
            "predicted": r["predicted"],
            "scores": r["scores_obs"],
            "z_scores": r["z_scores"],
            "detected_doublet_rate": r["detected_doublet_rate"],
            "detectable_doublet_fraction": r["detectable_doublet_fraction"],
            "overall_doublet_rate": r["overall_doublet_rate"],
            "n_cells": n_cells,
            "n_doublets": int(r["predicted"].sum()),
            "thresholds": {"all": r["threshold"]},
            "per_batch": {"all": {k: v for k, v in r.items()
                                  if k not in ("scores_obs", "predicted", "z_scores")}},
            "method": "scrublet（模拟双胞体 + kNN 贝叶斯打分）",
            "caveat": ("自动阈值是 advisory；PCA 用 numpy SVD（符号不固定）；"
                       "近邻用 cKDTree（并列距离顺序可能不同于 sklearn）"),
            "engine": "lib.scrublet.detect_doublets",
        }

    labels = [str(b) for b in batch]
    if len(labels) != n_cells:
        raise ValueError(f"batch 长度 {len(labels)} != 细胞数 {n_cells}")

    pred = np.zeros(n_cells, dtype=bool)
    scores = np.full(n_cells, np.nan)
    zs = np.full(n_cells, np.nan)
    per_batch: Dict[str, Any] = {}
    thresholds: Dict[str, Optional[float]] = {}
    kept_idx: Dict[str, np.ndarray] = {}
    skipped: Dict[str, str] = {}

    for bi, b in enumerate(dict.fromkeys(labels)):
        m = np.array([x == b for x in labels])
        sub = E[m]
        if sub.shape[0] < 10:
            skipped[b] = f"细胞数 {sub.shape[0]} < 10（批次太小，双胞体模拟不可靠）"
            continue
        try:
            r = _one_batch(sub, sim_doublet_ratio, expected_doublet_rate, stdev_doublet_rate,
                           n_neighbors, gene_variability_pctl, n_prin_comps,
                           min_counts, min_cells, threshold, seed + bi * batch_seed_offset,
                           min_genes_after_filter=min_genes_after_filter)
        except Exception as e:
            skipped[b] = f"检测失败：{str(e)[:70]}"
            continue

        idx = np.flatnonzero(m)
        pred[idx] = r["predicted"]
        scores[idx] = r["scores_obs"]
        zs[idx] = r["z_scores"]
        thresholds[b] = r["threshold"]
        kept_idx[b] = r["gene_index_kept"]
        per_batch[b] = {
            "n_cells": r["n_cells"], "n_doublets": int(r["predicted"].sum()),
            "threshold": r["threshold"], "k_used": r["k_used"],
            "n_genes_used": r["n_genes_used"], "n_sim": r["n_sim"],
            "detected_doublet_rate": r["detected_doublet_rate"],
            "detectable_doublet_fraction": r["detectable_doublet_fraction"],
            "overall_doublet_rate": r["overall_doublet_rate"],
        }

    valid = np.isfinite(scores)
    n_det = int(pred.sum())
    return {
        "predicted": pred,
        "scores": scores,
        "z_scores": zs,
        "detected_doublet_rate": float(n_det / valid.sum()) if valid.sum() else float("nan"),
        "overall_doublet_rate": float(np.nanmean(
            [v["overall_doublet_rate"] for v in per_batch.values()])) if per_batch else float("nan"),
        "n_cells": n_cells,
        "n_cells_scored": int(valid.sum()),
        "n_doublets": n_det,
        "thresholds": thresholds,
        "per_batch": per_batch,
        "skipped_batches": skipped,
        "gene_index_kept": kept_idx,
        "method": "scrublet（**按批次分别检测**，每批独立的基因过滤与归一化）",
        "caveat": ("自动阈值是 advisory；PCA 用 numpy SVD（符号不固定）；"
                   "近邻用 cKDTree（并列距离顺序可能不同于 sklearn）；"
                   "跳过的小批次必须显式处理，不得当作「无双胞体」"),
        "engine": "lib.scrublet.detect_doublets",
    }


def summarize_doublets(res: Dict[str, Any], batch: Optional[Sequence[Any]] = None) -> pd.DataFrame:
    """把结果压成「每批次一行」的表。"""
    rows = []
    for b, v in (res.get("per_batch") or {}).items():
        rows.append({"batch": b, **v})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["batch", "n_cells", "n_doublets", "threshold",
                                     "detected_doublet_rate"])
    return df.sort_values("detected_doublet_rate", ascending=False).reset_index(drop=True)


def doublet_score_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    """分数分布摘要（供画直方图与目视核对阈值）。"""
    s = np.asarray(res["scores"], dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"n": 0}
    return {
        "n": int(s.size),
        "min": float(np.min(s)), "max": float(np.max(s)),
        "mean": float(np.mean(s)), "median": float(np.median(s)),
        "q90": float(np.percentile(s, 90)), "q99": float(np.percentile(s, 99)),
        "thresholds": res.get("thresholds"),
        "n_doublets": int(res.get("n_doublets", 0)),
    }
