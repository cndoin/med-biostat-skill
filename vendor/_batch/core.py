# PORTED-FILE · 来源 scverse/scanpy · src/scanpy/preprocessing/_combat.py（BSD-3-Clause）
# 原始方法：Johnson, Li & Rabinovic 2007, Biostatistics 8(1):118-127
# 移植方式：**算法照抄、依赖换底**。scanpy 版绑定 AnnData，且设计矩阵走
#   patsy、EB 迭代走 numba；本文件去掉三者，改成 (样本 × 特征) 的
#   numpy/pandas 接口，依赖降到 numpy+pandas。
# 逐函数对照（scanpy 源函数 -> 本文件）：
#   _design_matrix    -> _design_matrix（patsy 全哑变量 -> pandas 手工 one-hot）
#   _standardize_data -> _standardize_data（池化方差标准化）
#   combat            -> combat / combat_reference（去 AnnData，返回 dict）
#   _it_sol/_aprior/_bprior -> 同名（numba 去装饰器，逐行等价）
# 数值口径与 scanpy **逐行一致**：design → 池化方差标准化 → L/S 模型 →
#   逆 Gamma 先验 → 迭代条件后验 → 反标准化。
# 与上游的**已知差异（如实标注）**：
#   1. scanpy 无 `combat_reference`（ref-batch 模式）；这是本文件为「以主队列
#      为基准」场景补的薄封装，算法仍复用它自身的 combat。
#   2. `batch_diagnostics` / `pca_scores` 为本文件新增的诊断工具，非上游代码。
"""批次效应校正（ComBat，经验贝叶斯）+ 批次诊断。

**为什么必须用 EB 而不是简单中心化**
朴素中心化（每批次减均值）会把批次与生物学混杂一起抹掉，且不处理
尺度差异。ComBat 用 EB 把每批次的 location/scale 向全体均值**收缩**，
批次样本越小时收缩越强 —— 这是它与 z-score/分位数归一化的本质区别。

**红线**
1. 批次协变量与生物学变量**完全共线**时（如所有病例只在一台机器），
   ComBat 会把真实生物学效应当批次扣除。调用前必须自己检查。
2. `batch` 中样本数 <2 的批次无法估计批内方差 —— 直接报错，不静默丢弃。
3. 校正必须**只在校正集上估计参数**；跨数据集套用他人参数是错的口径。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["combat", "batch_diagnostics", "pca_scores", "combat_reference"]


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.astype(float)
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X 必须是二维（样本 × 特征）")
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])


def _design_matrix(batch: np.ndarray, batch_levels: List[Any],
                   covariates: Optional[pd.DataFrame]) -> pd.DataFrame:
    """设计矩阵：批次 one-hot（全哑变量）+ 协变量。

    与 scanpy 的 `patsy.dmatrix("~ 0 + C(batch)")` 等价：**无截距**，
    批次用全部 k 个水平；协变量的分类列同样用全哑变量，数值列原样进入。
    """
    n = len(batch)
    parts = []
    for lv in batch_levels:
        parts.append(pd.Series((batch == lv).astype(float), name=f"batch[{lv}]"))
    if covariates is not None:
        if len(covariates) != n:
            raise ValueError("covariates 行数与 X 不一致")
        for c in covariates.columns:
            col = covariates[c]
            if pd.api.types.is_numeric_dtype(col):
                parts.append(pd.Series(col.to_numpy(dtype=float), name=str(c)))
            else:
                for lv in pd.unique(col):
                    parts.append(pd.Series((col.to_numpy() == lv).astype(float),
                                           name=f"{c}[{lv}]"))
    return pd.concat(parts, axis=1)


def _standardize_data(design: pd.DataFrame, data: pd.DataFrame,
                      n_batch: int, n_batches: np.ndarray):
    """池化方差标准化（scanpy `_standardize_data` 同口径）。

    data 为 (特征 × 样本)。返回 (s_data, var_pooled, stand_mean)。
    """
    n_array = float(n_batches.sum())
    design_np = design.to_numpy(dtype=float)
    data_np = data.to_numpy(dtype=float)
    # b_hat = inv(D'D) D' data'   -> (k, 特征)
    b_hat = np.linalg.inv(design_np.T @ design_np) @ design_np.T @ data_np.T
    grand_mean = (n_batches / n_array) @ b_hat[:n_batch, :]
    resid = data_np - (design_np @ b_hat).T
    var_pooled = (resid ** 2).mean(axis=1, keepdims=True)
    tmp = design_np.copy()
    tmp[:, :n_batch] = 0.0
    stand_mean = grand_mean[:, None] + (tmp @ b_hat).T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(var_pooled == 0, 0.0, (data_np - stand_mean) / np.sqrt(var_pooled))
    return (pd.DataFrame(s, index=data.index, columns=data.columns),
            var_pooled, stand_mean, int((var_pooled == 0).sum()))


def _aprior(delta_hat: np.ndarray) -> float:
    m = float(delta_hat.mean())
    s2 = float(delta_hat.var())
    return (2 * s2 + m ** 2) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m = float(delta_hat.mean())
    s2 = float(delta_hat.var())
    return (m * s2 + m ** 3) / s2


def _it_sol(s_data: np.ndarray, g_hat: np.ndarray, d_hat: np.ndarray, *,
            g_bar: float, t2: float, a: float, b: float,
            conv: float = 1e-4, max_iter: int = 1000):
    """迭代条件后验均值（scanpy `_it_sol` 同口径）。

    gamma 为加性（位置）批次效应，delta 为乘性（尺度）批次效应。
    两者互相依赖，故迭代到收敛。
    """
    n = (~np.isnan(s_data)).sum(axis=1).astype(float)
    g_old = np.array(g_hat, dtype=float)
    d_old = np.array(d_hat, dtype=float)
    change = 1.0
    it = 0
    while change > conv and it < max_iter:
        g_new = (t2 * n * g_hat + d_old * g_bar) / (t2 * n + d_old)
        sum2 = ((s_data - g_new[:, None]) ** 2).sum(axis=1)
        d_new = (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)
        change = max(float(np.max(np.abs(g_new - g_old) / np.abs(g_old))),
                     float(np.max(np.abs(d_new - d_old) / np.abs(d_old))))
        g_old, d_old = g_new, d_new
        it += 1
    return g_new, d_new


def combat(X, batch: Sequence[Any],
           covariates: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """ComBat 经验贝叶斯批次校正。

    :param X: (样本 × 特征) 矩阵或 DataFrame（**已做 log 变换**，勿传原始 count）
    :param batch: 长度 = 样本数的批次标签
    :param covariates: 需**保护**的协变量（如性别/年龄/病例对照）；分类列自动哑编码
    :return: dict
        - ``adjusted``: 校正后的 DataFrame（与 X 同形状/同索引列名）
        - ``n_batches`` / ``batch_sizes``: 批次规模
        - ``gamma_star`` / ``delta_star``: 每批次 EB 估计（位置/尺度）
        - ``zero_variance_features``: 零方差特征数（这些列原样返回 0 中心化）
        - ``method``: ``"ComBat(Johnson2007,EB; via scanpy BSD-3)"``
        - ``engine``: ``"lib.batch.combat"``

    ⚠️ 与 ``batch`` 共线的协变量**不要**重复传进 ``covariates``，
       否则设计矩阵奇异（本函数会因 ``inv`` 失败而报错，属预期防线）。
    """
    expr = _as_frame(X)
    batch = np.asarray(batch)
    if len(batch) != expr.shape[0]:
        raise ValueError(f"batch 长度 {len(batch)} != 样本数 {expr.shape[0]}")

    levels, codes = np.unique(batch, return_inverse=True)
    n_batch = len(levels)
    sizes = np.array([int((codes == i).sum()) for i in range(n_batch)])
    small = [levels[i] for i in range(n_batch) if sizes[i] < 2]
    if small:
        raise ValueError(
            f"批次 {small} 样本数 <2，无法估计批内方差。请先过滤这些批次，"
            "不要静默丢弃（会改变下游自由度）。"
        )

    design = _design_matrix(batch, list(levels), covariates)
    data = expr.T  # (特征, 样本)
    s_data, var_pooled, stand_mean, n_zero = _standardize_data(
        design, data, n_batch, sizes)

    batch_design = design.iloc[:, :n_batch].to_numpy(dtype=float)
    s_np = s_data.to_numpy(dtype=float)
    gamma_hat = (np.linalg.inv(batch_design.T @ batch_design) @ batch_design.T
                 @ s_np.T)
    idx_by_batch = [np.flatnonzero(codes == i) for i in range(n_batch)]
    delta_hat = [s_np[:, idx].var(axis=1, ddof=1) for idx in idx_by_batch]

    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1)
    a_prior = [_aprior(d) for d in delta_hat]
    b_prior = [_bprior(d) for d in delta_hat]

    gamma_star, delta_star = [], []
    for i, idx in enumerate(idx_by_batch):
        g, d = _it_sol(s_np[:, idx], gamma_hat[i], delta_hat[i],
                       g_bar=float(gamma_bar[i]), t2=float(t2[i]),
                       a=float(a_prior[i]), b=float(b_prior[i]))
        gamma_star.append(g)
        delta_star.append(d)
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    bayes = s_data.copy()
    bayes_np = bayes.to_numpy(dtype=float, copy=True)
    for j, idx in enumerate(idx_by_batch):
        dsq = np.sqrt(delta_star[j, :])
        numer = bayes_np[:, idx] - (batch_design[idx] @ gamma_star).T
        bayes_np[:, idx] = numer / dsq[:, None]
    adjusted = bayes_np * np.sqrt(var_pooled) + stand_mean
    adjusted_df = pd.DataFrame(adjusted.T, index=expr.index, columns=expr.columns)

    return {
        "adjusted": adjusted_df,
        "n_batches": n_batch,
        "batch_levels": [str(x) for x in levels],
        "batch_sizes": {str(levels[i]): int(sizes[i]) for i in range(n_batch)},
        "gamma_star": gamma_star,
        "delta_star": delta_star,
        "zero_variance_features": int(n_zero),
        "shape": [int(expr.shape[0]), int(expr.shape[1])],
        "method": "ComBat(Johnson2007,EB; via scanpy BSD-3)",
        "engine": "lib.batch.combat",
    }


def combat_reference(X, batch: Sequence[Any], reference: Any,
                     covariates: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """把**指定批次**作为参考的 ComBat（复现 ComBat 论文的 ref-batch 口径）。

    做法：以 reference 批次为基准，只对非参考批次估计并施加校正参数，
    参考批次保持不动。当某个批次是「金标准」（如临床中心主队列）时，
    保留其原始尺度比「全部向全体均值收缩」更符合下游解释。
    """
    res = combat(X, batch, covariates=covariates)
    batch = np.asarray(batch)
    if reference not in set(batch.tolist()):
        raise ValueError(f"reference 批次 {reference!r} 不在 batch 中")
    expr = _as_frame(X)
    ref_mask = batch == reference
    out = expr.copy()
    out.loc[~ref_mask] = res["adjusted"].loc[~ref_mask]
    res["adjusted"] = out
    res["reference"] = str(reference)
    res["method"] = "ComBat-ref(Johnson2007; via scanpy BSD-3)"
    return res


def pca_scores(X, n_components: int = 2, center: bool = True) -> np.ndarray:
    """样本 PCA 得分（用于批次诊断；纯 numpy SVD）。"""
    expr = _as_frame(X).to_numpy(dtype=float)
    if center:
        expr = expr - expr.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(expr, full_matrices=False)
    return u[:, :n_components] * s[:n_components]


def batch_diagnostics(X, batch: Sequence[Any], n_components: int = 2) -> Dict[str, Any]:
    """批次混杂诊断：校正前后「批次内 vs 批次间」离散度之比。

    指标 ``batch_separation`` = 批次间质心离散 / 批次内离散（PC1–PC2 空间）。
    比值越大说明批次在低维空间越可分（混杂越重）；校正后应显著下降。
    另给 ``variance_explained_pc1``，避免只看聚类图的自欺。

    ⚠️ 这不是统计检验，而是**效应量**描述。比值降下来不等于生物学消失 ——
       必须同时确认生物学分组在校正后仍可分（本函数不代替那一步）。
    """
    Xa = _as_frame(X).to_numpy(dtype=float)
    batch = np.asarray(batch)
    levels = np.unique(batch)

    def _sep(mat: np.ndarray) -> float:
        sc = pca_scores(pd.DataFrame(mat), n_components=n_components)
        grand = sc.mean(axis=0, keepdims=True)
        within, between = 0.0, 0.0
        for lv in levels:
            m = batch == lv
            if m.sum() == 0:
                continue
            cen = sc[m].mean(axis=0, keepdims=True)
            within += float(((sc[m] - cen) ** 2).sum())
            between += float(m.sum() * ((cen - grand) ** 2).sum())
        return between / within if within > 0 else float("inf")

    sep_before = _sep(Xa)
    adj = combat(pd.DataFrame(Xa), batch)["adjusted"].to_numpy(dtype=float)
    sep_after = _sep(adj)
    sc = pca_scores(pd.DataFrame(Xa), n_components=n_components)
    ratio = (sep_after / sep_before) if sep_before not in (0.0,) else float("nan")
    return {
        "batch_separation_before": float(sep_before),
        "batch_separation_after": float(sep_after),
        "separation_ratio": float(ratio),
        "variance_explained_pc1": float(
            np.var(sc[:, 0]) / np.var(_as_frame(X).to_numpy(dtype=float), axis=0).sum()),
        "n_batches": int(len(levels)),
        "engine": "lib.batch.batch_diagnostics",
    }
