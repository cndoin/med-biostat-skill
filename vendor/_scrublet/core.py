# PORTED-FILE · 来源 swolock/scrublet（MIT）
#   源文件 src/scrublet/scrublet.py（class Scrublet）、src/scrublet/helper_functions.py
#   另取自 scikit-image（BSD-3）filters/thresholding.py::threshold_minimum（自动阈值）
#
# 移植方式：**算法照抄、依赖换底**。上游把依赖绑死在 sklearn / annoy / numba /
#   matplotlib / scikit-image 上；本文件只保留统计计算，全部改成 numpy + scipy，
#   并把 sparse 矩阵接口改为 dense（(细胞 × 基因)）。
# 逐函数对照（上游 -> 本文件）：
#   helper_functions.tot_counts_norm        -> total_counts_normalize
#   helper_functions.runningquantile        -> runningquantile
#   helper_functions.get_vscores            -> vscores
#   helper_functions.filter_genes           -> filter_genes
#   Scrublet.simulate_doublets              -> simulate_doublets
#   helper_functions.sparse_zscore          -> gene_zscore
#   pipeline_pca（sklearn PCA）             -> pca_scores（numpy SVD）
#   helper_functions.get_knn_graph          -> knn_indices（scipy cKDTree）
#   Scrublet._nearest_neighbor_classifier   -> doublet_scores
#   Scrublet.call_doublets                  -> call_doublets
#   skimage.filters.threshold_minimum       -> threshold_minimum
#
# 与上游的**已知差异（如实标注，不得省略）**：
#   1. PCA 用 numpy SVD 代替 sklearn PCA。数值等价，但 **PCA 主成分符号本就不定**
#      （sklearn 固定 sign，numpy SVD 不固定）——符号翻转不影响距离与邻居。
#   2. 近邻用 scipy cKDTree 代替 sklearn NearestNeighbors（默认 algorithm='auto'）。
#      两者在**距离并列**时返回的邻居顺序可能不同，进而使 doublet score 在
#      极少数细胞上出现微小差异。这不是 bug，是等价实现的边界差异。
#   3. 自动阈值照抄 skimage 的 threshold_minimum（迭代平滑直方图直至只剩两个
#      局部极大值，取其间最小值）。上游用 ndi.uniform_filter1d 平滑，本文件同样
#      调用 scipy.ndimage.uniform_filter1d（同一函数，scipy 自带）。
#   4. 未移植上游的 `exclude_dominant_frac`（归一化时排除高丰度基因）与全部绘图。
#   5. 未使用 numba；纯 numpy，速度低于上游编译版，结果口径一致。
"""双胞体检测（Scrublet 算法，Wolock, Macosko & Gifford 2019, *Cell Systems* 8(4):329-337）。

原理：真实细胞 + **模拟双胞体**（随机两细胞 counts 相加）一起嵌入低维空间，
每个细胞的近邻里模拟双胞体占比越高、越可能是双胞体。

**为什么必须做**：双胞体在轨迹推断里会伪装成"过渡态细胞"，
直接污染伪时间排序与年龄轨迹——不是"可选的清洗步骤"。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as _ndi
import scipy.optimize as _opt
from scipy.spatial import cKDTree

__all__ = [
    "threshold_minimum", "runningquantile", "total_counts_normalize",
    "vscores", "filter_genes", "simulate_doublets", "gene_zscore",
    "pca_scores", "knn_indices", "doublet_scores", "call_doublets",
]


# ── 基因过滤：above-Poisson 噪声统计量 ────────────────────────────────────────
def runningquantile(x: np.ndarray, y: np.ndarray, p: float, nBins: int):
    """照抄 scrublet.helper_functions.runningquantile。

    在 x 的分箱内取 y 的第 p 百分位；空箱沿用前一箱的值（首箱为空则置 NaN）。
    """
    ind = np.argsort(x)
    x = np.asarray(x)[ind]
    y = np.asarray(y)[ind]

    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0] + dx / 2, x[-1] - dx / 2, nBins)
    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind2 = np.nonzero((x >= xOut[i] - dx / 2) & (x < xOut[i] + dx / 2))[0]
        if len(ind2) > 0:
            yOut[i] = np.percentile(y[ind2], p)
        else:
            yOut[i] = yOut[i - 1] if i > 0 else np.nan
    return xOut, yOut


def vscores(E: np.ndarray, min_mean: float = 0, nBins: int = 50,
            fit_percentile: float = 0.1, error_wt: float = 1):
    """照抄 scrublet.helper_functions.get_vscores（dense 版）。

    v-score = 高于泊松噪声的变异统计量（Klein et al. 2015, *Cell*）；
    先按 v-score 与表达门槛筛基因，再做 PCA——这是 scrublet 预处理的第一步。

    :return: (v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b)
    """
    E = np.asarray(E, dtype=float)
    mu_gene = E.mean(axis=0)
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]

    tmp = E[:, gene_ix] ** 2
    var_gene = tmp.mean(axis=0) - mu_gene ** 2
    del tmp
    FF_gene = var_gene / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    keep = ~np.isnan(y)
    x = x[keep]
    y = y[keep]

    def gLog(inp):
        return np.log(inp[1] * np.exp(-inp[0]) + inp[2])

    h, b = np.histogram(np.log(FF_gene[mu_gene > 0]), bins=200)
    b = b[:-1] + np.diff(b) / 2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))

    def errFun(b2):
        return np.sum(abs(gLog([x, c, b2]) - y) ** error_wt)

    b_hat = _opt.fmin(func=errFun, x0=[0.1], disp=False)
    b_hat = float(np.ravel(b_hat)[0])
    a = c / (1 + b_hat) - 1

    v_scores = FF_gene / ((1 + a) * (1 + b_hat) + b_hat * mu_gene)
    CV_eff = np.sqrt((1 + a) * (1 + b_hat) - 1)
    CV_input = np.sqrt(b_hat)
    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b_hat


def filter_genes(E: np.ndarray, min_vscore_pctl: float = 85,
                 min_counts: float = 3, min_cells: int = 3) -> np.ndarray:
    """照抄 scrublet.helper_functions.filter_genes：返回保留的**基因索引**。

    两个条件同时满足才保留：
      ① 至少在 ``min_cells`` 个细胞里 ≥ ``min_counts``；
      ② v-score ≥ 第 ``min_vscore_pctl`` 百分位。
    """
    E = np.asarray(E, dtype=float)
    Vscores, _CVe, _CVi, gene_ix, _mu, _FF, _a, _b = vscores(E)
    ix2 = Vscores > 0
    Vscores = Vscores[ix2]
    gene_ix = gene_ix[ix2]
    min_vscore = np.percentile(Vscores, min_vscore_pctl)
    ix = (((E[:, gene_ix] >= min_counts).sum(0) >= min_cells) & (Vscores >= min_vscore))
    return gene_ix[ix]


# ── 归一化与标准化 ────────────────────────────────────────────────────────────
def total_counts_normalize(E: np.ndarray, target_total: Optional[float] = None,
                           total_counts: Optional[Sequence[float]] = None) -> np.ndarray:
    """照抄 scrublet.helper_functions.tot_counts_norm（dense、去 exclude_dominant_frac）。

    每行（细胞）缩放到 ``target_total``；``target_total`` 缺省取各行总计数的均值
    ——**这是上游的默认口径**，换成固定值（如 1e6）会改变嵌入尺度。
    """
    E = np.asarray(E, dtype=float)
    if total_counts is None:
        tots_use = E.sum(axis=1)
    else:
        tots_use = np.asarray(total_counts, dtype=float).copy()

    if target_total is None:
        target_total = float(np.mean(tots_use))

    safe = np.where(tots_use == 0, 1.0, tots_use)
    return E * (float(target_total) / safe)[:, None]


def gene_zscore(E: np.ndarray, gene_mean: np.ndarray, gene_stdev: np.ndarray) -> np.ndarray:
    """照抄 scrublet.helper_functions.sparse_zscore：按基因标准化（用**观测细胞**的均值/标准差）。"""
    sd = np.where(np.asarray(gene_stdev) > 0, np.asarray(gene_stdev), 1.0)
    return (np.asarray(E, dtype=float) - np.asarray(gene_mean)) / sd


# ── 降维与近邻 ───────────────────────────────────────────────────────────────
def pca_scores(X: np.ndarray, n_comps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """numpy SVD 版的 PCA（替代 sklearn.decomposition.PCA）。

    :return: (scores, components, singular_values)
        ``scores = Xc @ components.T``，与 sklearn 的 ``pca.transform`` 同式。
        ⚠️ 主成分符号不固定（sklearn 做 sign flip，这里不做）——符号翻转
        不改变距离与近邻，但会让"载荷的符号"不同，**解读载荷时不要假设符号**。
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(n_comps, len(S)))
    return U[:, :k] * S[:k], Vt[:k], S[:k]


def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """k 近邻索引（scipy cKDTree 替代 sklearn NearestNeighbors）。

    ⚠️ 结果**包含自身**（与 sklearn 一致）；距离并列时顺序可能与 sklearn 不同。
    """
    X = np.asarray(X, dtype=float)
    k = int(max(1, k))
    tree = cKDTree(X)
    _d, idx = tree.query(X, k=k)
    if k == 1:
        idx = idx[:, None]
    return np.asarray(idx, dtype=int)


# ── 双胞体模拟、打分与判定 ────────────────────────────────────────────────────
def simulate_doublets(E_obs: np.ndarray, total_counts_obs: Sequence[float],
                      sim_doublet_ratio: float = 2.0, seed: int = 0,
                      subsampling: float = 1.0):
    """照抄 Scrublet.simulate_doublets：随机配对相加得到模拟双胞体。

    :param subsampling: <1 时对合并后的 UMI 再二项降采样（上游同名参数）
    :return: (E_sim, total_counts_sim, pair_index)
    """
    E_obs = np.asarray(E_obs, dtype=float)
    tots = np.asarray(total_counts_obs, dtype=float)
    n_obs = E_obs.shape[0]
    n_sim = int(n_obs * sim_doublet_ratio)

    # 上游用 np.random.seed + np.random.randint（legacy 全局状态）。
    # RandomState(seed) 与 legacy 序列**完全相同**，所以配对可复现、与上游一致。
    rng = np.random.RandomState(int(seed))
    pair_ix = rng.randint(0, n_obs, size=(n_sim, 2))

    E1 = E_obs[pair_ix[:, 0], :]
    E2 = E_obs[pair_ix[:, 1], :]
    tots1 = tots[pair_ix[:, 0]]
    tots2 = tots[pair_ix[:, 1]]

    if subsampling < 1:
        merged = E1 + E2
        rng2 = np.random.RandomState(int(seed))
        merged = rng2.binomial(np.round(merged).astype(int), subsampling).astype(float)
        current = merged.sum(axis=1)
        unsampled = (tots1 + tots2) - current
        down = rng2.binomial(np.round(np.maximum(unsampled, 0)).astype(int), subsampling)
        total_sim = current + down
        E_sim = merged
    else:
        E_sim = E1 + E2
        total_sim = tots1 + tots2
    return E_sim, np.asarray(total_sim, dtype=float), pair_ix


def doublet_scores(manifold_obs: np.ndarray, manifold_sim: np.ndarray, k: int = 40,
                   exp_doublet_rate: float = 0.1, stdev_doublet_rate: float = 0.03):
    """照抄 Scrublet._nearest_neighbor_classifier。

    对每个（观测或模拟）细胞，看它的 k 近邻里模拟双胞体占比 q，反解"该细胞的
    真实双胞体概率" Ld：

        q = (n_sim_neigh + 1) / (N + 2)          ← 贝叶斯平滑（加 1 / 加 2）
        Ld = q·ρ/r / (1 − ρ − q·(1 − ρ − ρ/r))    ← ρ=期望双胞体率, r=模拟/观测比

    :return: dict（obs/sim 的分数与标准误）
    """
    manifold_obs = np.asarray(manifold_obs, dtype=float)
    manifold_sim = np.asarray(manifold_sim, dtype=float)
    manifold = np.vstack((manifold_obs, manifold_sim))
    doub_labels = np.concatenate((np.zeros(manifold_obs.shape[0], dtype=int),
                                  np.ones(manifold_sim.shape[0], dtype=int)))
    n_obs = int(np.sum(doub_labels == 0))
    n_sim = int(np.sum(doub_labels == 1))

    k_adj = int(round(k * (1 + n_sim / float(n_obs))))
    neighbors = knn_indices(manifold, k=k_adj)

    doub_neigh_mask = doub_labels[neighbors] == 1
    n_sim_neigh = doub_neigh_mask.sum(1)
    n_obs_neigh = doub_neigh_mask.shape[1] - n_sim_neigh

    rho = float(exp_doublet_rate)
    r = n_sim / float(n_obs)
    nd = n_sim_neigh.astype(float)
    N = float(k_adj)

    q = (nd + 1) / (N + 2)
    Ld = q * rho / r / (1 - rho - q * (1 - rho - rho / r))

    se_q = np.sqrt(q * (1 - q) / (N + 3))
    se_rho = float(stdev_doublet_rate)
    se_Ld = (q * rho / r / (1 - rho - q * (1 - rho - rho / r)) ** 2
             * np.sqrt((se_q / q * (1 - rho)) ** 2 + (se_rho / rho * (1 - q)) ** 2))

    return {
        "scores_obs": Ld[doub_labels == 0],
        "scores_sim": Ld[doub_labels == 1],
        "errors_obs": se_Ld[doub_labels == 0],
        "errors_sim": se_Ld[doub_labels == 1],
        "k_adj": int(k_adj),
        "n_obs_neigh_mean": float(np.mean(n_obs_neigh)),
    }


def _find_local_maxima_idx(hist: np.ndarray):
    """照抄 skimage thresholding 内嵌函数（不能沿用 scipy.signal.argrelmax——平台期会失效）。"""
    maximum_idxs = []
    direction = 1
    for i in range(hist.shape[0] - 1):
        if direction > 0:
            if hist[i + 1] < hist[i]:
                direction = -1
                maximum_idxs.append(i)
        else:
            if hist[i + 1] > hist[i]:
                direction = 1
    return maximum_idxs


def threshold_minimum(values: Sequence[float], nbins: int = 256,
                      max_num_iter: int = 10000) -> float:
    """照抄 skimage.filters.threshold_minimum（BSD-3）。

    直方图迭代平滑（3 点均匀滤波）直到只剩 **2 个局部极大值**，
    取两者之间的最小值为阈值。

    ⚠️ 这是 **advisory**：scrublet 官方文档明确要求人工核对阈值直方图后
    可用 ``call_doublets(threshold=...)`` 覆盖。找不到双峰时抛错而不是瞎给一个值。
    """
    values = np.asarray(values, dtype=float)
    counts, bin_edges = np.histogram(values, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    smooth_hist = counts.astype("float32", copy=False)
    maximum_idxs = []
    counter = 0
    for counter in range(int(max_num_iter)):
        smooth_hist = _ndi.uniform_filter1d(smooth_hist, 3)
        maximum_idxs = _find_local_maxima_idx(smooth_hist)
        if len(maximum_idxs) < 3:
            break

    if len(maximum_idxs) != 2:
        raise RuntimeError("Unable to find two maxima in histogram")
    if counter == int(max_num_iter) - 1:
        raise RuntimeError("Maximum iteration reached for histogram smoothing")

    threshold_idx = int(np.argmin(smooth_hist[maximum_idxs[0]:maximum_idxs[1] + 1]))
    return float(bin_centers[maximum_idxs[0] + threshold_idx])


def call_doublets(scores_obs: np.ndarray, scores_sim: np.ndarray,
                  errors_obs: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
    """照抄 Scrublet.call_doublets：阈值判定 + 三项比例。

    ``threshold=None`` 时用 ``threshold_minimum(scores_sim)`` 自动定阈。
    """
    scores_obs = np.asarray(scores_obs, dtype=float)
    scores_sim = np.asarray(scores_sim, dtype=float)
    errors_obs = np.asarray(errors_obs, dtype=float)

    if threshold is None:
        threshold = threshold_minimum(scores_sim)

    z = (scores_obs - threshold) / errors_obs
    predicted = scores_obs > threshold
    detected_rate = float(predicted.sum() / len(scores_obs))
    detectable_fraction = float((scores_sim > threshold).sum() / len(scores_sim))
    overall = (detected_rate / detectable_fraction) if detectable_fraction > 0 else float("nan")

    return {
        "predicted_doublets": predicted,
        "z_scores": z,
        "threshold": float(threshold),
        "detected_doublet_rate": detected_rate,
        "detectable_doublet_fraction": detectable_fraction,
        "overall_doublet_rate": overall,
        "threshold_rule": ("threshold_minimum(scores_sim)（自动）" if threshold is not None
                           else "用户指定"),
    }
