# PORTED-FILE · 来源 lilab-bcb/harmony-pytorch（**BSD-3-Clause**，2026-10-12 查 GitHub 官方 SPDX）
#   论文：Korsunsky I, Millard N, Fan J, et al.
#         "Fast, sensitive and accurate integration of single-cell data with Harmony"
#         *Nature Methods* 2019;16(12):1289-1296
#   对应上游源码：harmony/harmony.py
#              （harmonize / initialize_centroids / clustering / correction /
#                compute_objective / is_convergent_harmony / is_convergent_clustering）
#
# 移植方式：**算法逐句照抄，运行底座换掉**。
#   上游跑在 torch.Tensor 上，并用 sklearn.cluster.KMeans 初始化质心。
#   torch 与 sklearn 都**不能作为本 skill 的运行依赖内嵌**（体积/可用性均不允许），
#   故逐句翻译为 numpy；KMeans 改为自实现的 k-means++ + Lloyd。
#
# 逐块对照（上游 -> 本文件）：
#   Z_norm = normalize(Z, p=2, dim=1)                   -> _l2_normalize
#   normalize(R, p=1, dim=1)                            -> _l1_normalize
#   KMeans(...).fit(Z_norm) 取 cluster_centers_         -> _kmeans（k-means++ 初始化）
#   R = exp(-2/σ · (1 - Z_norm·Y_normᵀ))                -> _assign_soft
#   E = Pr_b ⊗ ΣR，O = ΦᵀR                              -> harmonize 内同名变量
#   留一块 / 补一块更新 O、E（bh：leftout + full）       -> clustering 内的 -= / +=
#   ω = Φ_in · ((E+1)/(O+1))^θᵀ                         -> 同上（逐元素幂再乘）
#   岭回归去批次 cluster-by-cluster（Woodbury 式闭式解） -> correction
#   目标 = kmeans 误差 + σ·熵 + σ·θ·(O·log((O+1)/(E+1))) -> compute_objective
#
# 与上游的**已知差异（如实标注，不得省略）**：
#   1. **底座不同**：上游 torch.Tensor（可 CUDA/MPS），本文件 numpy float64 CPU。
#      → 逐元素乘法/除法顺序一致，故逐位误差应在 float64 机器精度量级，
#        但**无法在本机与 torch 版对拍**（本机无 torch），只能与本文件的
#        性质测试比对（批次混合度上升、生物信号保留、同种子可复现）。
#   2. **质心初始化不同**：上游用 sklearn 的 KMeans（双精度 Lloyd + k-means++），
#      本文件自实现 k-means++ + Lloyd（惯性下降停止）。两者在同样种子下
#      **不会**给出逐位相同的初始质心 —— 但 Harmony 迭代本身会重估质心，
#      且目标是软分配不是硬分配，故下游结论（批次混合、信号保留）一致。
#   3. **一处比上游更稳的改写**：``exp(-2/σ(1-cos))`` 在 σ 很小或 clustering
#      初期可能整行下溢为 0 → 行 L1 归一化变成 0/0（NaN 静默扩散）。本文件先减
#      行最大值再 exp：因为随后要按行归一 + **逐元素**乘 ω，乘常数不影响结果，
#      因此**数学等价**且消除下溢。此差异已在测试中断言（σ=0.01 仍无 NaN）。
#   4. 上游依赖 psutil（取物理核数）与 threadpoolctl（限线程），均为性能设施，
#      本项目单进程确定性执行，直接去掉，不影响任何数值。
# ─────────────────────────────────────────────────────────────────────────────
"""Harmony 批次整合：**模糊 k-means + 多样性惩罚 + 岭回归校正**的交替迭代。

输入是降维后的 embedding（通常是 PCA，行=细胞列=维度），不是原始表达矩阵。

为什么不用 ComBat 顶替（本项目此前只有 ComBat）：
  * ComBat 只做**逐特征的线性平移/缩放**，不输出 embedding、不对读数是细胞的
    联合几何做对齐，下游聚类/邻居图仍是"每批一坨"；
  * Harmony 的迭代目标是**让每个 soft cluster 里各批次的细胞数趋于均匀**，
    这个"多样性惩罚"项（θ 加权）正是跨供体合并时要的东西；
  * 关键前提：**必须传入生物协变量无关的批次键**（如供体 ID），
    否则 θ 惩罚会把年龄信号当成批次效应一起消掉 —— 这是用错 Harmony 最常见的方式。

”这是常被误解的一点“：Harmony **不保证**保留任何协变量的信号。它只保证
"消除批次差异"这一侧。协变量是否被抹掉必须由 `lib.integration.fidelity` 实测，
不能靠相信算法。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── 归一化 ───────────────────────────────────────────────────────────────────
def _l2_normalize(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """沿行作 L2 归一化（torch 的 normalize(p=2, dim=1)）。零向量行保持零。"""
    n = np.linalg.norm(A, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(n > eps, A / np.where(n > eps, n, 1.0), 0.0)
    return out


def _l1_normalize(A: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """沿行作 L1 归一化（torch 的 normalize(p=1, dim=1)），结果按行和为 1。"""
    s = A.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(s) > eps, A / np.where(np.abs(s) > eps, s, 1.0),
                       1.0 / max(A.shape[1], 1))
    return out


# ── k-means（替 sklearn.cluster.KMeans）───────────────────────────────────────
def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ 初始化：D² 加权抽样，避免随机取点导致的空簇与慢收敛。"""
    n = X.shape[0]
    idx0 = int(rng.integers(n))
    centers = [X[idx0]]
    d2 = ((X - X[idx0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = float(d2.sum())
        if not np.isfinite(total) or total <= 0:
            # 所有点重合（退化输入）：退化为随机取点，后续 Lloyd 立即收敛
            centers.append(X[int(rng.integers(n))])
            continue
        probs = d2 / total
        j = int(rng.choice(n, p=probs))
        centers.append(X[j])
        d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(axis=1))
    return np.asarray(centers, dtype=float)


def _kmeans(X: np.ndarray, k: int, *,
            random_state: int = 0,
            n_init: int = 1, max_iter: int = 300,
            init: str = "plus") -> np.ndarray:
    """Lloyd 迭代的 k-means，返回 (k × d) 质心。

    :param init: ``plus``（k-means++，默认）或 ``random``（随机取 k 个点）
    :param n_init: 独立重复次数，取惯性最小的一次

    说明：只用于**初始化** Harmony 的软簇质心，所以惯性是否为全局最优不重要；
    重要的是**确定性**（同种子必得同一初值），否则整条管线不可复现。
    """
    n, d = X.shape
    k = int(min(k, n))
    if k < 1:
        raise ValueError("k 必须 >= 1")
    best_c, best_inertia = None, np.inf
    for rep in range(int(n_init)):
        rng = np.random.default_rng(int(random_state) + rep)
        if init == "plus":
            c = _kmeans_plus_plus(X, k, rng)
        else:
            c = X[rng.choice(n, size=k, replace=False)].astype(float)
        for _ in range(int(max_iter)):
            d2 = ((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)   # (n, k)
            lab = np.argmin(d2, axis=1)
            new = c.copy()
            for j in range(k):
                m = lab == j
                if m.any():
                    new[j] = X[m].mean(axis=0)
                else:
                    # 空簇：取距当前质心最远的点补位（比留着旧质心更稳）
                    new[j] = X[int(np.argmax(d2.min(axis=1)))]
            shift = float(np.abs(new - c).max())
            c = new
            if shift < 1e-9:
                break
        inertia = float(((X[:, None, :] - c[None, :, :]) ** 2).sum(axis=2).min(axis=1).sum())
        if inertia < best_inertia:
            best_c, best_inertia = c, inertia
    return np.asarray(best_c, dtype=float)


# ── 批次编码 ─────────────────────────────────────────────────────────────────
def _batch_codes(batch: Sequence[Any]) -> Tuple[np.ndarray, List[Any]]:
    """把任意可哈希的批次标签转成整数编码，返回 (codes, categories)。"""
    b = np.asarray(list(batch))
    if b.ndim != 1:
        raise ValueError(f"batch 必须是一维，收到形状 {b.shape}")
    cats: List[Any] = []
    seen: Dict[Any, int] = {}
    for v in b:
        # numpy 标量不可直接做 dict 键的相等比较（np.nan 等），转 python 原生
        key = v.item() if hasattr(v, "item") else v
        if key not in seen:
            seen[key] = len(cats)
            cats.append(key)
    codes = np.array([seen[v.item() if hasattr(v, "item") else v] for v in b], dtype=int)
    return codes, cats


def _one_hot(codes: np.ndarray, n_batches: int) -> np.ndarray:
    """one-hot（行=细胞，列=批次）。"""
    Phi = np.zeros((codes.size, int(n_batches)), dtype=float)
    Phi[np.arange(codes.size), codes] = 1.0
    return Phi


# ── 核心目标与更新 ───────────────────────────────────────────────────────────
def _assign_soft(Z_norm: np.ndarray, Y_norm: np.ndarray, sigma: float) -> np.ndarray:
    """``R = softmax_row(-2/σ · (1 - cos))`` 的行归一化结果（未乘多样性权重 ω）。

    与上游唯一的形式差异：先减行最大值再 exp（见文件头差异第 3 条）。
    """
    cos = Z_norm @ Y_norm.T
    logits = -(2.0 / float(sigma)) * (1.0 - np.clip(cos, -1.0, 1.0))
    logits -= logits.max(axis=1, keepdims=True)      # 数学等价，杜绝下溢
    return _l1_normalize(np.exp(logits))


def compute_objective(Y_norm: np.ndarray, Z_norm: np.ndarray, R: np.ndarray,
                      theta_row: np.ndarray, sigma: float,
                      O: np.ndarray, E: np.ndarray) -> float:
    """Harmony 目标 = kmeans 误差 + σ·熵 + σ·θ·(O·log((O+1)/(E+1)))。

    :param theta_row: 形状 (1, B) 的多样性惩罚权重
    """
    kmeans_error = float(np.sum(R * 2.0 * (1.0 - Z_norm @ Y_norm.T)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.where(R > 0, R * np.log(np.where(R > 0, R, 1.0)), 0.0)
    entropy_term = float(sigma) * float(-ent.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        div = O * np.log((O + 1.0) / (E + 1.0))
    diversity_penalty = float(sigma) * float((theta_row @ div).sum())
    return kmeans_error + entropy_term + diversity_penalty


def correction(X: np.ndarray, R: np.ndarray, Phi: np.ndarray,
               O: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """对每个 soft cluster 做**岭回归**去批次效应，返回校正后的 embedding。

    逐句照抄上游的 Woodbury 闭式解：先把 Φ 增广一列全 1（截距），
    再按 cluster k 用 ``(Φᵀ diag(R_k) Φ + Λ)⁻¹`` 的形式解岭回归，
    截距项系数置 0（**不**平移整体水平，只去掉批次偏移）。
    """
    n_cells = X.shape[0]
    n_clusters = R.shape[1]
    n_batches = Phi.shape[1]
    Phi_1 = np.hstack([np.ones((n_cells, 1)), Phi])      # (n, B+1)
    Z = X.copy().astype(float)
    P = np.eye(n_batches + 1)

    for k in range(n_clusters):
        O_k = O[:, k]
        N_k = float(O_k.sum())
        factor = 1.0 / (O_k + float(ridge_lambda))       # (B,)
        c = N_k + float(np.sum(-factor * O_k ** 2))
        c_inv = 1.0 / c

        P[0, 1:] = -factor * O_k
        P_t_B_inv = np.diag(np.concatenate([[c_inv], factor]).ravel())
        P_t_B_inv[1:, 0] = P[0, 1:] * c_inv
        inv_mat = P_t_B_inv @ P

        # 注意两个 X/Z 的分工（上游如此，改动会造成重复扣减）：
        #   · 系数 W 由**原始** X 算出（否则先修好的批次会把系数本身带着漂移，
        #     后续 cluster 再扣一次，等于按累计步长多次校正，会过冲）；
        #   · 修正量累加到 Z（迭代中被减的那个副本）。
        Phi_t_diag_R = Phi_1.T * R[:, k][None, :]        # (B+1, n)
        W = inv_mat @ (Phi_t_diag_R @ X)
        W[0, :] = 0.0                                    # 截距不参与修正
        Z -= Phi_t_diag_R.T @ W
    return Z


def clustering(Z_norm: np.ndarray, Pr_b: np.ndarray, Phi: np.ndarray,
               R: np.ndarray, theta_row: np.ndarray,
               tol: float, max_iter: int, sigma: float,
               block_size: int, rng: np.random.Generator,
               trace: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一轮 Harmony 的聚类步：重估质心 -> 分块留一更新 R -> 判收敛。

    返回更新后的 ``(R, E, O)``。分块更新（block_size）是为了**留一**：
    更新某一块之前，先把它自己对 O、E 的贡献扣掉，避免用自己预测自己。
    """
    n_cells = Z_norm.shape[0]
    objectives: List[float] = []

    # O、E 必须与**当前** R 一致：这里重算一次而不是相信调用方传进来的值。
    # 不同步会让"扣掉本块贡献"扣成另一个量级，留一步就废，且不会报错只给错结果。
    O = Phi.T @ R
    E = Pr_b @ np.sum(R, axis=0, keepdims=True)

    for _ in range(int(max_iter)):
        # 1) 由当前软分配重估质心，再 L2 归一
        Y = R.T @ Z_norm
        Y_norm = _l2_normalize(Y)

        # 2) 分块留一更新 R
        idx_list = rng.permutation(n_cells)
        pos = 0
        while pos < len(idx_list):
            idx_in = idx_list[pos:pos + block_size]
            R_in = R[idx_in, :]
            Phi_in = Phi[idx_in, :]

            O -= Phi_in.T @ R_in                                    # (B, K)
            E -= Pr_b @ np.sum(R_in, axis=0, keepdims=True)          # (B, K)

            R_in = _assign_soft(Z_norm[idx_in, :], Y_norm, sigma)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                omega = Phi_in @ np.power((E + 1.0) / (O + 1.0), theta_row.T)
            R_in = _l1_normalize(R_in * omega)
            R[idx_in, :] = R_in

            O += Phi_in.T @ R_in
            E += Pr_b @ np.sum(R_in, axis=0, keepdims=True)
            pos += block_size

        objectives.append(compute_objective(Y_norm, Z_norm, R, theta_row, sigma, O, E))
        if len(objectives) >= 4:
            obj_old = sum(objectives[-2 - i] for i in range(3))
            obj_new = sum(objectives[-1 - i] for i in range(3))
            if (obj_old - obj_new) < tol * abs(obj_old):
                break

    if trace is not None:
        trace.extend(objectives)
    return R, E, O


def harmonize(X: np.ndarray, batch: Sequence[Any], *,
              n_clusters: Optional[int] = None,
              max_iter_harmony: int = 10,
              max_iter_clustering: int = 200,
              tol_harmony: float = 1e-4,
              tol_clustering: float = 1e-5,
              ridge_lambda: float = 1.0,
              sigma: float = 0.1,
              block_proportion: float = 0.05,
              init_centroids_method: str = "default",
              theta: float = 2.0,
              tau: int = 0,
              random_state: int = 0,
              stop_on_increase: bool = True,
              return_diagnostics: bool = False):
    """Harmony 整合。行=细胞，列=embedding 维度（通常是 PCA）。

    :param X: (细胞 × 维度) 降维矩阵
    :param batch: 每个细胞的批次标签（**通常传供体 ID**）
    :param n_clusters: soft cluster 数，默认 ``min(100, n/30)``
    :param theta: 多样性惩罚权重，越大越强硬地要求每个簇里各批次均衡
    :param stop_on_increase: 目标上升时是否算收敛（见下方「已知的早停陷阱」）
    :param return_diagnostics: True 时一并返回 objective 轨迹与停止原因
    :returns: 校正后的 embedding；``return_diagnostics`` 时为 ``(Z, diag)``

    **已知的早停陷阱（上游同款，实测会咬人）**：
      ``objectives_harmony`` 各轮是在**不同的** Z_norm 上算出来的 —— 上一轮的
      correction 已经改了 embedding，下一轮 clustering 再在改过的 embedding 上
      算目标。**跨轮目标值不可比**，而 ``is_convergent_harmony`` 比的正是相邻两轮。
      批次偏移大时目标几乎必然在某轮上升，于是 ``(old - new) < tol·|old|`` 立即
      成立 → **第一次校正后就收工**。实测：默认参数下 2 批次大偏移只有
      mix 0.891→0.777（没整干净），跑满 10 轮才到 0.568，且生物纯度反而更高。
      本文件不改变上游默认，但把 ``stopped_by`` 报出来；包装层
      ``lib.integration.harmonize`` 默认关掉这个早停（``stop_on_increase=False``）。

    口径提醒（两个常被搞混的地方）：
      * 输入的 X **必须已做过 per-batch 无关的标准化**（scanpy 的 scale），
        因为 Harmony 的余弦相似建立在行 L2 归一上，量纲不一致等于给某些维度加权；
      * ``batch`` 里**不要**放生物协变量。要检验协变量是否被保住，
        用 ``lib.integration.fidelity``，别看 Harmony 自己的输出。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X 必须是二维 (细胞 × 维度)，收到形状 {X.shape}")
    n_cells, n_dims = X.shape
    if n_cells < 2:
        raise ValueError("细胞数必须 >= 2")
    if not np.all(np.isfinite(X)):
        raise ValueError("X 含 NaN/Inf —— Harmony 的岭回归会把 NaN 扩散到整个 embedding")

    codes, cats = _batch_codes(batch)
    n_batches = len(cats)
    if n_batches < 2:
        raise ValueError(f"只有 {n_batches} 个批次，无需整合（Harmony 要 >= 2 个批次）")
    if n_batches > n_cells:
        raise ValueError(f"批次数 {n_batches} 超过细胞数 {n_cells} —— 每个批次一个细胞时"
                         "批次效应与生物效应不可分离，整合结果无意义")

    if not (0 < float(block_proportion) <= 1):
        raise ValueError(f"block_proportion 必须在 (0, 1]，收到 {block_proportion}")
    if float(sigma) <= 0:
        raise ValueError(f"sigma 必须 > 0，收到 {sigma}")
    if init_centroids_method not in ("default", "harmony-paper"):
        raise ValueError("init_centroids_method 只能是 default / harmony-paper")

    counts = np.bincount(codes, minlength=n_batches)
    if counts.min() < 2:
        raise ValueError(
            f"存在只含 1 个细胞的批次（批次数={n_batches}）—— Harmony 的岭回归要"
            "在该批次内部估计偏移量，单点无法估计，只会把噪声当成批次效应写进 embedding。"
            "请把稀有批次合并或剔除后再整合")
    N_b = counts.astype(float)
    Pr_b = (N_b / n_cells).reshape(-1, 1)              # (B, 1)
    Phi = _one_hot(codes, n_batches)                   # (n, B)

    if n_clusters is None:
        n_clusters = int(min(100, max(1, n_cells / 30.0)))
    n_clusters = int(max(1, min(int(n_clusters), n_cells)))

    theta_vec = np.full(n_batches, float(theta))
    if int(tau) > 0:
        # 小批次放宽多样性惩罚（否则会被强制去抹平它本身很弱的批次结构）
        theta_vec = theta_vec * (1.0 - np.exp(-N_b / (n_clusters * float(tau))) ** 2)
    theta_row = theta_vec.reshape(1, -1)               # (1, B)

    block_size = max(1, int(round(n_cells * float(block_proportion))))

    Z_norm = _l2_normalize(X)
    rng = np.random.default_rng(int(random_state))

    # ── 初始化：KMeans 质心 -> 软分配 R -> E/O
    if init_centroids_method == "harmony-paper":
        Y = _kmeans(Z_norm, n_clusters, random_state=random_state,
                    n_init=10, max_iter=25, init="random")
    else:
        Y = _kmeans(Z_norm, n_clusters, random_state=random_state,
                    n_init=1, max_iter=300, init="plus")
    Y_norm = _l2_normalize(Y)
    R = _assign_soft(Z_norm, Y_norm, sigma)
    E = Pr_b @ np.sum(R, axis=0, keepdims=True)        # (B, K)
    O = Phi.T @ R                                      # (B, K)

    objectives_harmony = [compute_objective(Y_norm, Z_norm, R, theta_row, sigma, O, E)]
    harmony_trace: List[List[float]] = []
    cluster_trace: List[float] = []

    stopped_by = "max_iter"
    for i in range(int(max_iter_harmony)):
        R, E, O = clustering(Z_norm, Pr_b, Phi, R, theta_row,
                             tol=float(tol_clustering), max_iter=int(max_iter_clustering),
                             sigma=float(sigma), block_size=block_size, rng=rng,
                             trace=cluster_trace)
        Z_hat = correction(X, R, Phi, O, ridge_lambda)
        if not np.all(np.isfinite(Z_hat)):
            raise ValueError("校正后出现 NaN/Inf：通常是 ridge_lambda 太小或批次极度不平衡")
        Z_norm = _l2_normalize(Z_hat)

        # 收敛判据用 clustering 自己最后一轮的目标值（与上游一致：
        # objectives_harmony.append(objectives_clustering[-1])），
        # 不另算一套 —— 两套定义混用会让 tol 失去意义。
        if cluster_trace:
            objectives_harmony.append(cluster_trace[-1])
        harmony_trace.append(list(cluster_trace))
        cluster_trace = []

        if len(objectives_harmony) >= 2:
            obj_old = objectives_harmony[-2]
            obj_new = objectives_harmony[-1]
            increased = obj_new > obj_old
            # 目标**上升**不算收敛：跨轮不在同一个 embedding 上（见 docstring），
            # 上升只说明 embedding 被大幅改动，不代表已经对齐好了。
            converged = (not increased) and ((obj_old - obj_new) < float(tol_harmony) * abs(obj_old))
            if increased and not bool(stop_on_increase):
                continue
            if converged:
                stopped_by = "converged"
                break
            if increased:
                stopped_by = "objective_increased"
                break

    if return_diagnostics:
        diag = {
            "objectives_harmony": objectives_harmony,
            "clustering_objectives": harmony_trace,
            "n_clusters": int(n_clusters),
            "n_batches": int(n_batches),
            "batch_categories": cats,
            "theta": theta_vec,
            "sigma": float(sigma),
            "ridge_lambda": float(ridge_lambda),
            "stopped_by": stopped_by,
            "n_harmony_iter_run": int(len(harmony_trace)),
            "soft_assignment": R,
            "cluster_usage": R.sum(axis=0),
            "engine": "vendor._harmony.harmonize",
            "inference": ("算法逐句照抄 lilab-bcb/harmony-pytorch（BSD-3），"
                          "底座由 torch 换 numpy、KMeans 换自实现 k-means++；"
                          "未与 torch 版对拍（本机无 torch）"),
        }
        return Z_hat, diag
    return Z_hat
