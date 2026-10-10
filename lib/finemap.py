"""分析法 ⑪：统计精细定位（fine-mapping）——ABF + SuSiE（IBSS 单效应回归）。

为什么不是「最显著的那个 SNP 就是因果变异」：
LD 让一群 SNP 的 P 值高度相关，最显著的那个往往只是**最好的标签**，
真正的因果变异可能在可信集（credible set）里的任何位置。

两个方法，回答的粒度不同：
  · `abf`   — 单 causal variant 假设下的贝叶斯因子（Wakefield），给每个 SNP 后验概率
  · `susie` — SuSiE（Wang et al. 2020, JRSS-B）：允许**多个**因果变异，
              用迭代贝叶斯逐步选择（IBSS）逐个拟合单效应回归，输出多个可信集
  · `posterior_from_p` — 无个体数据时，用汇总统计 + LD 的近似（P 值 → 后验）

⚠️ **必须给 LD 矩阵**。没有 LD 的精细定位是自欺欺人：
   把一堆连锁 SNP 当成独立证据，后验会被稀释得毫无意义。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import stats as _st


def abf(beta: Sequence[float], se: Sequence[float],
        prior_sd: float = 0.15, snp: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Wakefield 近似贝叶斯因子（ABF），单因果变异假设。

    ABF = sqrt(V/(V+se²)) · exp( z²/2 · V/(V+se²) )，其中 V = prior_sd² 为先验方差。
    后验概率 = ABF_i / Σ ABF_j（在区间内归一化）。

    :param prior_sd: 效应量先验的标准差；**不同性状要不同**（常见 0.15 用于连续性状，
                     罕见二分类性状需要更大的值）。默认 0.15，须随结果报告。
    """
    b = np.asarray(beta, dtype=float)
    s = np.asarray(se, dtype=float)
    if np.any(s <= 0):
        raise ValueError("se 必须为正")
    V = float(prior_sd) ** 2
    z = b / s
    ratio = V / (V + s ** 2)
    log_abf = 0.5 * np.log(ratio) + 0.5 * z ** 2 * ratio
    m = np.max(log_abf)
    w = np.exp(log_abf - m)
    post = w / w.sum()
    order = np.argsort(-post)
    cs = []
    cum = 0.0
    for i in order:
        cs.append(i)
        cum += post[i]
        if cum >= 0.95:
            break
    return {
        "posterior": post,
        "log_abf": log_abf,
        "credible_set": list(cs),
        "credible_set_coverage": float(cum),
        "credible_set_snps": None if snp is None else [str(snp[i]) for i in cs],
        "prior_sd": float(prior_sd),
        "assumption": "区域内**恰好一个**因果变异（多因果时应用 SuSiE）",
    }


def susie(beta: Sequence[float], se: Sequence[float], ld: np.ndarray,
          L: int = 5, prior_variance: float = 0.2, max_iter: int = 100,
          tol: float = 1e-7, estimate_prior: bool = True,
          snp: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """SuSiE：迭代贝叶斯逐步选择（IBSS），允许多个因果变异。

    模型：b̂ = Σ_{l=1..L} R · b_l + e，每个单效应 b_l 只有一个非零元素，
    先验 b_l ~ N(0, V)。用变分/IBSS 交替更新每个单效应，输出 L 个可信集。

    :param beta: 边际效应估计（z 统计量的标准化形式），长度 p
    :param ld: p×p 的 LD 相关矩阵（**同一批 SNP、同一顺序**）
    :param L: 假定的因果变异个数上限
    :return: 每个单效应的后验（pip）、可信集列表、收敛信息

    实现口径说明：这里用「z 统计量 + LD」的 SuSiE-ss 形式
    （等价于 susieR 的 `susie_rss`，不需要个体数据）。
    """
    z = np.asarray(beta, dtype=float) / np.asarray(se, dtype=float)
    R = np.asarray(ld, dtype=float)
    p = z.shape[0]
    if R.shape != (p, p):
        raise ValueError(f"LD 矩阵必须是 {p}x{p}，收到 {R.shape}")
    L = int(min(L, p))
    V = float(prior_variance)
    # SuSiE-ss 的 IBSS 在 **z 空间**迭代。模型是 ẑ ~ N(R·b, R)，
    # 所以第 l 个单效应的残差是 r = ẑ − R·Σ_{m≠l} α_m（**z 空间**，不要再左乘 R）。
    # 实测把残差放到 Xty=R·z 空间会让 PIP 锁死在错的位点上（真因果 5 号被判成 28 号）。
    # 单效应的 logBF_j = −½·log(1+V·R_jj) + ½·V·r_j²/(1+V·R_jj)
    # （与 susieR::susie_rss 的形式只差一个与 j 无关的常数，归一化后等价）。
    # ⚠️ α_l 是**概率**（和为 1），**不是效应大小**。残差里要减掉的是
    # R·Σ_m(α_m ⊙ μ_m)，μ_m 是单效应的后验均值 μ_j = V·r_j/(1+V·R_jj)。
    # 直接减 R·α 等于把效应大小固定成 1 —— 残差减不干净，第二个因果位点
    # 永远冒不出来（实测双因果场景 4 个可信集全是 [5]，20 号位点找不到）。
    alpha = np.zeros((L, p))
    mu = np.zeros((L, p))
    b_eff = np.zeros((L, p))          # α ⊙ μ，各单效应的效应向量
    d = np.diag(R)
    for it in range(max_iter):
        alpha_old = alpha.copy()
        total_b = b_eff.sum(axis=0)
        for l in range(L):
            r = z - R @ (total_b - b_eff[l])
            logbf = _single_effect_logbf(r, R, V)
            m_ = np.max(logbf)
            w = np.exp(logbf - m_)
            alpha[l] = w / w.sum()
            mu[l] = V * r / (1.0 + V * d)          # 单效应后验均值
            new_b = alpha[l] * mu[l]
            # 顺序更新：第 l 个定下后立刻并进 total_b，让第 l+1 个在
            # 「已减掉 l 的信号」的残差上找位点（否则 L 个效应完全对称）
            total_b = total_b - b_eff[l] + new_b
            b_eff[l] = new_b
        if estimate_prior:
            # 简约的经验先验更新：V = Σ_l Σ_j α_lj · (z_j² - 1) / p（加权矩估计）
            num = float(np.sum(alpha * (z ** 2 - 1.0)[None, :]))
            V = max(num / max(np.sum(alpha), 1e-9), 1e-4)
        if np.max(np.abs(alpha - alpha_old)) < tol:
            break
    pip = np.clip(alpha.sum(axis=0), 0, 1)
    sets = []
    for l in range(L):
        idx = np.argsort(-alpha[l])
        cum, members = 0.0, []
        for i in idx:
            members.append(int(i))
            cum += float(alpha[l, i])
            if cum >= 0.95 or len(members) >= p:
                break
        if alpha[l].max() < 1e-6:
            continue
        sets.append({
            "index": members,
            "snps": None if snp is None else [str(snp[i]) for i in members],
            "coverage": float(cum),
            "min_purity": float(_cs_purity(R, members)),
            "top_pip": float(alpha[l].max()),
        })
    # 多余的单效应会分散到噪声上（top_pip 很低），必须显式区分「真信号」与
    # 「填空的效应」——否则会把噪声集合也当成发现报出去。
    reliable = [s for s in sets if s["top_pip"] >= 0.5]
    return {
        "pip": pip,
        "alpha": alpha,
        "credible_sets": sets,
        "n_effects": len(sets),
        "n_credible_sets_reliable": len(reliable),
        "reliable_credible_sets": reliable,
        "L": L,
        "prior_variance_estimated": V if estimate_prior else float(prior_variance),
        "method": "SuSiE-ss（IBSS，z + LD），等价于 susieR::susie_rss",
    }


def _single_effect_logbf(r: np.ndarray, R: np.ndarray, V: float) -> np.ndarray:
    """单效应回归的对数贝叶斯因子（对每个候选 j），闭式解。

    在 z 空间：logBF_j = −½·log(1 + V·R_jj) + ½·V·r_j²/(1 + V·R_jj)。
    r 是 **z 空间残差**（= z − R·Σ_{m≠l} α_m）。
    """
    d = np.diag(R)
    return -0.5 * np.log1p(V * d) + 0.5 * V * r ** 2 / (1.0 + V * d)


def _cs_purity(R: np.ndarray, members: Sequence[int]) -> float:
    """可信集纯度：集合内 SNP 两两 |r| 的最小值（纯度低说明集合仍被 LD 拉扯）。"""
    idx = list(members)
    if len(idx) < 2:
        return 1.0
    sub = np.abs(R[np.ix_(idx, idx)])
    np.fill_diagonal(sub, 1.0)
    return float(sub[np.triu_indices(len(idx), 1)].min())


def posterior_from_p(pvalue: Sequence[float], ld: Optional[np.ndarray] = None,
                     snp: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """无效应量时的退路：用 P 值近似后验（需 LD，否则只能给归一化权重）。

    ⚠️ 这是**退路不是等价方法**：只用 P 值会丢掉效应方向与量级信息，
       精度明显低于 ABF / SuSiE。能拿到 beta + se 就不要用它。
    """
    p = np.asarray(pvalue, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalue 必须在 [0,1]")
    # 极显著位点的 p 会**浮点下溢成 0**（z>38 时必现）。直接报错等于
    # 让最强信号的位点把整个函数搞挂；裁剪到最小正正规数并标注。
    p = np.clip(p, np.finfo(float).tiny, 1.0)
    z = np.abs(_st.norm.ppf(p / 2))
    w = z ** 2
    if ld is not None:
        R = np.asarray(ld, dtype=float)
        # 用 LD 做一次稀释：强连锁的位点共享证据，权重按相关调整
        w = w / np.abs(R).sum(axis=1)
    w = np.nan_to_num(w, nan=0.0)
    post = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
    order = np.argsort(-post)
    cum, cs = 0.0, []
    for i in order:
        cs.append(int(i))
        cum += post[i]
        if cum >= 0.95:
            break
    return {
        "posterior": post,
        "credible_set": cs,
        "coverage": float(cum),
        "credible_set_snps": None if snp is None else [str(snp[i]) for i in cs],
        "caution": "仅用 P 值的退路方法，精度低于 ABF/SuSiE；无 LD 时尤其粗糙",
    }
