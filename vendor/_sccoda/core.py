# PORTED-FILE · 来源 theislab/scCODA（**BSD-3-Clause**，2026-10-12 查 GitHub 官方 SPDX）
#   论文：Büttner M, Ostner J, Müller CL, Theis FJ, Schubert B.
#         "scCODA is a Bayesian model for compositional single-cell data analysis"
#         *Nature Communications* 2021;12:4696
#   对应上游源码：sccoda/model/*.py（dirichlet_models.py / base_model.py）
#
# 移植方式：**统计模型照抄、推断引擎换底**。
#   上游用 numpyro / tensorflow-probability 的 **NUTS-HMC 抽样** 给出后验包含概率
#   （pip / "acceptance probability"）。这三个依赖（tensorflow + tfp + arviz）
#   **不能作为本 skill 的运行依赖内嵌**（体积与可用性都不允许），
#   故改为 **惩罚最大后验（MAP）+ 供体 bootstrap** 的确定性推断。
#
# 逐块对照（上游 -> 本文件）：
#   层次 Dirichlet-Multinomial 似然（每个样本一行计数）
#                                        -> dm_loglik
#   softmax 链接 η = X β，参考细胞类型整行 β ≡ 0（可辨识性约束）
#                                        -> _softmax / _eta / fit
#   spike-and-slab 先验（β = 0 的尖峰 + 弥散 slabs）
#                                        -> **L1（Laplace）惩罚**：MAP 下尖峰分量
#                                           正是"把系数压到恰好 0"，L1 是它
#                                           的连续松弛且给出同样的稀疏解
#   HMC 后验包含概率（pip / acceptance prob）-> **供体 bootstrap 的选中频率**
#   θ（DM 分散度）                          -> 同为待估参数，exp 参数化保正
#
# 与上游的**已知差异（如实标注，不得省略）**：
#   1. **推断引擎不同**：上游是 HMC 后验抽样（报后验包含概率 + 后验均值），
#      本文件是 L1 惩罚 MAP（报点估计）+ 供体 bootstrap（报 CI 与选中频率）。
#      → 两者**不可逐位对比**：本文件的 `sel_freq` **不是** 上游的 pip，
#        `estimate` **不是** 上游的后验均值。只能定性比对（方向 + 是否被选中）。
#      结果 dict 里带 `inference` 字段，写论文时必须照抄这个说明。
#   2. **惩罚强度 λ 由 BIC 网格选**（确定性、可复现），上游用的是先验超参
#      （默认 slab 方差与 inclusion prob），不是同一件东西。
#   3. **未移植**：上游的 `tascCODA`（树状层次结构 / 细胞类型谱系聚合）、
#      `nocoda` 变体、模型诊断（MCMC 诊断当然是 HMC 特有的）、ArviZ 相关绘图。
#   4. **未使用** patsy 公式接口；X 由调用方直接给设计矩阵（更可控，
#      也避免公式解析口径漂移）。
#   5. 输入是 dense numpy：n 供体 × K 细胞类型的计数矩阵。上游走 AnnData。
"""scCODA 模型族：成分数据的供体层级差异丰度检验。

**为什么必须是"供体层级 + 成分模型"**
- 供体层级：细胞不是独立样本（伪重复），检验单位是供体。
- 成分模型：细胞类型比例**天生和为 1**，某一类型上升必然挤掉其他类型。
  若对各类别独立做 t 检验，会同时犯两类错：① 忽略约束导致方差错；
  ② 一个类型真涨了，其他类型"被稀释"看起来也像变了。
  成分模型（此处为 Dirichlet-Multinomial + log-ratio 效应）是这类问题的正解。

**效应量的含义**：``estimate`` 是 **相对参考细胞类型的 log 丰度比**随每单位
协变量的变化，不是"比例的变化量"。报告时必须写明参考细胞类型是哪一类
（结果里带 ``reference`` 字段）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize

__all__ = ["dm_loglik", "fit", "bootstrap", "DEFAULT_LAMBDA_GRID"]


DEFAULT_LAMBDA_GRID: Tuple[float, ...] = (
    0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
)


def _softmax(E: np.ndarray) -> np.ndarray:
    """行 softmax（减去行最大值防溢出，不影响结果）。"""
    z = E - E.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def dm_loglik(Y: np.ndarray, Pi: np.ndarray, theta: float) -> float:
    """Dirichlet-Multinomial 对数似然（α_k = θ·π_k，Σα = θ）。

    log L = Σ_i [ logΓ(N_i+1) − Σ_k logΓ(y_ik+1)
                  + logΓ(θ) − logΓ(N_i+θ)
                  + Σ_k ( logΓ(y_ik+θπ_ik) − logΓ(θπ_ik) ) ]
    """
    N = Y.sum(axis=1)
    a = theta * Pi
    const = (gammaln(N + 1.0) - gammaln(Y + 1.0).sum(axis=1)
             + gammaln(theta) - gammaln(N + theta))
    return float(np.sum(const + (gammaln(Y + a) - gammaln(a)).sum(axis=1)))


def _unpack(params: np.ndarray, shape: Tuple[int, int], ref: int) -> Tuple[np.ndarray, float]:
    """把扁平参数还原为 (β 的 (K,P) 矩阵[参考行置 0], θ)。"""
    K, P = shape
    beta_small = params[: (K - 1) * P].reshape(K - 1, P)
    beta = np.zeros((K, P))
    rows = [r for r in range(K) if r != ref]
    beta[rows, :] = beta_small
    theta = float(np.exp(params[-1]))
    return beta, theta


def _neg_obj(params: np.ndarray, Y: np.ndarray, X: np.ndarray,
             ref: int, lam: float, pen_mask: np.ndarray,
             keep_mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """负对数后验（含 L1 惩罚）及其梯度。返回 (f, grad)。

    :param keep_mask: (K−1 × P) 布尔掩码；为 False 的系数**恒为 0**。
        用于"L1 选完变量后，在选中的支持集上做无惩罚重拟合"——
        惩罚 MAP 会把估计值系统性拉向 0（实测 0.35 的真值被拉到 0.325），
        这一步把**选择**与**估计**分开：选择归 L1，估计归无惩罚 MLE。
    """
    K, P = Y.shape[1], X.shape[1]
    rows = [r for r in range(K) if r != ref]
    n = Y.shape[0]

    raw = params[: (K - 1) * P].reshape(K - 1, P)
    if keep_mask is not None:
        raw = raw * keep_mask
    beta = np.zeros((K, P))
    beta[rows, :] = raw
    theta = float(np.exp(params[-1]))
    theta = float(np.clip(theta, 1e-3, 1e6))

    # 数值护栏：softmax 在 |η|>30 时已完全饱和，再大只会让 exp 溢出成 NaN 梯度。
    # 裁掉这一段不改变模型含义，但能杜绝 L-BFGS-B 收到 NaN 后静默跑偏。
    Eta = np.clip(X @ beta.T, -30.0, 30.0)
    Pi = _softmax(Eta)
    a = theta * Pi
    N = Y.sum(axis=1)

    # ── 对数似然 ──
    ll = dm_loglik(Y, Pi, theta)

    # ── ∂L/∂π 与 ∂L/∂η ──
    g = theta * (digamma(Y + a) - digamma(a))          # (n, K)
    dEta = Pi * (g - (Pi * g).sum(axis=1, keepdims=True))   # (n, K)

    # ── ∂L/∂β（仅自由行）──
    dbeta = np.zeros((K, P))
    dbeta[rows, :] = dEta[:, rows].T @ X               # (K-1, P)

    # ── ∂L/∂θ（θ = exp(logθ)，链式）──
    dtheta = float(np.sum(digamma(theta) - digamma(N + theta)
                          + (Pi * (digamma(Y + a) - digamma(a))).sum(axis=1)))

    # ── L1 惩罚（只对被惩罚的列）──
    pen_beta = beta[rows, :] * pen_mask[None, :]
    l1 = lam * float(np.abs(pen_beta).sum())
    # |x| 的次梯度在 0 处取 0（与 L-BFGS-B 的数值行为一致，且 MAP 解恰在 0 上）
    dpen = lam * np.sign(pen_beta) * pen_mask[None, :]

    f = -(ll - l1)
    grad_small = (-(dbeta[rows, :] - dpen)).ravel()
    if keep_mask is not None:
        grad_small = grad_small * keep_mask.ravel()
    grad_theta = np.array([-(dtheta * theta)])
    grad = np.concatenate([grad_small, grad_theta])
    if not np.isfinite(f) or not np.all(np.isfinite(grad)):
        # 兜底：绝不把 NaN 交给 L-BFGS-B（它会"成功"返回一个毫无意义的解）
        return 1e12, np.zeros_like(grad)
    return float(f), grad


def fit(Y, X, reference: Optional[int] = None,
        lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
        intercept: bool = True,
        theta_init: float = 100.0,
        max_iter: int = 2000) -> Dict[str, Any]:
    """拟合 scCODA 模型族（L1 惩罚 MAP，λ 由 BIC 网格选）。

    :param Y: (n_donor × K) **非负计数**矩阵（每个供体每个细胞类型的细胞数）
    :param X: (n_donor × P) 设计矩阵。`intercept=True` 时**第一列视为截距**，
        不参与 L1 惩罚（截距是基线组成，不该被压向均匀）
    :param reference: 参考细胞类型下标（默认 = 最后一列）。整行 β≡0，
        所有效应都是**相对它**的 log 丰度比
    :param lambda_grid: λ 网格；BIC 最小者入选（确定性，无随机性）
    :return: dict，含 ``beta``(K×P) / ``theta`` / ``lambda`` / ``bic`` /
        ``loglik`` / ``selected`` / ``inference``
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if Y.ndim != 2 or X.ndim != 2:
        raise ValueError("Y 与 X 都必须为二维")
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"Y 行数 {Y.shape[0]} != X 行数 {X.shape[0]}")
    if np.any(~np.isfinite(Y)) or np.any(Y < 0):
        raise ValueError("Y 必须为非负有限计数")
    # 逐个格子查整数，不能只查行合计：实测只查合计时 Y*1.5（合计恰好是整数）
    # 能蒙混过关，而 DM 的 gammaln(y+1) 对非整数 y 是有定义的 —— 于是
    # "归一化过的假计数"会静默算出一个看起来完全正常的结果。
    if not np.allclose(Y, np.round(Y)):
        raise ValueError(
            "Y 必须是**整数计数**（细胞个数），不能是比例/归一化值/CPM。"
            "DM 似然里 gammaln(y+1) 对非整数也有定义，所以喂错数据不会报错，"
            "只会静默给出一个错的结果")
    n, K = Y.shape
    P = X.shape[1]
    if K < 2:
        raise ValueError("至少要有 2 个细胞类型（成分数据才有意义）")
    if n < 3:
        raise ValueError(f"供体数 {n} 太少；成分检验的单位是供体，n<3 没有分辨力")
    if reference is None:
        reference = K - 1
    if not (0 <= reference < K):
        raise ValueError(f"reference={reference} 越界（K={K}）")
    if Y.sum(axis=1).min() <= 0:
        raise ValueError("存在总计数为 0 的供体（该供体不携带任何成分信息）")

    pen_mask = np.ones(P)
    if intercept and P >= 1:
        pen_mask[0] = 0.0

    rows = [r for r in range(K) if r != reference]
    n_free = (K - 1) * P
    x0 = np.concatenate([np.zeros(n_free), [np.log(theta_init)]])
    bounds = [(None, None)] * n_free + [(np.log(1e-3), np.log(1e6))]

    best = None
    for lam in lambda_grid:
        res = minimize(_neg_obj, x0, args=(Y, X, reference, float(lam), pen_mask),
                       jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
        beta, theta = _unpack(res.x, (K, P), reference)
        bs = beta[rows, :]
        nz = int(np.sum(np.abs(bs[:, pen_mask > 0]) > 1e-8))
        df = int(np.sum(pen_mask == 0) * (K - 1)) + nz + 1
        ll = dm_loglik(Y, _softmax(np.clip(X @ beta.T, -30, 30)), theta)
        bic = -2.0 * ll + np.log(n) * df
        cand = {"bic": float(bic), "loglik": float(ll), "beta": beta,
                "theta": float(theta), "lambda": float(lam), "df": int(df),
                "n_nonzero": nz, "success": bool(res.success)}
        if best is None or cand["bic"] < best["bic"]:
            best = cand

    # ── 选择 vs 估计分离 ────────────────────────────────────────────────
    # L1 只负责**选出**哪些 (细胞类型 × 协变量) 非零；选中之后在支持集上做
    # **无惩罚重拟合**，避免惩罚把效应量系统性拉向 0（实测真值 0.35 → 0.325）。
    keep = np.abs(best["beta"][rows, :]) > 1e-8
    if keep.any():
        km = keep.astype(float)
        res2 = minimize(_neg_obj, x0, args=(Y, X, reference, 0.0, pen_mask, km),
                        jac=True, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8})
        beta2, theta2 = _unpack(res2.x, (K, P), reference)
        ll2 = dm_loglik(Y, _softmax(np.clip(X @ beta2.T, -30, 30)), theta2)
        if np.all(np.isfinite(beta2)) and ll2 >= best["loglik"] - 1e-6:
            best = dict(best, beta=beta2, theta=float(theta2), loglik=float(ll2),
                        success=bool(res2.success))

    beta = best["beta"]
    selected = np.zeros((K, P), dtype=bool)
    selected[rows, :] = np.abs(beta[rows, :]) > 1e-8
    return {
        "beta": beta,
        "selected": selected,
        "theta": best["theta"],
        "lambda": best["lambda"],
        "bic": best["bic"],
        "loglik": best["loglik"],
        "df": best["df"],
        "n_nonzero": best["n_nonzero"],
        "reference": int(reference),
        "rows_free": rows,
        "n": int(n),
        "K": int(K),
        "P": int(P),
        "intercept": bool(intercept),
        "optimization_success": bool(best["success"]),
        "estimation": ("L1 选变量（λ 由 BIC 网格选）→ 选中支持集上无惩罚 MLE 重拟合；"
                       "选择归 L1、估计归 MLE，避免惩罚把效应量拉向 0"),
        "inference": ("惩罚 MAP + 供体 bootstrap — **不是**上游的 HMC 后验；"
                      "sel_freq 不是后验包含概率，估计值不是后验均值"),
        "engine": "vendor._sccoda.fit",
    }


def bootstrap(Y, X, reference: Optional[int] = None, lam: Optional[float] = None,
              B: int = 200, seed: int = 0, alpha: float = 0.05,
              intercept: bool = True) -> Dict[str, Any]:
    """**供体 bootstrap**：给出估计的不确定性（HMC 后验的确定性替身）。

    :param lam: λ（不给则先用全量数据按 BIC 选一次，之后**固定**复用——
        每次重抽样重选 λ 会让各次重抽样的模型不同，CI 就不可比了）
    :param B: 重抽样次数（默认 200；固定 seed，**可复现**）
    :return: dict，含 ``est`` / ``ci_low`` / ``ci_high`` / ``p_boot`` /
        ``sel_freq`` / ``padj``(BH) / ``B`` / ``seed``
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = Y.shape[0]
    K, P = Y.shape[1], X.shape[1]
    if reference is None:
        reference = K - 1
    rows = [r for r in range(K) if r != reference]

    base = fit(Y, X, reference=reference, intercept=intercept)
    lam_use = base["lambda"] if lam is None else float(lam)
    # B 下限不是拍脑袋定的：实测在**纯零假设**下（n=18 供体、5 类、1 个协变量）
    #   B=40  → 12 次独立重复里出现 1 次假阳性（padj<0.05）
    #   B=200 → 48 次零假设检验 **0** 假阳性
    # bootstrap 的 p = 2·min(frac_pos, frac_neg)，其分辨率是 2/B；B 太小时
    # "恰好同号"就被读成 p≈0。故把下限抬到 100，并在文档里写明这个实测依据。
    if B < 100:
        raise ValueError(
            f"B={B} 太小（下限 100）。bootstrap p 的分辨率是 2/B，B 过小时"
            f"零假设下也会出现 padj<0.05 的假阳性（实测 B=40 有，B=200 无）")

    rng = np.random.default_rng(seed)
    draws = np.zeros((B, K, P))
    ok = 0
    for b in range(B):
        idx = rng.integers(0, n, n)
        try:
            r = _fit_fixed_lambda(Y[idx], X[idx], reference, lam_use, intercept)
        except Exception:
            continue
        if r is None:
            continue
        draws[ok] = r
        ok += 1
    if ok < 20:
        raise ValueError(f"bootstrap 只有 {ok}/{B} 次成功拟合，"
                         f"无法给出可信的分位数（多半是重抽样后某个供体缺失）")
    draws = draws[:ok]

    est = base["beta"]
    lo = np.percentile(draws, 100 * alpha / 2, axis=0)
    hi = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    frac_pos = (draws > 0).mean(axis=0)
    frac_neg = (draws < 0).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_boot = np.clip(2.0 * np.minimum(frac_pos, frac_neg), 0.0, 1.0)
    sel_freq = (np.abs(draws) > 1e-8).mean(axis=0)

    # 参考行的系数**恒为 0**（可辨识性约束），它的 p 值不是 0 而是"不存在"。
    # 留成 0.0 会让"参考类型显著变化"这种没有意义的行混进结果表。
    p_boot[reference, :] = np.nan
    sel_freq[reference, :] = np.nan

    pv = p_boot[rows, :].ravel()
    order = np.argsort(pv)
    m = pv.size
    ranked = np.clip(pv[order] * m / (np.arange(m) + 1), 0, 1)
    q = np.empty(m)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    padj = np.full((K, P), np.nan)
    padj[rows, :] = q.reshape(len(rows), P)

    return {
        "est": est,
        "ci_low": lo,
        "ci_high": hi,
        "p_boot": p_boot,
        "padj": padj,
        "sel_freq": sel_freq,
        "selected": base["selected"],
        "theta": base["theta"],
        "lambda": lam_use,
        "reference": int(reference),
        "rows_free": rows,
        "B": int(ok),
        "B_requested": int(B),
        "seed": int(seed),
        "alpha": float(alpha),
        "inference": ("供体 bootstrap（固定 seed，可复现）—— 替代上游 HMC 的后验区间；"
                      "**sel_freq 不是后验包含概率**"),
        "engine": "vendor._sccoda.bootstrap",
    }


def _fit_fixed_lambda(Y, X, reference, lam, intercept):
    """固定 λ 的单次 MAP 拟合（bootstrap 内部用，失败返回 None）。"""
    K, P = Y.shape[1], X.shape[1]
    pen_mask = np.ones(P)
    if intercept and P >= 1:
        pen_mask[0] = 0.0
    rows = [r for r in range(K) if r != reference]
    x0 = np.concatenate([np.zeros((K - 1) * P), [np.log(100.0)]])
    bounds = [(None, None)] * ((K - 1) * P) + [(np.log(1e-3), np.log(1e8))]
    res = minimize(_neg_obj, x0, args=(Y, X, reference, float(lam), pen_mask),
                   jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 800, "ftol": 1e-11})
    beta, theta = _unpack(res.x, (K, P), reference)
    if not np.all(np.isfinite(beta)):
        return None
    # 与主拟合同口径：选中支持集上无惩罚重拟合，否则 bootstrap 分布会带
    # L1 收缩偏差，CI 系统性偏窄且中心偏低。
    keep = np.abs(beta[rows, :]) > 1e-8
    if keep.any():
        km = keep.astype(float)
        res2 = minimize(_neg_obj, res.x,
                        args=(Y, X, reference, 0.0, pen_mask, km),
                        jac=True, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 800, "ftol": 1e-11})
        b2, _t2 = _unpack(res2.x, (K, P), reference)
        if np.all(np.isfinite(b2)) and \
                dm_loglik(Y, _softmax(np.clip(X @ b2.T, -30, 30)), _t2) >= \
                dm_loglik(Y, _softmax(np.clip(X @ beta.T, -30, 30)), theta) - 1e-6:
            beta = b2
    return beta
