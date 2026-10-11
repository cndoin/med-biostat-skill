# PORTED-FILE · 来源 dswah/pyGAM（Apache-2.0）
#   源文件 pygam/utils.py（gen_edge_knots / b_spline_basis）
#           pygam/penalties.py（derivative / sparse_diff）
#           pygam/pygam.py（_pirls 的 QR+SVD 稳定求解、_estimate_model_statistics
#                          的 edof 与 scale、_estimate_GCV_UBRE 的 GCV 公式）
# 移植方式：**算法照抄、依赖换底**。上游是通用 GAM 框架（任意分布/连接函数/
#   多变量张量积/约束/稀疏）；本文件只保留「**高斯族 + identity 连接 + 单变量
#   P 样条**」这一条路径，改成 dense numpy，去掉 progressbar2 / sksparse 可选依赖。
# 逐函数对照（上游 -> 本文件）：
#   utils.gen_edge_knots          -> gen_edge_knots
#   utils.b_spline_basis          -> b_spline_basis（含线性外推；periodic 恒 False）
#   penalties.derivative          -> derivative_penalty
#   pygam._pirls（QR + SVD）      -> fit_once（高斯特化：W=I、伪数据=y、一步收敛）
#   pygam._estimate_model_statistics（edof / scale / cov）-> fit_once 的返回
#   pygam._estimate_GCV_UBRE      -> gcv_score（高斯族走 GCV：n·dev/(n−γ·edof)²）
#
# 与上游的**已知差异（如实标注）**：
#   1. 只有高斯族 + identity 连接。Poisson/Binomial/其他连接函数**未移植**。
#   2. 只有单变量样条项（SplineTerm）。张量积、因子项、约束（单调/凸）未移植。
#   3. 平滑参数只走 **GCV 网格搜索**（pyGAM 默认 lam=0.6 不搜索；本文件 lam=None
#      时按 GCV 选）。选择口径随结果输出（`lam_selection`）。
#   4. pyGAM 默认 `gamma=1.4`（Wood 2006），此处保持同值——**这是有效自由度
#      的额外惩罚系数**，改它会改变平滑度，故必须随结果输出。
#   5. 参数协方差用 `(B·Bᵀ)·scale²`（pyGAM 原式），不做 Fisher 信息修正。
"""单变量广义可加模型（惩罚 P 样条 + GCV 选平滑参数）。

用途：拟合「基因表达 ~ 年龄」这类**非线性、单调性未知**的关系，并给出
带置信带的曲线——「失守点」（`lib/age_onset.py`）就建立在这条曲线上。

**为什么用 GAM 而不是线性回归**：年龄效应通常不是线性的（发育期陡、成年期平、
老年期再加速）。用线性回归拟合会把这个形状强行压成一条直线，
得到的"斜率"和"显著性"都不代表真实过程。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "gen_edge_knots", "b_spline_basis", "derivative_penalty", "spline_basis",
    "fit_once", "gcv_score", "select_lam", "fit_gam",
]

_EPS = float(np.finfo(np.float64).eps)


# ── 基函数 ───────────────────────────────────────────────────────────────────
def gen_edge_knots(data: Sequence[float], dtype: str = "numerical") -> np.ndarray:
    """照抄 pygam.utils.gen_edge_knots：只返回**两个端点节点**。

    中间节点由 ``b_spline_basis`` 在 [0,1] 上均分（**不是**分位数节点）——
    这是 pyGAM 的口径，与 R 的 mgcv 默认（分位数节点）不同。
    """
    data = np.asarray(data, dtype=float)
    if dtype not in ("categorical", "numerical"):
        raise ValueError(f"unsupported dtype: {dtype}")
    if dtype == "categorical":
        return np.r_[np.min(data) - 0.5, np.max(data) + 0.5]
    return np.r_[np.min(data), np.max(data)]


def b_spline_basis(x: Sequence[float], edge_knots: Sequence[float],
                   n_splines: int = 20, spline_order: int = 3,
                   periodic: bool = False) -> np.ndarray:
    """照抄 pygam.utils.b_spline_basis（dense 版）。

    向量化 De Boor 递推生成 B 样条基；**端点外线性外推**（不是平坦外推）。
    :return: (len(x), n_splines) 的基矩阵
    """
    x = np.asarray(x, dtype=float)
    if np.ravel(x).ndim != 1:
        raise ValueError(f"Data must be 1-D, but found {np.ravel(x).ndim}")
    if n_splines < 1:
        raise ValueError("n_splines must be int >= 1")
    if spline_order < 0:
        raise ValueError("spline_order must be int >= 1")
    if n_splines < spline_order + 1:
        raise ValueError(
            "n_splines must be >= spline_order + 1. "
            f"found: n_splines = {n_splines} and spline_order = {spline_order}")

    n_splines += spline_order * int(periodic)

    edge_knots = np.sort(np.array(edge_knots, dtype=float))
    offset = edge_knots[0]
    scale = edge_knots[-1] - edge_knots[0]
    if scale == 0:
        scale = 1.0
    boundary_knots = np.linspace(0, 1, 1 + n_splines - spline_order)
    diff = np.diff(boundary_knots[:2])[0]

    x = (np.ravel(x) - offset) / scale
    if periodic:
        x = x % (1 + 1e-9)

    x = np.r_[x, 0.0, 1.0]

    x_extrapolate_l = x < 0
    x_extrapolate_r = x > 1
    x_interpolate = ~(x_extrapolate_r + x_extrapolate_l)

    x = np.atleast_2d(x).T

    aug = np.arange(1, spline_order + 1) * diff
    aug_knots = np.r_[-aug[::-1], boundary_knots, 1 + aug]
    aug_knots[-1] += 1e-9

    bases = (x >= aug_knots[:-1]).astype(int) * (x < aug_knots[1:]).astype(int)
    bases[-1] = bases[-2][::-1]

    maxi = len(aug_knots) - 1
    prev_bases = None
    for m in range(2, spline_order + 2):
        maxi -= 1
        num = x - aug_knots[:maxi]
        num = num * bases[:, :maxi]
        denom = aug_knots[m - 1: maxi + m - 1] - aug_knots[:maxi]
        left = num / denom

        num = (aug_knots[m: maxi + m] - x) * bases[:, 1: maxi + 1]
        denom = aug_knots[m: maxi + m] - aug_knots[1: maxi + 1]
        right = num / denom

        prev_bases = bases[-2:]
        bases = left + right

    if periodic and spline_order > 0:
        bases[:, :spline_order] = np.max(
            [bases[:, :spline_order], bases[:, -spline_order:]], axis=0)
        bases = bases[:, :-spline_order]

    if (any(x_extrapolate_r) or any(x_extrapolate_l)) and spline_order > 0:
        bases[~x_interpolate] = 0.0
        denom = aug_knots[spline_order:-1] - aug_knots[:-spline_order - 1]
        left = prev_bases[:, :-1] / denom
        denom = aug_knots[spline_order + 1:] - aug_knots[1:-spline_order]
        right = prev_bases[:, 1:] / denom
        grads = spline_order * (left - right)

        if any(x_extrapolate_l):
            bases[x_extrapolate_l] = grads[0] * x[x_extrapolate_l] + bases[-2]
        if any(x_extrapolate_r):
            bases[x_extrapolate_r] = grads[1] * (x[x_extrapolate_r] - 1) + bases[-1]

    bases = bases[:-2]
    return np.asarray(bases, dtype=float)


def derivative_penalty(n: int, derivative: int = 2, periodic: bool = False) -> np.ndarray:
    """照抄 pygam.penalties.derivative（dense 版，periodic 恒 False）。

    P 样条惩罚：**惩罚相邻系数的 d 阶差分平方和**
    （P = DᵀD，D 为 d 阶差分矩阵）。这是 P 样条区别于普通回归样条的关键
    ——它让"摆动"付出代价，从而由平滑参数控制曲线粗糙度。
    """
    if n == 1:
        return np.zeros((1, 1))
    if derivative < 1:
        raise ValueError("derivative must be >= 1")
    if n - derivative < 1:
        raise ValueError(f"n={n} 太小，无法做 {derivative} 阶差分——至少需要 {derivative + 1} 个系数")
    D = np.diff(np.eye(n), n=derivative, axis=1)      # (n, n-derivative)
    return D @ D.T                                     # (n, n)


def spline_basis(x: Sequence[float], n_splines: int = 20, spline_order: int = 3,
                 edge_knots: Optional[Sequence[float]] = None) -> np.ndarray:
    """便利封装：端点缺省即由数据决定（**训练数据决定端点**，预测时须复用）。"""
    x = np.asarray(x, dtype=float)
    if edge_knots is None:
        edge_knots = gen_edge_knots(x)
    return b_spline_basis(x, edge_knots, n_splines=n_splines, spline_order=spline_order)


# ── 拟合（高斯族 PIRLS 特化）────────────────────────────────────────────────
def fit_once(B: np.ndarray, y: Sequence[float], P: np.ndarray,
             lam: float) -> Dict[str, Any]:
    """照抄 pygam._pirls 的**高斯族特化**（W=I、伪数据=y，故一步收敛）。

    用 QR + SVD 求解（上游的稳定算法），不解正规方程——后者在
    (BᵀB + λP) 病态时会给出发散的解。

    :return: dict（coef / edof / scale / cov / fitted / stderr / B_hat）
    """
    B = np.asarray(B, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, m = B.shape

    S = np.eye(m) * np.sqrt(_EPS)
    P_lam = S + np.asarray(P, dtype=float) * float(lam)

    # 上游 cholesky 返回**上**三角 U 使 A = UᵀU；numpy 返回下三角 L 使 A = LLᵀ
    L = np.linalg.cholesky(P_lam)
    E = L.T

    Q, R = np.linalg.qr(B)
    min_n_m = int(np.min([m, n]))
    U, d, Vt = np.linalg.svd(np.vstack([R, E]), full_matrices=False)

    d_inv = d ** -1
    U1 = U[:min_n_m, :min_n_m]
    Vt = Vt[:min_n_m]
    d_inv = d_inv[:min_n_m]

    B_hat = (Vt.T * d_inv).dot(U1.T).dot(Q.T)
    coef = B_hat.dot(y).flatten()

    fitted = B @ coef
    edof_per_coef = np.diagonal(U1.dot(U1.T))
    edof = float(edof_per_coef.sum())

    # 高斯族的 scale：phi = RSS/(n − edof)，scale = sqrt(phi)（pyGAM 原式）
    dof = float(n - edof)
    phi = float(np.sum((y - fitted) ** 2) / dof) if dof > 0 else float("nan")
    scale = float(np.sqrt(phi)) if phi == phi else float("nan")

    cov = (B_hat.dot(B_hat.T)) * (scale ** 2)
    se_coef = np.sqrt(np.clip(np.diagonal(cov), 0, None))

    # 拟合值的标准误：Var(mu_i) = Σ_j (B_i · B_hat[:,j])² · scale²
    scaled = (B @ B_hat) * (scale if scale == scale else 0.0)
    se_fit = np.sqrt(np.clip(np.sum(scaled ** 2, axis=1), 0, None))

    return {
        "coef": coef,
        "B_hat": B_hat,
        "edof": edof,
        "edof_per_coef": edof_per_coef,
        "scale": scale,
        "phi": phi,
        "cov": cov,
        "se_coef": se_coef,
        "fitted": fitted,
        "se_fit": se_fit,
        "deviance": float(np.sum((y - fitted) ** 2)),
        "lam": float(lam),
    }


def gcv_score(deviance: float, n: int, edof: float, gamma: float = 1.4) -> float:
    """照抄 pygam._estimate_GCV_UBRE 的 GCV 分支（高斯族 scale 未知）：

        GCV = n · deviance / (n − γ · edof)²

    γ（默认 1.4，Wood 2006）**额外惩罚有效自由度**——γ 越大曲线越平滑。
    改 γ 会改变选出的平滑度，故它必须随结果输出。
    """
    denom = (n - gamma * edof) ** 2
    if denom <= 0:
        return float("inf")
    return float(n * deviance / denom)


def select_lam(B: np.ndarray, y: Sequence[float], P: np.ndarray,
               lams: Optional[Sequence[float]] = None, gamma: float = 1.4) -> Dict[str, Any]:
    """网格搜索选平滑参数 λ（GCV 最小）。

    ⚠️ **λ 必须在用于推断的数据上按统一规则选**，不允许"看着图好看"手工调
    ——事后调参会让 GCV/显著性失去意义。
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    if lams is None:
        lams = np.logspace(-3, 6, 25)

    best = None
    table = []
    for lam in np.asarray(lams, dtype=float):
        try:
            r = fit_once(B, y, P, lam)
        except np.linalg.LinAlgError:
            continue
        g = gcv_score(r["deviance"], n, r["edof"], gamma=gamma)
        table.append({"lam": float(lam), "gcv": g, "edof": r["edof"],
                      "deviance": r["deviance"]})
        if best is None or g < best[1]:
            best = (float(lam), g, r)

    if best is None:
        raise RuntimeError("所有候选 λ 都拟合失败——检查基矩阵是否退化（如年龄全为常数）")

    return {
        "lam": best[0],
        "gcv": best[1],
        "fit": best[2],
        "table": table,
        "gamma": float(gamma),
        "lam_selection": f"GCV 网格搜索（gamma={gamma}，{len(table)} 个候选）",
    }


def fit_gam(x: Sequence[float], y: Sequence[float], n_splines: int = 20,
            spline_order: int = 3, lam: Optional[float] = None,
            lams: Optional[Sequence[float]] = None, gamma: float = 1.4,
            derivative: int = 2) -> Dict[str, Any]:
    """拟合单变量高斯 GAM：``y ~ s(x)``（惩罚 P 样条）。

    :param lam: 给定则固定（**用于跨数据集复用同一平滑度**）；None 则 GCV 选
    :return: dict（basis/edge_knots/trim 参数、拟合值、edof、scale、口径）
        - ``coef`` / ``fitted`` / ``se_fit`` / ``edof`` / ``scale``
        - ``edge_knots`` / ``n_splines`` / ``spline_order`` / ``derivative``
          —— **预测新数据时必须复用这些**，否则基函数不一致
        - ``lam`` / ``lam_selection`` / ``gcv`` / ``gamma``
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x 长度 {x.shape[0]} != y 长度 {y.shape[0]}")
    if x.shape[0] < n_splines:
        raise ValueError(
            f"样本数 {x.shape[0]} 少于样条数 {n_splines}——会过拟合到饱和。"
            "请减小 n_splines（经验：每个样条至少 3-5 个观测）")
    if np.unique(x).size < 2:
        raise ValueError("x 必须至少有 2 个不同取值（年龄全相同无法拟合趋势）")

    edge_knots = gen_edge_knots(x)
    B = b_spline_basis(x, edge_knots, n_splines=n_splines, spline_order=spline_order)
    P = derivative_penalty(n_splines, derivative=derivative)

    # ⚠️ y 若为常数，RSS≈0 → scale≈0 → 置信带宽度 0。拟合本身合法（就是水平线），
    # 但**零宽置信带会让下游把任何微小波动判成显著**。所以显式标记，交由调用方决定。
    y_sd = float(np.std(y))
    degenerate = (y_sd == 0.0)

    if lam is None:
        sel = select_lam(B, y, P, lams=lams, gamma=gamma)
        fit = sel["fit"]
        lam_out, gcv, rule = sel["lam"], sel["gcv"], sel["lam_selection"]
        table = sel["table"]
    else:
        fit = fit_once(B, y, P, lam)
        n = y.shape[0]
        gcv = gcv_score(fit["deviance"], n, fit["edof"], gamma=gamma)
        lam_out, rule, table = float(lam), "用户指定（未搜索）", None

    return {
        "x": x,
        "y": y,
        "edge_knots": edge_knots,
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "derivative": int(derivative),
        "basis": B,
        "penalty": P,
        "coef": fit["coef"],
        "B_hat": fit["B_hat"],
        "fitted": fit["fitted"],
        "se_fit": fit["se_fit"],
        "edof": fit["edof"],
        "scale": fit["scale"],
        "phi": fit["phi"],
        "deviance": fit["deviance"],
        "lam": lam_out,
        "lam_selection": rule,
        "gcv": float(gcv),
        "gamma": float(gamma),
        "lam_table": table,
        "n": int(x.shape[0]),
        "degenerate": degenerate,
        "y_sd": y_sd,
    }


def predict_basis(model: Dict[str, Any], x_new: Sequence[float]) -> np.ndarray:
    """按已拟合模型的基参数构建新数据的基矩阵（**推理必须走这里**）。

    复用 ``edge_knots`` / ``n_splines`` / ``spline_order``，
    否则新数据的基与原模型不在同一个函数空间里，预测无意义。
    """
    return b_spline_basis(np.asarray(x_new, dtype=float), model["edge_knots"],
                          n_splines=model["n_splines"],
                          spline_order=model["spline_order"])
