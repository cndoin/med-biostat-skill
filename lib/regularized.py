"""分析法 ⑦：正则化回归（Ridge / LASSO / Elastic Net / Cox-LASSO）+ 交叉验证选 λ。

算法：坐标下降（cyclic coordinate descent），与 glmnet（Friedman, Hastie,
Tibshirani 2010, JSS）同一族算法；glmnet 本身是 GPL-2，未内嵌，本模块是**公开
算法的等价实现**，不是 glmnet 的包装。

四个模型的**用途差别**（选错模型是这类分析最常见的错误）：
  · Ridge (α=0)      ：不筛变量，只收缩——变量多且都想要时用它
  · LASSO (α=1)      ：会筛变量（部分系数压成 0）——做变量挑选时用
  · Elastic Net (0<α<1)：相关变量成组进出——高维共线性数据（组学）首选
  · Cox-LASSO        ：生存结局的 Cox 偏似然 + L1，用于预后模型

⚠️ **λ 必须由交叉验证选，且必须在训练集内选**。先在全数据上选 λ 再交叉验证，
   会把测试集信息泄漏进模型，AUC/C-index 系统性乐观——这是预后模型假阳性的头号来源。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _soft(a: float, b: float) -> float:
    """软阈值算子 sign(a)(|a|-b)_+。"""
    if a > b:
        return a - b
    if a < -b:
        return a + b
    return 0.0


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd


def elastic_net(X, y, alpha: float = 1.0, lambdas: Optional[Sequence[float]] = None,
                n_lambda: int = 50, lambda_min_ratio: float = 0.01,
                max_iter: int = 1000, tol: float = 1e-7,
                standardize: bool = True) -> Dict[str, Any]:
    """弹性网路径（含 Ridge 与 LASSO 两端）。

    :param X: 特征矩阵（样本×特征）
    :param y: 连续结局
    :param alpha: 0=Ridge，1=LASSO，中间为 Elastic Net
    :return: 每个 λ 下的系数路径、截距、非零系数个数
    目标函数： 1/(2n)·||y - Xb||² + λ·[ (1-α)/2·||b||² + α·||b||₁ ]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if standardize:
        Xs, mu_x, sd_x = _standardize(X)
    else:
        Xs, mu_x, sd_x = X.copy(), np.zeros(p), np.ones(p)
    y_bar = float(y.mean())
    yc = y - y_bar
    # λ 网格：从 λ_max（所有系数为 0 的最小值）按对数递减
    if lambdas is None:
        lam_max = float(np.max(np.abs(Xs.T @ yc)) / n / max(alpha, 1e-9))
        lam_max = lam_max if np.isfinite(lam_max) and lam_max > 0 else 1.0
        lambdas = np.logspace(np.log10(lam_max), np.log10(lam_max * lambda_min_ratio), n_lambda)
    lambdas = np.asarray(lambdas, dtype=float)

    d = (Xs ** 2).sum(axis=0) / n
    paths, intercepts = [], []
    beta = np.zeros(p)
    for lam in lambdas:
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r = yc - Xs @ beta + Xs[:, j] * beta[j]
                rho = float(Xs[:, j] @ r) / n
                denom = d[j] + lam * (1.0 - alpha)
                beta[j] = _soft(rho, lam * alpha) / denom if denom > 0 else 0.0
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        paths.append(beta.copy() / sd_x)          # 还原到原始尺度
        intercepts.append(y_bar - float(mu_x @ (beta / sd_x)))
    paths = np.array(paths)
    return {
        "lambda": lambdas,
        "coef_path": paths,                       # (n_lambda, p)
        "intercept": np.array(intercepts),
        "n_nonzero": np.array([int(np.count_nonzero(np.abs(b) > 1e-12)) for b in paths]),
        "alpha": float(alpha),
        "standardize": standardize,
        "objective": "1/(2n)||y-Xb||^2 + lambda*[(1-alpha)/2*||b||^2 + alpha*||b||_1]",
    }


def lasso(X, y, **kw) -> Dict[str, Any]:
    """LASSO（alpha=1）。"""
    kw["alpha"] = 1.0
    return elastic_net(X, y, **kw)


def ridge(X, y, **kw) -> Dict[str, Any]:
    """Ridge（alpha=0）。"""
    kw["alpha"] = 0.0
    return elastic_net(X, y, **kw)


def cv_elastic_net(X, y, alpha: float = 1.0, n_folds: int = 5, seed: int = 0,
                   **kw) -> Dict[str, Any]:
    """K 折交叉验证选 λ。

    返回 lambda_min（CV 误差最小）与 lambda_1se（最小误差一个标准误内最简模型）。
    **推荐用 lambda_1se**：lambda_min 常常过拟合，1se 规则得到更简洁的模型
    （glmnet 的默认值就是 1se）。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    ref = elastic_net(X, y, alpha=alpha, **kw)
    lambdas = ref["lambda"]
    err = np.zeros((len(folds), len(lambdas)))
    for f, te in enumerate(folds):
        tr = np.setdiff1d(idx, te)
        fit = elastic_net(X[tr], y[tr], alpha=alpha, lambdas=lambdas, **kw)
        Xte, yte = X[te], y[te]
        for li in range(len(lambdas)):
            pred = Xte @ fit["coef_path"][li] + fit["intercept"][li]
            err[f, li] = float(np.mean((yte - pred) ** 2))
    mean_err = err.mean(axis=0)
    se = err.std(axis=0, ddof=1) / np.sqrt(len(folds))
    i_min = int(np.argmin(mean_err))
    thr = mean_err[i_min] + se[i_min]
    cand = np.flatnonzero(mean_err <= thr)
    # 1se：误差在阈值内且**最简洁**（λ 最大）
    i_1se = int(cand[np.argmax(lambdas[cand])])
    return {
        "lambda": lambdas,
        "cv_mean_mse": mean_err,
        "cv_se": se,
        "lambda_min": float(lambdas[i_min]),
        "lambda_1se": float(lambdas[i_1se]),
        "index_min": i_min,
        "index_1se": i_1se,
        "coef_at_1se": ref["coef_path"][i_1se],
        "intercept_at_1se": float(ref["intercept"][i_1se]),
        "n_folds": n_folds,
        "alpha": float(alpha),
        "rule": "lambda_1se = 一个标准误内最简洁模型；默认推荐它而非 lambda_min",
    }


def cox_lasso(X, time, event, alpha: float = 1.0, lambdas: Optional[Sequence[float]] = None,
              n_lambda: int = 30, lambda_min_ratio: float = 0.05,
              max_iter: int = 200, tol: float = 1e-6) -> Dict[str, Any]:
    """Cox 比例风险模型的 L1 正则路径（Cox-LASSO）。

    用 Breslow 偏似然 + 坐标下降（Simon et al. 2011 的 coxnet 思路）。
    实现说明：为控制依赖，这里用**牛顿-坐标下降**的简化版——
    对每个 λ 用岭惩罚的 Newton 步迭代到收敛。

    ⚠️ 与 glmnet 的 coxnet 数值上不完全等价（coxnet 用更精细的
    majorization-minimization），变量挑选的**顺序**应当一致，
    但系数值可能有小差异；做预后模型时用它筛变量、再用无惩罚 Cox 重拟合，
    这是标准做法（避免 LASSO 系数有偏）。
    """
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    n, p = X.shape
    Xs, mu_x, sd_x = _standardize(X)
    order = np.argsort(-time)
    Xs, time, event = Xs[order], time[order], event[order]

    # 目标函数按 **-(1/n)·log(偏似然)** 定义：梯度与海森都要除以 n，
    # 否则 λ 的量级与梯度不匹配——实测后果是整条路径压根不收缩
    # （λ 相对梯度太小，n_nonzero 从第一个 λ 起就恒等于 p）。
    if lambdas is None:
        g0 = _cox_grad(Xs, np.zeros(p), time, event) / n
        lam_max = float(np.max(np.abs(g0)) / max(alpha, 1e-9))
        lam_max = lam_max if np.isfinite(lam_max) and lam_max > 0 else 1.0
        lambdas = np.logspace(np.log10(lam_max), np.log10(lam_max * lambda_min_ratio), n_lambda)
    lambdas = np.asarray(lambdas, dtype=float)

    paths = []
    beta = np.zeros(p)
    for lam in lambdas:
        for _ in range(max_iter):
            g = _cox_grad(Xs, beta, time, event) / n
            h = _cox_hess_diag(Xs, beta, time, event) / n
            step = np.zeros(p)
            for j in range(p):
                # 近端 Newton：b_j = soft(b_j - g_j/H_jj, λ/H_jj)
                # 注意 **L1 的 λ 不进分母**——把 λ 加到对角线是 L2 的写法，
                # 混用会让收缩强度随 λ 二次放大，路径形状失真。
                denom = h[j] + lam * (1.0 - alpha)
                denom = denom if denom > 1e-12 else 1e-12
                step[j] = _soft(beta[j] - g[j] / denom, lam * alpha / denom)
            if np.max(np.abs(step - beta)) < tol:
                beta = step
                break
            beta = step
        paths.append(beta.copy() / sd_x)
    paths = np.array(paths)
    return {
        "lambda": lambdas,
        "coef_path": paths,
        "n_nonzero": np.array([int(np.count_nonzero(np.abs(b) > 1e-12)) for b in paths]),
        "alpha": float(alpha),
        "note": "Cox-LASSO 系数有偏；筛变量后建议用无惩罚 Cox 重拟合",
    }


def _cox_grad(X, beta, time, event) -> np.ndarray:
    """负对数偏似然的梯度（Breslow 处理结）。

    ⚠️ **风险集方向**：调用方已按 time **降序**排列（argsort(-time)），
    所以「t_k >= t_i」对应索引 **0..i**，风险集累加必须**正向 cumsum**。
    写成 `cumsum(v[::-1])[::-1]`（累加 i..n-1）等于把风险集取成了
    「生存时间更短的那些人」——梯度方向整体反号，
    实测 LASSO 选出的变量对但**系数符号全反**（真值 +1.2 估成 -0.80）。
    """
    eta = X @ beta
    e = np.exp(eta - np.max(eta))
    cum = np.cumsum(e)                      # Σ_{k: t_k >= t_i} e_k
    c = np.where(cum == 0, 1e-12, cum)
    g = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        s1 = np.cumsum(X[:, j] * e)         # Σ_{k: t_k >= t_i} x_kj e_k
        g[j] = -float(np.sum(event * (X[:, j] - s1 / c)))
    return g


def _cox_hess_diag(X, beta, time, event) -> np.ndarray:
    """负对数偏似然海森的**对角**（坐标下降只需要对角）。"""
    eta = X @ beta
    e = np.exp(eta - np.max(eta))
    cum = np.cumsum(e)
    c = np.where(cum == 0, 1e-12, cum)
    h = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        s1 = np.cumsum(X[:, j] * e)
        s2 = np.cumsum((X[:, j] ** 2) * e)
        h[j] = float(np.sum(event * (s2 / c - (s1 / c) ** 2)))
    return np.maximum(h, 1e-9)


def select_variables(coef: np.ndarray, names: Optional[Sequence[str]] = None,
                     eps: float = 1e-12) -> List[str]:
    """挑出非零系数对应的变量名（LASSO 变量挑选的标准出口）。"""
    idx = np.flatnonzero(np.abs(np.asarray(coef, dtype=float)) > eps)
    if names is None:
        return [int(i) for i in idx]
    return [str(names[i]) for i in idx]
