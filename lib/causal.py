"""分析法 ⑨：观察性数据的因果推断（倾向性评分 / 加权 / 双重稳健 / 中介 / DiD）。

为什么需要这一层：临床数据库（电子病历、医保、登记库）**不是随机分组**的，
直接比较两组的结局差 = 把混杂当效应。这一套方法把「可比性」显式做出来。

方法树（按前提假设从弱到强）：
  · `propensity_score`  + `psm`   — 匹配：只比较可比的个体，丢弃不可比的（会丢样本）
  · `iptw`                        — 加权：全体保留，每人按 PS 加权（稳定权重防极端值）
  · `g_computation`               — 结局回归：需要结局模型正确
  · `aipw`                        — 双重稳健：PS 或结局模型**任一**正确即一致（首选）
  · `did`                         — 双重差分：有前后两期数据时用，吸收不随时间变的混杂
  · `sccs`                        — 自身对照病例系列：疫苗/药物短期暴露的经典设计
  · `mediate`                     — 中介分析：拆直接效应与间接效应

⚠️ 硬前提（不满足时这套方法都救不了）：
  1.  positivity（重叠）：每个协变量组合下都有被治疗和未被治疗的人
  2.  no unmeasured confounding：所有共同原因都被测到并放进 PS 模型
  3.  consistency：干预定义明确、可执行
  诊断必须报告 `smd`（标准化差异），**不要用 p 值判断平衡**（p 值受样本量支配）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import stats as _st


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def propensity_score(X, treatment: Sequence[int], max_iter: int = 200,
                     lr: float = 0.1, l2: float = 1e-4) -> np.ndarray:
    """倾向性评分：logistic 回归估计 P(T=1|X)（牛顿法/梯度下降自实现，无 sklearn）。

    用梯度下降而非 Newton，避免 Hessian 奇异；l2 防止完全分离时系数发散。
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(treatment, dtype=float).ravel()
    n, p = X.shape
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    beta = np.zeros(p)
    for _ in range(max_iter):
        e = _sigmoid(Xs @ beta)
        grad = Xs.T @ (t - e) / n - l2 * beta
        beta = beta + lr * grad
    return _sigmoid(Xs @ beta)


def smd(x: np.ndarray, t: np.ndarray) -> float:
    """标准化差异（Standardized Mean Difference），协变量平衡的标准度量。

    连续变量：(mean1 - mean0) / sqrt((var1+var0)/2)。
    **|SMD| < 0.1 才算平衡**——不是 0.05，也不是 p>0.05。
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=int)
    a, b = x[t == 1], x[t == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt((va + vb) / 2)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def balance_report(X, treatment: Sequence[int], weights: Optional[Sequence[float]] = None,
                   names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """所有协变量的平衡报告（加权前后的 SMD）。"""
    X = np.asarray(X, dtype=float)
    t = np.asarray(treatment, dtype=int)
    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)

    def _wsmd(col: np.ndarray) -> float:
        a = col[t == 1]
        b = col[t == 0]
        wa = w[t == 1]
        wb = w[t == 0]
        if wa.sum() == 0 or wb.sum() == 0:
            return float("nan")
        ma = float(np.sum(wa * a) / wa.sum())
        mb = float(np.sum(wb * b) / wb.sum())
        va = float(np.sum(wa * (a - ma) ** 2) / wa.sum())
        vb = float(np.sum(wb * (b - mb) ** 2) / wb.sum())
        pooled = np.sqrt((va + vb) / 2)
        return (ma - mb) / pooled if pooled > 0 else 0.0

    rows = []
    for j in range(X.shape[1]):
        rows.append({
            "covariate": names[j] if names is not None else f"x{j}",
            "smd_unweighted": smd(X[:, j], t),
            "smd_weighted": _wsmd(X[:, j]) if weights is not None else float("nan"),
        })
    df = pd.DataFrame(rows)
    df["balanced_after"] = df["smd_weighted"].abs() < 0.1 if weights is not None else False
    return df


def psm(X, treatment: Sequence[int], outcome: Sequence[float],
        caliper: Optional[float] = 0.2, ratio: int = 1,
        ps: Optional[Sequence[float]] = None, seed: int = 0) -> Dict[str, Any]:
    """倾向性评分匹配（最近邻 + 卡尺）。

    :param caliper: 卡尺 = caliper × PS 的标准差；**必须给**，不给会出现「远亲匹配」
    :param ratio: 1:1（默认）或 1:N
    :return: ATT（处理组的平均处理效应）+ 匹配后的平衡报告
    """
    t = np.asarray(treatment, dtype=int)
    y = np.asarray(outcome, dtype=float)
    e = np.asarray(ps, dtype=float) if ps is not None else propensity_score(X, t)
    rng = np.random.default_rng(seed)
    treated = np.flatnonzero(t == 1)
    controls = np.flatnonzero(t == 0)
    scale = caliper * float(np.std(e)) if caliper is not None else np.inf

    order = rng.permutation(treated)
    pairs: List[Tuple[int, int]] = []
    used = set()
    for i in order:
        d = np.abs(e[controls] - e[i])
        cand = [(d[j], controls[j]) for j in range(len(controls))
                if controls[j] not in used and d[j] <= scale]
        if not cand:
            continue
        cand.sort(key=lambda z: z[0])
        for _, c in cand[:ratio]:
            pairs.append((int(i), int(c)))
            used.add(int(c))
    if not pairs:
        return {"n_matched_pairs": 0, "att": float("nan"),
                "note": "没有可匹配的对照——多半是重叠性（positivity）不成立"}
    ti = np.array([p[0] for p in pairs])
    ci = np.array([p[1] for p in pairs])
    att = float(np.mean(y[ti] - y[ci]))
    se = float(np.std(y[ti] - y[ci], ddof=1) / np.sqrt(len(pairs)))
    ci_low, ci_high = att - 1.96 * se, att + 1.96 * se
    return {
        "n_matched_pairs": int(len(pairs)),
        "n_treated_unmatched": int(len(treated) - len(pairs)),
        "att": att, "se": se, "ci_low": ci_low, "ci_high": ci_high,
        "pvalue": float(2 * (1 - _st.norm.cdf(abs(att / se)))) if se > 0 else float("nan"),
        "caliper": caliper,
        "weights": _matched_weights(len(t), ti, ci),
        "estimand": "ATT（处理组的平均处理效应）",
        "note": "匹配会丢样本；结论只适用于被匹配上的那一批人",
    }


def _matched_weights(n: int, ti: np.ndarray, ci: np.ndarray) -> np.ndarray:
    w = np.zeros(n)
    w[ti] = 1.0
    w[ci] = 1.0
    return w


def iptw(X, treatment: Sequence[int], outcome: Sequence[float],
         ps: Optional[Sequence[float]] = None, stabilized: bool = True,
         trim: Optional[float] = 0.01) -> Dict[str, Any]:
    """逆概率加权（IPTW）。

    stabilized=True 用边缘概率做分子，权重更稳（推荐）。
    trim 做**尾部截断**（不是删除样本），抑制极端权重导致的方差爆炸。
    """
    t = np.asarray(treatment, dtype=int)
    y = np.asarray(outcome, dtype=float)
    e = np.asarray(ps, dtype=float) if ps is not None else propensity_score(X, t)
    e = np.clip(e, 1e-6, 1 - 1e-6)
    if stabilized:
        p = float(np.mean(t))
        w = np.where(t == 1, p / e, (1 - p) / (1 - e))
    else:
        w = np.where(t == 1, 1 / e, 1 / (1 - e))
    if trim is not None:
        lo, hi = np.quantile(w, trim), np.quantile(w, 1 - trim)
        w = np.clip(w, lo, hi)
    m1 = float(np.sum(w[t == 1] * y[t == 1]) / np.sum(w[t == 1]))
    m0 = float(np.sum(w[t == 0] * y[t == 0]) / np.sum(w[t == 0]))
    ate = m1 - m0
    # 稳健（三明治）方差的简化：按加权样本量近似
    n_eff = float(np.sum(w) ** 2 / np.sum(w ** 2))
    var1 = float(np.sum(w[t == 1] ** 2 * (y[t == 1] - m1) ** 2) / np.sum(w[t == 1]) ** 2)
    var0 = float(np.sum(w[t == 0] ** 2 * (y[t == 0] - m0) ** 2) / np.sum(w[t == 0]) ** 2)
    se = float(np.sqrt(var1 + var0))
    return {
        "ate": float(ate), "se": se,
        "ci_low": ate - 1.96 * se, "ci_high": ate + 1.96 * se,
        "pvalue": float(2 * (1 - _st.norm.cdf(abs(ate / se)))) if se > 0 else float("nan"),
        "weights": w,
        "weight_range": [float(w.min()), float(w.max())],
        "effective_n": n_eff,
        "stabilized": stabilized, "trim": trim,
        "estimand": "ATE（人群平均处理效应）",
    }


def g_computation(X, treatment: Sequence[int], outcome: Sequence[float],
                  binary_outcome: bool = False) -> Dict[str, Any]:
    """G-computation（标准化/参数化 g-formula）：拟合结局模型，全员反事实预测。

    与 IPTW 的区别：它用的是**结局模型**而非处理模型。
    结局模型错就错，没有第二次机会——所以首选 AIPW。
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(treatment, dtype=int)
    y = np.asarray(outcome, dtype=float)
    Xd = np.column_stack([X, t])
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xd)), Xd]), y, rcond=None)

    def _predict(t_set: int) -> np.ndarray:
        Z = np.column_stack([np.ones(len(X)), X, np.full(len(X), float(t_set))])
        pred = Z @ beta
        return _sigmoid(pred) if binary_outcome else pred

    m1 = float(np.mean(_predict(1)))
    m0 = float(np.mean(_predict(0)))
    return {"ate": m1 - m0, "mean_y1": m1, "mean_y0": m0,
            "coefficients": beta, "binary_outcome": binary_outcome,
            "note": "依赖结局模型正确设定；无双重稳健保护"}


def aipw(X, treatment: Sequence[int], outcome: Sequence[float],
         ps: Optional[Sequence[float]] = None, n_folds: int = 5,
         seed: int = 0) -> Dict[str, Any]:
    """AIPW / 增广逆概率加权（双重稳健，推荐默认）。

    估计量（Robins-Rotnitzky-Zhao）：
      ψ = 1/n Σ [ (T·(Y-μ1(X))/e(X) - (1-T)·(Y-μ0(X))/(1-e(X))) + (μ1(X) - μ0(X)) ]

    双重稳健的含义：PS 模型**或**结局模型**任一**正确 → 估计一致。
    用**交叉拟合**（cross-fitting）：在某个折里估计的 nuisance 参数，
    只用来预测另一折的样本——避免过拟合把 nuisance 误差带进估计量。

    ⚠️ 这里的结局模型是**线性**的；二分类结局应当换用 logistic，
       得到的是风险差而非 OR（风险差才是因果上可解释的量）。
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(treatment, dtype=int).ravel()
    y = np.asarray(outcome, dtype=float).ravel()
    n = X.shape[0]
    e = np.asarray(ps, dtype=float) if ps is not None else propensity_score(X, t)
    e = np.clip(e, 1e-6, 1 - 1e-6)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    for fold in np.array_split(idx, n_folds):
        tr = np.setdiff1d(idx, fold)
        for arm in (1, 0):
            m = t[tr] == arm
            if m.sum() < X.shape[1] + 2:
                # 该折里某个臂样本太少：退回用全局均值，避免 lstsq 出病态解
                fill = float(np.mean(y[tr][m])) if m.sum() else float(np.mean(y[tr]))
                out = np.full(len(fold), fill)
            else:
                Z = np.column_stack([np.ones(m.sum()), X[tr][m]])
                b, *_ = np.linalg.lstsq(Z, y[tr][m], rcond=None)
                Ze = np.column_stack([np.ones(len(fold)), X[fold]])
                out = Ze @ b
            if arm == 1:
                mu1[fold] = out
            else:
                mu0[fold] = out

    psi = (t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)) + (mu1 - mu0)
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    return {
        "ate": ate, "se": se,
        "ci_low": ate - 1.96 * se, "ci_high": ate + 1.96 * se,
        "pvalue": float(2 * (1 - _st.norm.cdf(abs(ate / se)))) if se > 0 else float("nan"),
        "influence_function": psi,
        "n_folds": n_folds,
        "estimand": "ATE",
        "note": "双重稳健 + 交叉拟合；PS 或结局模型任一正确即一致",
    }


def did(outcome_pre: Sequence[float], outcome_post: Sequence[float],
        group: Sequence[int]) -> Dict[str, Any]:
    """双重差分 DiD：(处理后-处理前) 的组间差。

    前提：**平行趋势假设**（若无干预，两组的变化趋势相同）。
    该假设不可直接检验，只能用处理前的多期数据做安慰剂检验。
    """
    pre = np.asarray(outcome_pre, dtype=float)
    post = np.asarray(outcome_post, dtype=float)
    g = np.asarray(group, dtype=int)
    d_pre = float(np.mean(pre[g == 1]) - np.mean(pre[g == 0]))
    d_post = float(np.mean(post[g == 1]) - np.mean(post[g == 0]))
    delta = d_post - d_pre
    # 方差：用个体层面差分的两样本 t 检验
    d_ind = (post - pre)
    a, b = d_ind[g == 1], d_ind[g == 0]
    stat, p = _st.ttest_ind(a, b, equal_var=False)
    se = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
    return {
        "att": float(delta), "se": se,
        "ci_low": delta - 1.96 * se, "ci_high": delta + 1.96 * se,
        "pvalue": float(p),
        "diff_pre": d_pre, "diff_post": d_post,
        "assumption": "平行趋势（不可直接检验，需处理前多期安慰剂检验）",
    }


def mediate(X, treatment: Sequence[int], mediator: Sequence[float],
            outcome: Sequence[float]) -> Dict[str, Any]:
    """中介分析（Baron-Kenny / 差值法）。

    分解：总效应 = 直接效应 + 间接效应（a×b）。
      a：T → M 的效应（控制协变量）
      b：M → Y 的效应（控制 T 与协变量）
      c：T → Y 总效应；c'：T → Y 直接效应（控制 M）
    间接效应的 CI 用 **bootstrap**——a×b 的分布不是正态，
    用 Sobel 的正态近似在小样本下严重失真。
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(treatment, dtype=float).ravel()
    m = np.asarray(mediator, dtype=float).ravel()
    y = np.asarray(outcome, dtype=float).ravel()

    def _ols(A: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        Z = np.column_stack([np.ones(len(A)), A])
        coef, *_ = np.linalg.lstsq(Z, bvec, rcond=None)
        return coef

    # a: M ~ T + X
    ca = _ols(np.column_stack([t, X]), m)
    a = float(ca[1])
    # b, c': Y ~ T + M + X
    cb = _ols(np.column_stack([t, m, X]), y)
    c_prime, b = float(cb[1]), float(cb[2])
    # c: Y ~ T + X
    cc = _ols(np.column_stack([t, X]), y)
    c = float(cc[1])
    indirect = a * b
    return {
        "a": a, "b": b, "total_effect": c, "direct_effect": c_prime,
        "indirect_effect": indirect,
        "proportion_mediated": indirect / c if c != 0 else float("nan"),
        "caution": "间接效应的 CI 需 bootstrap（a×b 非正态）；Sobel 正态近似在小样本下失真",
    }


def bootstrap_indirect(X, treatment, mediator, outcome, n_boot: int = 2000,
                       seed: int = 0, level: float = 0.95) -> Dict[str, Any]:
    """中介效应的 bootstrap 置信区间（百分位法）。"""
    X = np.asarray(X, dtype=float)
    t = np.asarray(treatment)
    m = np.asarray(mediator)
    y = np.asarray(outcome)
    n = len(t)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            r = mediate(X[idx], t[idx], m[idx], y[idx])
            vals.append(r["indirect_effect"])
        except Exception:  # noqa: BLE001 —— 重抽样可能碰到秩不足，跳过该次
            continue
    vals = np.array(vals, dtype=float)
    alpha = (1 - level) / 2
    return {
        "n_boot": int(len(vals)),
        "ci_low": float(np.quantile(vals, alpha)),
        "ci_high": float(np.quantile(vals, 1 - alpha)),
        "point": float(np.mean(vals)),
        "method": "percentile bootstrap",
    }


def negative_control_outcome(X, treatment: Sequence[int], outcome: Sequence[float],
                             control_outcome: Sequence[float]) -> Dict[str, Any]:
    """阴性对照结局检验：理论上不受处理影响的结局，若也「有效」说明有残余混杂。

    这是观察性研究里**最能暴露残余混杂**的诊断之一：
    若对照结局的效应显著 ≠ 0，多半是未测混杂或选择偏倚在作祟。
    """
    t = np.asarray(treatment, dtype=int)
    yc = np.asarray(control_outcome, dtype=float)
    a, b = yc[t == 1], yc[t == 0]
    stat, p = _st.ttest_ind(a, b, equal_var=False)
    return {
        "effect_on_control": float(np.mean(a) - np.mean(b)),
        "pvalue": float(p),
        "flag": "⚠️ 显著 ≠0 提示存在残余混杂/选择偏倚" if p < 0.05 else "未见明显残余混杂信号",
        "caution": "不显著不等于已排除混杂（功效有限）",
    }
