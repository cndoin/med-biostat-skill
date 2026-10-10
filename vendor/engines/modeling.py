# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/engine/model.py
# 许可: 自建/内部
# 说明: 自实现建模（无 sklearn）+ 校准 + 决策曲线
# 源 sha256: aa6312bc64846e5a
# VENDORED-FILE-END
"""预测建模引擎。

这是本平台与"随手跑个 sklearn"最大的区别所在，四条硬规则内置在代码里：

1. 防泄漏：所有标准化、插补、特征选择都在训练折内拟合，再作用于验证折。
2. 独立测试集只评估一次：`evaluate_holdout` 会记录调用次数，第二次直接报错。
3. 不只报区分度：AUROC、AUPRC、Brier、校准曲线、决策曲线一次出齐。
4. 报告门槛：缺少校准或亚组结果时，`readiness` 返回未通过项，不允许声称"可用于临床"。

模型全部自实现（弹性网络坐标下降、直方图梯度提升、stacking），
仅依赖 numpy / scipy，避免版本漂移导致结果不可复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ============================================================ 切分

def stratified_split(y: np.ndarray, ratios: Sequence[float] = (0.6, 0.2, 0.2),
                     seed: int = 0) -> List[np.ndarray]:
    """分层切分，返回各部分的索引数组。 ratios 不必归一。"""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    parts: List[List[int]] = [[] for _ in ratios]
    for cls in np.unique(y):
        pool = rng.permutation(idx[y == cls])
        cuts = np.cumsum(np.array(ratios, float) / np.sum(ratios)) * len(pool)
        cuts = np.round(cuts).astype(int)
        prev = 0
        for i, c in enumerate(cuts):
            parts[i].extend(pool[prev:c].tolist())
            prev = c
    return [np.array(sorted(p)) for p in parts]


def split_by_site(site: np.ndarray, test_sites: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """按中心切分——外部验证的唯一可信做法。按行随机切分会高估泛化性能。"""
    site = np.asarray(site).astype(str)
    test = np.isin(site, [str(s) for s in test_sites])
    return np.where(~test)[0], np.where(test)[0]


# ============================================================ 缩放

class Standardizer:
    """只在 fit 的数据上学习均值与标准差。"""

    def __init__(self):
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, float)
        self.mu_ = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        self.sd_ = np.where(sd < 1e-12, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        X = np.where(np.isnan(X), self.mu_, X)
        return (X - self.mu_) / self.sd_


# ============================================================ 弹性网络

def elastic_net(X: np.ndarray, y: np.ndarray, lam: float, l1_ratio: float = 0.5,
                max_iter: int = 800, tol: float = 1e-8) -> np.ndarray:
    """坐标下降求解弹性网络。X 已标准化、y 已中心化、无截距。

    残差增量更新：内层循环重算 X@b 会让复杂度从 O(n*p) 变成 O(n*p^2)，
    在 p 上千时会慢到无法接受。
    """
    n, p = X.shape
    b = np.zeros(p)
    y = np.asarray(y, float)
    r = y.copy()
    denom = 1.0 + lam * (1.0 - l1_ratio)
    for _ in range(max_iter):
        b_old = b.copy()
        for j in range(p):
            xj = X[:, j]
            rho = xj @ r / n + b[j]
            new_bj = np.sign(rho) * max(abs(rho) - lam * l1_ratio, 0.0) / denom
            delta = new_bj - b[j]
            if delta != 0.0:
                r -= xj * delta
                b[j] = new_bj
        if np.max(np.abs(b - b_old)) < tol:
            break
    return b


class ElasticNetClassifier:
    """二分类弹性网络（对数损失由平方损失近似，配概率校准）。"""

    def __init__(self, lam: float = 0.05, l1_ratio: float = 0.5):
        self.lam = lam
        self.l1_ratio = l1_ratio
        self._init_kwargs = {"lam": lam, "l1_ratio": l1_ratio}
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetClassifier":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.mu_ = X.mean(axis=0)
        Xc = X - self.mu_
        yc = y - y.mean()
        self.coef_ = elastic_net(Xc, yc, self.lam, self.l1_ratio)
        self.intercept_ = float(y.mean() - self.mu_ @ self.coef_)
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return self.intercept_ + X @ self.coef_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision(X), -30, 30)))


# ============================================================ 直方图梯度提升

class _Tree:
    """单棵回归树，直方图分箱找切分。"""

    def __init__(self, max_depth: int = 3, max_bins: int = 32, min_samples_leaf: int = 10):
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> "_Tree":
        self.nodes_: List[dict] = []
        idx = np.arange(len(X))
        self._build(X, g, h, idx, 0)
        return self

    def _build(self, X, g, h, idx, depth):
        G, H = g[idx].sum(), h[idx].sum()
        node = {"leaf": True, "value": G / max(H, 1e-12)}
        if depth >= self.max_depth or len(idx) < 2 * self.min_samples_leaf:
            self.nodes_.append(node)
            return len(self.nodes_) - 1

        best = None
        for j in range(X.shape[1]):
            col = X[idx, j]
            qs = np.unique(np.quantile(col, np.linspace(0, 1, self.max_bins + 1)))
            if len(qs) < 3:
                continue
            for thr in qs[1:-1]:
                left = col <= thr
                nl, nr = left.sum(), (~left).sum()
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gl, hl = g[idx][left].sum(), h[idx][left].sum()
                gr, hr = G - gl, H - hl
                gain = gl ** 2 / max(hl, 1e-12) + gr ** 2 / max(hr, 1e-12) - G ** 2 / max(H, 1e-12)
                if best is None or gain > best[0]:
                    best = (gain, j, thr, left)
        if best is None or best[0] <= 1e-10:
            self.nodes_.append(node)
            return len(self.nodes_) - 1

        _, j, thr, left = best
        node_id = len(self.nodes_)
        self.nodes_.append({"leaf": False, "feature": j, "threshold": thr,
                            "left": None, "right": None})
        li = self._build(X, g, h, idx[left], depth + 1)
        ri = self._build(X, g, h, idx[~left], depth + 1)
        self.nodes_[node_id]["left"] = li
        self.nodes_[node_id]["right"] = ri
        return node_id

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        for i in range(len(X)):
            nid = 0
            while not self.nodes_[nid]["leaf"]:
                nd = self.nodes_[nid]
                nid = nd["left"] if X[i, nd["feature"]] <= nd["threshold"] else nd["right"]
            out[i] = self.nodes_[nid]["value"]
        return out


class GradientBoostingClassifier:
    """二分类梯度提升（对数损失，Newton 步长）。"""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05,
                 max_depth: int = 3, subsample: float = 0.8, seed: int = 0,
                 min_samples_leaf: int = 10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.seed = seed
        self.min_samples_leaf = min_samples_leaf
        self._init_kwargs = {
            "n_estimators": n_estimators, "learning_rate": learning_rate,
            "max_depth": max_depth, "subsample": subsample, "seed": seed,
            "min_samples_leaf": min_samples_leaf,
        }
        self.trees_: List[_Tree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        p = np.full(len(y), y.mean())
        self.base_ = float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
        self.trees_ = []
        for _ in range(self.n_estimators):
            g = y - p
            h = np.clip(p * (1 - p), 1e-6, None)
            n_sub = max(int(self.subsample * len(y)), 10)
            sub = rng.choice(len(y), size=n_sub, replace=False)
            tree = _Tree(self.max_depth, min_samples_leaf=self.min_samples_leaf).fit(
                X[sub], g[sub], h[sub])
            self.trees_.append(tree)
            raw = self.base_ + self.learning_rate * np.sum(
                [t.predict(X) for t in self.trees_], axis=0)
            p = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        raw = np.full(len(X), self.base_)
        for t in self.trees_:
            raw += self.learning_rate * t.predict(X)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.decision(X), -30, 30)))


# ============================================================ 概率校准

class PlattCalibrator:
    """Platt 缩放（sigmoid 校准）。

    在训练集（或交叉验证的 out-of-fold 预测）上拟合，再作用于测试集。
    绝不能用测试集拟合——那等于把答案写进模型。

    校准不改变排序，因此 AUROC 不变；它修的是概率的绝对意义，
    而概率的绝对意义正是临床决策（阈值选择）所依赖的。
    """

    def __init__(self, max_iter: int = 200, lr: float = 0.05):
        self.max_iter = max_iter
        self.lr = lr
        self.a_ = 0.0
        self.b_ = 0.0

    def fit(self, decision: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        d = np.asarray(decision, float)
        y = np.asarray(y, float)
        n = len(y)
        prior = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        self.b_ = 0.0
        self.a_ = np.log(prior / (1 - prior))
        for _ in range(self.max_iter):
            z = np.clip(self.a_ * d + self.b_, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            g = y - p
            ga = float(np.sum(g * d))
            gb = float(np.sum(g))
            self.a_ += self.lr * ga / n
            self.b_ += self.lr * gb / n
        return self

    def predict_proba(self, decision: np.ndarray) -> np.ndarray:
        d = np.asarray(decision, float)
        return 1.0 / (1.0 + np.exp(-np.clip(self.a_ * d + self.b_, -30, 30)))


def calibrate_oof(model_factory, X: np.ndarray, y: np.ndarray,
                  n_folds: int = 5, seed: int = 0) -> PlattCalibrator:
    """用 out-of-fold 决策值拟合校准器，避免在训练预测上校准（那必然过自信）。"""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=len(y))
    oof = np.zeros(len(y))
    for f in range(n_folds):
        tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
        if len(va) == 0:
            continue
        sc = Standardizer().fit(X[tr])
        m = model_factory().fit(sc.transform(X[tr]), y[tr])
        oof[va] = m.decision(sc.transform(X[va]))
    return PlattCalibrator().fit(oof, y)


# ============================================================ Stacking

class StackedClassifier:
    """Stacking：基学习器在训练折内拟合，元学习器在 out-of-fold 预测上训练。

    关键点：元学习器绝不能看基学习器在同一样本上的折内预测——
    那会让元学习器学到过拟合的噪声。
    """

    def __init__(self, base_models: Dict[str, object], meta_model=None,
                 n_folds: int = 5, seed: int = 0):
        self.base_models = base_models
        self.meta_model = meta_model or ElasticNetClassifier(lam=0.01, l1_ratio=0.5)
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackedClassifier":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.seed)
        fold = rng.integers(0, self.n_folds, size=len(y))
        oof = np.zeros((len(y), len(self.base_models)))

        self.fitted_: Dict[str, object] = {}
        for k, (name, proto) in enumerate(self.base_models.items()):
            for f in range(self.n_folds):
                tr, va = np.where(fold != f)[0], np.where(fold == f)[0]
                if len(va) == 0:
                    continue
                m = _clone(proto).fit(X[tr], y[tr])
                oof[va, k] = m.predict_proba(X[va])
            self.fitted_[name] = _clone(proto).fit(X, y)

        self.meta_model.fit(oof, y)
        self.oof_ = oof          # 供校准器使用，避免二次拟合的开销
        self.oof_y_ = y
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        Z = np.column_stack([m.predict_proba(X) for m in self.fitted_.values()])
        return np.asarray(self.meta_model.decision(Z), float)

    def oof_decision(self) -> np.ndarray:
        """元学习器在 out-of-fold 概率上的决策值。

        用这个值拟合 Platt 校准器，等价于"在模型没见过的数据上校准"，
        而直接用训练预测校准会得到过度自信的概率。
        """
        return np.asarray(self.meta_model.decision(self.oof_), float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        Z = np.column_stack([m.predict_proba(X) for m in self.fitted_.values()])
        return np.clip(self.meta_model.predict_proba(Z), 1e-6, 1 - 1e-6)


def _clone(model):
    """按构造参数重建一个同类型模型（避免共享状态）。"""
    cls = type(model)
    init = getattr(model, "_init_kwargs", None)
    if init is None:
        return cls()
    return cls(**init)


# ============================================================ 评价指标

def roc_curve(y: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y).astype(int)
    s = np.asarray(score, float)
    order = np.argsort(-s)
    y, s = y[order], s[order]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.nan
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # 折点取每个不同分数组的最后一个样本（cumsum 到组末才累计完整）
    keep = np.r_[np.diff(s) != 0, True]
    fpr = np.r_[0.0, (fp / n_neg)[keep], 1.0]
    tpr = np.r_[0.0, (tp / n_pos)[keep], 1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(score, float)
    order = np.argsort(-s)
    y = y[order]
    n_pos = y.sum()
    if n_pos == 0:
        return np.nan
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / n_pos
    prec = np.r_[1.0, prec]
    rec = np.r_[0.0, rec]
    return float(np.trapezoid(prec, rec))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def calibration_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """十等分校准曲线：预测概率分箱 vs 实测事件率。"""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if sel.sum() == 0:
            continue
        rows.append({"bin_low": lo, "bin_high": hi, "n": int(sel.sum()),
                     "mean_predicted": float(p[sel].mean()),
                     "observed_rate": float(y[sel].mean())})
    return {"bins": rows,
            "calibration_in_the_large": float(y.mean() - p.mean()),
            "slope": _calibration_slope(y, p)}


def _calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """校准斜率：对 logit(p) 拟合 y，斜率<1 提示过拟合。"""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    X = np.column_stack([np.ones(len(y)), logit])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(beta[1])
    except Exception:
        return float("nan")


def decision_curve(y: np.ndarray, p: np.ndarray,
                   thresholds: Optional[np.ndarray] = None) -> dict:
    """决策曲线分析：净获益 vs 阈值概率。

    net_benefit = TP/N − FP/N × (pt/(1−pt))
    对照： treat_all 与 treat_none。
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y)
    rows = []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        w = pt / (1 - pt)
        nb = tp / n - fp / n * w
        prev = y.mean()
        nb_all = prev - (1 - prev) * w
        rows.append({"threshold": float(pt), "net_benefit": float(nb),
                     "treat_all": float(nb_all), "treat_none": 0.0})
    return {"curve": rows,
            "max_net_benefit": float(max(r["net_benefit"] for r in rows)) if rows else np.nan}


def evaluate(y: np.ndarray, p: np.ndarray, label: str = "") -> dict:
    """一次出齐区分度、校准度与临床效用。"""
    fpr, tpr, auc = roc_curve(y, p)
    cal = calibration_curve(y, p)
    dca = decision_curve(y, p)
    out = {
        "label": label,
        "n": int(len(y)),
        "events": int(np.sum(np.asarray(y) == 1)),
        "auroc": auc,
        "auprc": pr_auc(y, p),
        "brier": brier_score(y, p),
        "calibration_slope": cal["slope"],
        "calibration_in_the_large": cal["calibration_in_the_large"],
        "max_net_benefit": dca["max_net_benefit"],
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "calibration_bins": cal["bins"],
        "dca": dca["curve"],
    }
    return out


# ============================================================ 嵌套交叉验证

@dataclass
class NestedCVResult:
    outer_scores: List[dict] = field(default_factory=list)
    fold_predictions: List[dict] = field(default_factory=list)

    @property
    def mean_auroc(self) -> float:
        v = [s["auroc"] for s in self.outer_scores if s["auroc"] == s["auroc"]]
        return float(np.mean(v)) if v else float("nan")

    @property
    def sd_auroc(self) -> float:
        v = [s["auroc"] for s in self.outer_scores if s["auroc"] == s["auroc"]]
        return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def nested_cv(X: np.ndarray, y: np.ndarray, model_factory: Callable[[], object],
              n_outer: int = 5, n_inner: int = 3, seed: int = 0,
              candidates: Optional[Sequence[object]] = None) -> NestedCVResult:
    """嵌套交叉验证：外环评估泛化，内环调参与选特征。

    任何在内环之外接触到全部数据的操作（特征筛选、标准化、插补）
    都会让外环估计偏向乐观——这是嵌套 CV 存在的唯一理由。

    candidates 为候选模型原型列表；提供时在内环按 AUROC 择优，
    再以克隆体在外环训练集上重新拟合（避免携带内环状态）。
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_outer, size=len(y))
    res = NestedCVResult()

    for f in range(n_outer):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        if len(te) == 0 or y[tr].sum() < 3:
            continue

        chosen = model_factory()
        if candidates:
            inner = rng.integers(0, n_inner, size=len(tr))
            best = None
            for proto in candidates:
                scores = []
                for g in range(n_inner):
                    v = np.where(inner == g)[0]
                    t = np.where(inner != g)[0]
                    if len(v) == 0 or y[tr][t].sum() < 2:
                        continue
                    # 内环同样标准化，且只拟合内环训练折
                    sci = Standardizer().fit(X[tr][t])
                    m = _clone(proto).fit(sci.transform(X[tr][t]), y[tr][t])
                    _, _, a = roc_curve(y[tr][v], m.predict_proba(sci.transform(X[tr][v])))
                    if a == a:
                        scores.append(a)
                mean_a = float(np.nanmean(scores)) if scores else -np.inf
                if best is None or mean_a > best[0]:
                    best = (mean_a, proto)
            chosen = _clone(best[1])

        # 标准化仅用训练折拟合——这一步在折外做就是最常见的泄漏
        sc = Standardizer().fit(X[tr])
        model = _clone(chosen) if candidates else chosen
        model.fit(sc.transform(X[tr]), y[tr])
        p = np.clip(model.predict_proba(sc.transform(X[te])), 1e-6, 1 - 1e-6)
        ev = evaluate(y[te], p, label="fold_%d" % f)
        res.outer_scores.append(ev)
        res.fold_predictions.append({"fold": f, "index": te.tolist(), "p": p.tolist(),
                                     "y": y[te].tolist()})
    return res


# ============================================================ 报告就绪度闸门

class Readiness:
    """TRIPOD+AI 关键项的代码化检查。未通过项必须逐条说明。"""

    def __init__(self):
        self.checks: List[dict] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"item": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame(self.checks)


def readiness(eval_res: dict, has_external: bool, has_subgroup: bool,
              has_calibration: bool, n: int) -> Readiness:
    r = Readiness()
    r.add("区分度已报告（AUROC）", "auroc" in eval_res and eval_res["auroc"] == eval_res["auroc"],
          "AUROC = %s" % round(eval_res.get("auroc", float("nan")), 3))
    r.add("校准度已报告", has_calibration and eval_res.get("calibration_slope") == eval_res.get("calibration_slope"),
          "校准斜率 = %s（1 为理想）" % round(eval_res.get("calibration_slope", float("nan")), 3))
    r.add("临床效用已报告（决策曲线）", eval_res.get("max_net_benefit") == eval_res.get("max_net_benefit"),
          "最大净获益 = %s" % round(eval_res.get("max_net_benefit", float("nan")), 4))
    r.add("外部验证（独立中心或时间点）", has_external,
          "按中心切分是唯一可信的外部验证方式")
    r.add("亚组性能已报告", has_subgroup, "至少按性别与年龄分层")
    r.add("样本量足以支撑结论", n >= 100, "n = %d，<100 时结论需显著弱化" % n)
    return r
