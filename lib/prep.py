# -*- coding: utf-8 -*-
"""prep.py —— 数据清洗与变量挑选（**全部是确定性计算**）

为什么单独成模块：生信/临床分析里，**结论错了有一半不是模型错，是数据错**
（缺失被 silently 丢掉、离群点把均值拉飞、高度相关的变量被重复计入）。
这些步骤必须在建模**之前**显式执行并留下记录，否则后面的 p 值无从解释。

**红线**：本模块不做任何「看上去更合理」的自动修补。每一步只报告事实 +
给出明确规则的处理结果；填补 / 剔除都要求调用方显式选择策略，
绝不默认丢行（默认丢行是临床数据里最常见的静默偏差来源）。
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

__all__ = [
    "load_table", "missing_report", "outlier_report",
    "drop_low_variance", "correlation_prune", "vif_report",
    "standardize", "one_hot", "prep_pipeline",
]


def _as_2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"需要 1/2 维数据，得到 {a.ndim} 维")
    return a


def load_table(path: str, sep: str | None = None, header: bool = True):
    """读取表格（csv/tsv）。纯标准库实现，不依赖 pandas。

    返回 (列名 or None, 字符串矩阵)。数值解释交给后续函数，
    这里**不做隐式类型推断**——隐式推断会把 "07" 变成 7、把 "NA" 变成 nan。
    """
    import csv

    if sep is None:
        low = str(path).lower()
        sep = "\t" if low.endswith((".tsv", ".tab", ".txt")) else ","
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=sep))
    if not rows:
        return None, np.empty((0, 0), dtype=object)
    if header:
        return [c.strip() for c in rows[0]], np.asarray(rows[1:], dtype=object)
    return None, np.asarray(rows, dtype=object)


def missing_report(matrix) -> Dict[str, Any]:
    """缺失情况报告：逐列缺失率 + 缺失模式分布 + 完整行数。

    不填补、不丢行——只报告。是否填补必须由调用方决定。
    """
    a = _as_2d(matrix) if not isinstance(matrix, np.ndarray) or matrix.dtype != object else matrix
    if isinstance(a, np.ndarray) and a.dtype == object:
        mask = np.array([[str(v).strip() in ("", "NA", "N/A", "NaN", "nan",
                                             "null", "None", ".", "-")
                          for v in row] for row in a], dtype=bool)
        n_rows, n_cols = mask.shape
    else:
        mask = ~np.isfinite(a.astype(float))
        n_rows, n_cols = mask.shape
    col_rate = mask.mean(axis=0) if n_rows else np.zeros(n_cols)
    row_complete = ~mask.any(axis=1)
    patterns: Dict[str, int] = {}
    for row in mask:
        key = "".join("1" if v else "0" for v in row)
        patterns[key] = patterns.get(key, 0) + 1
    top = sorted(patterns.items(), key=lambda kv: -kv[1])[:5]
    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "missing_per_col": [float(x) for x in col_rate],
        "n_complete_rows": int(row_complete.sum()),
        "complete_row_pct": float(row_complete.mean()) if n_rows else 0.0,
        "cols_with_any_missing": [int(i) for i in np.nonzero(col_rate)[0]],
        "top_missing_patterns": top,
        "n_patterns": len(patterns),
    }


def outlier_report(x: Sequence[float], method: str = "iqr",
                   k: float | None = None) -> Dict[str, Any]:
    """离群点报告（三种口径，口径必须随结果一起输出，否则数字无从解读）。

    iqr    : 低于 Q1-k*IQR 或高于 Q3+k*IQR（Tukey）
    zscore : |x-mean|/sd > k
    mad    : |x-median|/(1.4826*MAD) > k  —— **对离群点本身稳健**，推荐首选

    `k=None` 时按各方法的**惯例**取默认，而不是复用同一个数：
      iqr→1.5（Tukey）、zscore→3.0、mad→3.5。
    为什么不统一：三者量纲不同。实测把 Tukey 的 1.5 用到 z-score 上，
    **均匀分布的两端也会被判成离群**（占比约 13%），那是默认值害的，不是数据有问题。
    """
    DEFAULT_K = {"iqr": 1.5, "zscore": 3.0, "mad": 3.5}
    if method not in DEFAULT_K:
        raise ValueError(f"未知 method={method!r}；可用 {sorted(DEFAULT_K)}")
    if k is None:
        k = DEFAULT_K[method]
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        raise ValueError("空序列或非有限值过多")
    if method == "iqr":
        q1, q3 = np.percentile(a, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
    elif method == "zscore":
        m, s = a.mean(), a.std(ddof=1) if a.size > 1 else 0.0
        if s == 0:
            lo, hi = -np.inf, np.inf
        else:
            lo, hi = m - k * s, m + k * s
    elif method == "mad":
        med = np.median(a)
        mad = np.median(np.abs(a - med))
        scale = 1.4826 * mad
        if scale == 0:
            lo, hi = -np.inf, np.inf
        else:
            lo, hi = med - k * scale, med + k * scale
    is_out = (a < lo) | (a > hi)
    return {
        "method": method, "k": float(k),
        "lower_bound": float(lo) if np.isfinite(lo) else None,
        "upper_bound": float(hi) if np.isfinite(hi) else None,
        "n": int(a.size),
        "n_outliers": int(is_out.sum()),
        "outlier_pct": float(is_out.mean()),
        "outlier_idx": [int(i) for i in np.nonzero(is_out)[0]],
        "outlier_values": [float(v) for v in a[is_out]],
    }


def drop_low_variance(matrix, threshold: float = 0.0):
    """剔除（近）常数列。threshold=0 只剔除完全无变化的列。"""
    a = _as_2d(matrix)
    if a.shape[0] < 2:
        return a, list(range(a.shape[1]))
    var = np.nanvar(a, axis=0, ddof=1)
    keep = np.nonzero(var > threshold)[0]
    return a[:, keep], [int(i) for i in keep]


def correlation_prune(matrix, threshold: float = 0.9) -> Dict[str, Any]:
    """相关去冗余：|r| > threshold 的成对列，保留先出现的那个。

    为什么做：高度共线的变量同时进入回归会让系数符号乱跳、p 值虚高。
    """
    a = _as_2d(matrix)
    n_cols = a.shape[1]
    X = np.where(np.isfinite(a), a, np.nan)
    mean = np.nanmean(X, axis=0)
    Xc = X - mean
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.nanstd(X, axis=0, ddof=1)
        std[std == 0] = np.nan
        Z = Xc / std
        R = np.corrcoef(np.nan_to_num(Z), rowvar=False)
    R = np.atleast_2d(R)
    dropped, pairs = [], []
    keep = list(range(n_cols))
    for i in range(n_cols):
        if i not in keep:
            continue
        for j in range(i + 1, n_cols):
            if j not in keep:
                continue
            r = R[i, j] if np.isfinite(R[i, j]) else 0.0
            if abs(r) > threshold:
                dropped.append(j)
                keep.remove(j)
                pairs.append({"keep": i, "drop": j, "r": float(r)})
    return {"keep": keep, "dropped": sorted(dropped),
            "pairs": pairs, "n_original": n_cols, "n_kept": len(keep)}


def vif_report(matrix, names: Sequence[str] | None = None) -> Dict[str, Any]:
    """方差膨胀因子。VIF > 5 提示共线性，> 10 通常认为严重。

    计算：对每个列做对其余列的线性拟合，VIF = 1/(1-R²)。
    """
    a = _as_2d(matrix)
    n, p = a.shape
    if n <= p:
        raise ValueError(f"样本数 {n} 必须大于变量数 {p} 才能算 VIF")
    X = np.where(np.isfinite(a), a, np.nan)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, col_mean)
    out = []
    for j in range(p):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(n), others])
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = float("inf") if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
        except np.linalg.LinAlgError:
            r2, vif = float("nan"), float("nan")
        out.append({
            "name": (names[j] if names and j < len(names) else f"col{j}"),
            "index": j, "r2": r2, "VIF": vif,
            "flag": "high" if (np.isfinite(vif) and vif > 10)
                    else ("elevated" if (np.isfinite(vif) and vif > 5) else "ok"),
        })
    flagged = [o["name"] for o in out if o["flag"] != "ok"]
    return {"vif": out, "flagged": flagged, "max_vif": max(
        (o["VIF"] for o in out if np.isfinite(o["VIF"])), default=float("nan"))}


def standardize(matrix, ddof: int = 1):
    """Z-score 标准化（按列）。返回 (Z, mean, sd)。"""
    a = _as_2d(matrix)
    mean = np.nanmean(a, axis=0)
    sd = np.nanstd(a, axis=0, ddof=ddof)
    sd = np.where(sd == 0, 1.0, sd)
    return (a - mean) / sd, mean, sd


def one_hot(labels: Sequence[Any], drop_first: bool = False):
    """独热编码。返回 (矩阵, 类别顺序)；类别顺序必须输出否则列含义不明。"""
    vals = list(labels)
    levels = sorted({str(v) for v in vals})
    if drop_first and len(levels) > 1:
        levels = levels[1:]
    idx = {lv: i for i, lv in enumerate(levels)}
    M = np.zeros((len(vals), len(levels)), dtype=float)
    for r, v in enumerate(vals):
        s = str(v)
        if s in idx:
            M[r, idx[s]] = 1.0
    return M, levels


def prep_pipeline(matrix, *, variance_threshold: float = 0.0,
                  corr_threshold: float = 0.95,
                  outlier_method: str = "mad", outlier_k: float = 3.5,
                  names: Sequence[str] | None = None) -> Dict[str, Any]:
    """一次跑完「缺失 → 低方差 → 离群 → 共线性」并给出**可执行的下一步建议**。

    注意：本函数**不修改数据**，只产出报告与应保留/剔除的索引。
    自动改数据是临床分析里最难追责的一类操作。
    """
    a = _as_2d(matrix)
    report: Dict[str, Any] = {"n_rows": int(a.shape[0]), "n_cols": int(a.shape[1])}
    report["missing"] = missing_report(a)
    _, keep_var = drop_low_variance(a, variance_threshold)
    report["low_variance_dropped"] = [int(i) for i in range(a.shape[1])
                                      if i not in keep_var]
    sub = a[:, keep_var] if keep_var else a[:, :0]
    report["outliers"] = [
        outlier_report(sub[:, j], method=outlier_method, k=outlier_k)
        for j in range(sub.shape[1])
    ] if sub.shape[1] else []
    report["correlation"] = (correlation_prune(sub, corr_threshold)
                             if sub.shape[1] > 1 else {"keep": list(range(sub.shape[1])),
                                                       "dropped": [], "pairs": []})
    # VIF 必须在**相关剪枝之前**算：剪枝后再算必然接近 1，
    # 于是「原始数据共线性严重」这个事实会被自己抹掉（实测 r=0.998 的两列
    # 剪枝后 max_vif=1.019，看起来完全没问题）。VIF 的作用是暴露问题，
    # 剪枝只是应对手段，不能先掩盖再报告。
    if sub.shape[1] > 1 and sub.shape[0] > sub.shape[1]:
        nm = [names[i] for i in keep_var] if names else None
        report["vif"] = vif_report(sub, nm)
    else:
        report["vif"] = None

    advice = []
    if report["missing"]["cols_with_any_missing"]:
        advice.append(f"列 {report['missing']['cols_with_any_missing']} 存在缺失，"
                      f"完整行占比 {report['missing']['complete_row_pct']:.1%}；"
                      f"先决定填补策略（lib/impute.py）再建模")
    if report["low_variance_dropped"]:
        advice.append(f"低方差列 {report['low_variance_dropped']} 建议剔除")
    n_out = sum(o["n_outliers"] for o in report["outliers"])
    if n_out:
        advice.append(f"共检出 {n_out} 个离群值（{outlier_method}, k={outlier_k}）；"
                      f"需人工确认是录入错误还是真实极端表型，**不要自动删除**")
    if report["vif"] and report["vif"]["flagged"]:
        advice.append(f"共线性：{report['vif']['flagged']} VIF 偏高，"
                      f"同时进入回归会使系数不稳定")
    report["advice"] = advice
    return report
