# PORTED-FILE · 来源 mortazavilab/PyWGCNA（MIT）· 文件 PyWGCNA/wgcna.py
# 源 commit: 由 tools/vendor_sync.py --check-ported 校验（见 vendor/ATTRIBUTION.md）
# 移植方式：**算法照抄、依赖换底**。原实现依赖 sklearn(KNNImputer/preprocessing.scale)、
# statsmodels(ols)、matplotlib/seaborn/pyvis/reactome2py/gseapy（绘图与联网），
# 本文件只保留**纯算法核心**，把这些依赖逐个替换为 numpy / scipy 等价实现：
#   · statsmodels.ols(formula=...)  -> np.linalg.lstsq（OLS 闭式解，数值等价）
#   · sklearn.preprocessing.scale  -> np 列标准化（同口径：按 axis=0 去均值除标准差）
#   · sklearn.impute.KNNImputer    -> 列均值填补（默认关闭；缺失多时要求显式开启）
# 涉及口径的函数均标注上游函数名，便于逐行对照。
# **不要直接编辑本文件**；它是手工移植件，改动需同步更新 ATTRIBUTION.md 的来源行号。
"""WGCNA 核心算法（加权基因共表达网络分析）。

算法出处：Zhang & Horvath (2005) Stat Appl Genet Mol Biol；实现口径对照
PyWGCNA（MIT）的 pickSoftThreshold / adjacency / TOMsimilarity /
moduleEigengenes / mergeCloseModules。

与上游的**已知差异（如实标注）**：
1. 模块识别用 `scipy.cluster.hierarchy` 层次聚类 + 基于模块特征基因(ME)距离的
   动态剪枝近似，**不是** R 包 dynamicTreeCut 的完整 hybrid 剪枝
   （上游 cutreeHybrid 有 `branch_nSingletons[large] = ns` 的索引缺陷，未逐字抄）。
   → 模块边界可能与 R 版略有差异；软阈值/邻接/TOM/ME/kME 与上游一致。
2. 缺失值默认不插补（上游默认 KNNImputer）；需要时显式 `impute=True` 走列均值。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

NETWORK_TYPES = ("unsigned", "signed", "signed hybrid")
TOM_TYPES = ("unsigned", "signed")
TOM_DENOMS = ("min", "mean")

# WGCNA 标准调色板（上游 labels2colors 的 colorSeq 前若干色，保持同名以便对照）
_STD_COLORS = [
    "turquoise", "blue", "brown", "yellow", "green", "red", "black", "pink",
    "magenta", "purple", "greenyellow", "tan", "salmon", "cyan", "grey60",
    "lightcyan", "lightgreen", "lightyellow", "royalblue", "darkred",
    "darkgreen", "darkturquoise", "darkgrey", "orange", "darkorange",
    "white", "grey",
]
_GREY = "grey"


def _as_expr(x) -> pd.DataFrame:
    """统一成 DataFrame：行=样本，列=基因。"""
    if isinstance(x, pd.DataFrame):
        return x
    arr = np.asarray(x, dtype=float)
    return pd.DataFrame(arr)


def _ols_r2(x: np.ndarray, y: np.ndarray) -> tuple:
    """一元 OLS：返回 (R^2, slope)。等价于 statsmodels ols 的 rsquared 与 params[1]。

    口径：带截距，R^2 = 1 - SS_res/SS_tot（SS_tot 相对均值）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(x.size), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(r2), float(beta[1])


def _ols_r2_adj_2var(x: np.ndarray, y: np.ndarray) -> float:
    """二元 OLS 的 adjusted R^2：y ~ x + 10^x（上游 truncatedExponentialAdjRsquared）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(10.0 ** x)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return float("nan")
    X = np.column_stack([np.ones(x.size), x, 10.0 ** x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = x.size, X.shape[1]
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if not np.isfinite(r2) or n - p - 1 <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


# ---------------------------------------------------------------------------
# 上游：scaleFreeFitIndex
# ---------------------------------------------------------------------------
def scale_free_fit_index(k: Sequence[float], n_breaks: int = 10) -> Dict[str, float]:
    """标度自由拓扑拟合指数。

    口径（同 PyWGCNA.scaleFreeFitIndex）：把连接度 k 分 n_breaks 箱，
    对「箱内平均连接度 dk」与「箱内占比 p_dk」做 log10-log10 线性回归：
      · SFT.R.sq  = 该回归的 R^2（越接近 1 越标度自由）
      · slope     = 回归斜率（通常期望为负）
      · truncated.R.sq = 加入 10^log_dk 项的 adjusted R^2（截尾指数模型）
    """
    k = np.asarray(k, dtype=float)
    k = k[np.isfinite(k)]
    if k.size < 3:
        raise ValueError("连接度样本不足（<3），无法拟合标度自由拓扑")
    # 分箱：固定 n_breaks 个箱，**空箱也要占一个位置**（否则后面 dk2 填补会
    # 出现 boolean index 长度不匹配——上游用 groupby(categorical) 天然保留空箱，
    # 这里用 np.digitize 显式保留，语义与上游一致）。
    n_breaks = int(n_breaks)
    breaks = np.linspace(k.min(), k.max(), n_breaks + 1)
    which = np.clip(np.digitize(k, breaks[1:-1], right=False), 0, n_breaks - 1)
    counts = np.bincount(which, minlength=n_breaks).astype(float)
    sums = np.bincount(which, weights=k, minlength=n_breaks)
    with np.errstate(divide="ignore", invalid="ignore"):
        dk = np.where(counts > 0, sums / np.where(counts == 0, 1, counts), np.nan)
    p_dk = counts / len(k)
    _, edges = np.histogram(k, bins=breaks)
    dk2 = 0.5 * (edges[1:] + edges[:-1])  # 箱中位（用于填补空箱/零值）
    dk = np.where(np.isnan(dk), dk2, dk)
    dk = np.where(dk == 0, dk2, dk)
    p_dk = np.where(np.isnan(p_dk), 0.0, p_dk)
    log_dk = np.log10(dk)
    log_p = np.log10(p_dk + 1e-9)
    r2, slope = _ols_r2(log_dk, log_p)
    return {
        "SFT.R.sq": r2,
        "slope": slope,
        "truncated.R.sq": _ols_r2_adj_2var(log_dk, log_p),
        "n_breaks": int(n_breaks),
    }


# ---------------------------------------------------------------------------
# 上游：adjacency
# ---------------------------------------------------------------------------
def adjacency(dat_expr, power: float = 6.0, network_type: str = "unsigned") -> np.ndarray:
    """软阈值邻接矩阵：A = |cor|^power（unsigned）/ ((1+cor)/2)^power（signed）。

    :param dat_expr: 行=样本、列=基因的表达矩阵
    :param power: 软阈值幂；unsigned/signed hybrid 下 1 表示硬阈值
    :param network_type: unsigned（默认）| signed | signed hybrid
    """
    if network_type not in NETWORK_TYPES:
        raise ValueError(f"network_type 必须是 {NETWORK_TYPES} 之一，收到 {network_type!r}")
    expr = _as_expr(dat_expr)
    if expr.shape[1] < 3:
        raise ValueError("基因数 <3，相关网络退化")
    mat = expr.to_numpy(dtype=float)
    cor = np.corrcoef(mat, rowvar=False)
    cor = np.nan_to_num(cor, nan=0.0)
    if network_type == "unsigned":
        cor = np.abs(cor)
    elif network_type == "signed":
        cor = (1.0 + cor) / 2.0
    else:  # signed hybrid：负相关置 0
        cor = np.where(cor < 0, 0.0, cor)
    np.fill_diagonal(cor, 1.0 if network_type != "unsigned" else np.abs(np.diag(cor)))
    return np.power(cor, power)


def tom_similarity(adj: np.ndarray, tom_type: str = "signed",
                   tom_denom: str = "min") -> np.ndarray:
    """拓扑重叠矩阵 TOM（上游 TomSimilarityFromAdj / TOMsimilarity）。

    TOM_ij = (L_ij + a_ij) / (min(k_i, k_j) + 1 - a_ij)，其中 L = A@A、k_i = sum_j a_ij。
    tom_denom="mean" 时把 min 换成 (k_i+k_j)/2（上游注明为实验性）。
    """
    if tom_type not in TOM_TYPES:
        raise ValueError(f"tom_type 必须是 {TOM_TYPES} 之一，收到 {tom_type!r}")
    if tom_denom not in TOM_DENOMS:
        raise ValueError(f"tom_denom 必须是 {TOM_DENOMS} 之一，收到 {tom_denom!r}")
    a = np.array(adj, dtype=float, copy=True)
    np.fill_diagonal(a, 0.0)
    L = a @ a
    ki = a.sum(axis=1)
    kj = a.sum(axis=0)
    if tom_denom == "min":
        mink = np.minimum(ki[:, None], kj[None, :])
    else:
        mink = (ki[:, None] + kj[None, :]) / 2.0
    if tom_type == "unsigned":
        tom = (L + a) / (mink + 1.0 - a)
    else:
        tom = np.abs(L + a) / (mink + 1.0 - np.abs(a))
    np.fill_diagonal(tom, 1.0)
    return tom


# ---------------------------------------------------------------------------
# 上游：pickSoftThreshold
# ---------------------------------------------------------------------------
def pick_soft_threshold(dat_expr, powers: Optional[Sequence[float]] = None,
                        network_type: str = "unsigned", rsq_cut: float = 0.9,
                        mean_cut: float = 100.0, n_breaks: int = 10) -> Dict[str, Any]:
    """软阈值（power）筛选：对每个候选幂算标度自由拟合指数与平均连接度。

    推荐规则（与上游一致）：取**达到 SFT.R.sq ≥ rsq_cut 的最小 power**；
    若所有幂都不达标，返回 R^2 最大的那个并标 `reached_cut=False`。
    """
    expr = _as_expr(dat_expr)
    if powers is None:
        powers = list(range(1, 11)) + list(range(12, 21, 2))
    powers = sorted(set(float(p) for p in powers))
    rows = []
    for p in powers:
        adj = adjacency(expr, power=p, network_type=network_type)
        k = adj.sum(axis=1) - 1.0  # 去掉自连接
        fit = scale_free_fit_index(k, n_breaks=n_breaks)
        rows.append({
            "power": p,
            "SFT.R.sq": fit["SFT.R.sq"],
            "slope": fit["slope"],
            "truncated.R.sq": fit["truncated.R.sq"],
            "mean.k": float(np.mean(k)),
            "median.k": float(np.median(k)),
            "max.k": float(np.max(k)),
        })
    table = pd.DataFrame(rows)
    ok = table[(table["SFT.R.sq"] >= rsq_cut) & (table["mean.k"] <= mean_cut)]
    if len(ok):
        best = ok.iloc[0]
        reached = True
    else:
        ok2 = table[table["SFT.R.sq"] >= rsq_cut]
        best = ok2.iloc[0] if len(ok2) else table.loc[table["SFT.R.sq"].idxmax()]
        reached = len(ok2) > 0
    return {
        "table": table,
        "power_estimate": float(best["power"]),
        "reached_rsq_cut": bool(reached),
        "rsq_cut": float(rsq_cut),
        "mean_cut": float(mean_cut),
        "network_type": network_type,
        "rule": "最小 power 使 SFT.R.sq >= rsq_cut 且 mean.k <= mean_cut；不达标取 R^2 最大者",
    }


# ---------------------------------------------------------------------------
# 上游：moduleEigengenes（1st PC）+ align="along average"
# ---------------------------------------------------------------------------
def _standardize_cols(x: np.ndarray) -> np.ndarray:
    """按列标准化（等价 sklearn.preprocessing.scale，标准差为 0 时置 0 不除）。"""
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def module_eigengenes(expr, colors: Sequence[str], impute: bool = False,
                      n_pc: int = 1, exclude_grey: bool = False,
                      grey: str = _GREY) -> Dict[str, Any]:
    """模块特征基因（ME）：模块内表达矩阵按**基因**标准化后的第一主成分。

    口径（同上游 moduleEigengenes）：
      · 每个模块取 datModule = expr[:, 模块基因].T（行=基因、列=样本）
      · scaleVar=True：按基因（行）跨样本标准化——上游 `scale(datModule.T).T`
      · SVD 后取右奇异向量 v[0]（长度=样本数）作为 ME
      · align="along average"：若 ME 与模块平均表达负相关则整体系取反
      · varExplained：各 PC 与模块内基因表达相关² 的均值
    """
    e = _as_expr(expr)
    colors = np.asarray(colors, dtype=object)
    if e.shape[1] != len(colors):
        raise ValueError("expr 列数必须等于 colors 长度")
    levels = list(pd.Categorical(colors).categories)
    if exclude_grey and grey in levels:
        levels = [c for c in levels if c != grey]
    if not levels:
        raise ValueError("没有非空模块")
    mes = {}
    avg = {}
    var_exp = {}
    valid = {}
    for level in levels:
        sub = e.loc[:, colors == level]
        if sub.shape[1] == 0:
            continue
        mat = sub.to_numpy(dtype=float)
        if impute:
            col_mean = np.nanmean(np.where(np.isfinite(mat), mat, np.nan), axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            idx = np.where(~np.isfinite(mat))
            mat[idx] = np.take(col_mean, idx[1])
        mat = np.nan_to_num(mat, nan=0.0)
        # mat: 样本×基因；按**基因**（列）标准化后转置 -> 基因×样本，
        # 对应上游 `scale(datModule.T).T`（datModule = 基因×样本，scale 按列=基因）。
        scaled = _standardize_cols(mat).T
        if scaled.shape[0] < 2:
            # 单基因模块：该基因标准化表达即为 ME
            pc1 = scaled[0]
            ves = [1.0]
        else:
            u, s, vt = np.linalg.svd(scaled, full_matrices=False)
            ncomp = min(int(n_pc), len(s))
            pc1 = vt[0]
            ves = []
            for j in range(max(1, ncomp)):
                r = np.array([np.corrcoef(vt[j], scaled[g])[0, 1]
                              for g in range(scaled.shape[0])])
                ves.append(float(np.nanmean(np.nan_to_num(r) ** 2)))
        avg_expr = mat.mean(axis=1)  # 每个样本的模块平均表达（用于 along-average 对齐）
        if np.isfinite(avg_expr).all() and np.std(avg_expr) > 0 and np.std(pc1) > 0:
            if float(np.corrcoef(pc1, avg_expr)[0, 1]) < 0:
                pc1 = -pc1
        mes[f"ME{level}"] = np.asarray(pc1, dtype=float)
        avg[f"AE{level}"] = np.asarray(avg_expr, dtype=float)
        var_exp[f"ME{level}"] = ves[0] if ves else float("nan")
        valid[f"ME{level}"] = True
    return {
        "eigengenes": pd.DataFrame(mes, index=e.index if e.index is not None else None),
        "average_expr": pd.DataFrame(avg, index=e.index if e.index is not None else None),
        "var_explained": var_exp,
        "valid_mes": valid,
        "align": "along average",
    }


def labels2colors(labels: Sequence[int], zero_is_grey: bool = True) -> List[str]:
    """把数值簇标签映射为 WGCNA 标准颜色（上游 labels2colors）。0 -> grey。"""
    out = []
    for lab in labels:
        lab = int(lab)
        if lab == 0 and zero_is_grey:
            out.append(_GREY)
        else:
            out.append(_STD_COLORS[(lab - 1) % (len(_STD_COLORS) - 1)])
    return out


# ---------------------------------------------------------------------------
# 模块识别 + 合并（上游 cutreeHybrid / mergeCloseModules 的口径近似）
# ---------------------------------------------------------------------------
def detect_modules(expr, power: Optional[float] = None, network_type: str = "unsigned",
                   tom_type: str = "signed", tom_denom: str = "min",
                   min_module_size: int = 20, merge_cut_height: float = 0.25,
                   deep_split: int = 0, n_breaks: int = 10,
                   diss_threshold: Optional[float] = None) -> Dict[str, Any]:
    """一步跑完 WGCNA：软阈值 → 邻接 → TOM → 层次聚类 → 模块合并 → ME。

    :param power: 为 None 时自动挑选（pick_soft_threshold）
    :param merge_cut_height: 模块合并阈值——ME 距离（1-|cor|）低于此值即合并
    :param deep_split: 0–4，取第 (deep_split+1) 大的高度间隙作为切点，越大切得越碎
    :param diss_threshold: 显式指定 TOM 相异度切点；给定后忽略 deep_split
    :return: colors / MEs / 软阈值表 / TOM / 诊断信息
    """
    e = _as_expr(expr)
    if power is None:
        sft = pick_soft_threshold(e, network_type=network_type, n_breaks=n_breaks)
        power = float(sft["power_estimate"])
    else:
        sft = None
    adj = adjacency(e, power=power, network_type=network_type)
    tom = tom_similarity(adj, tom_type=tom_type, tom_denom=tom_denom)
    diss = 1.0 - tom
    np.fill_diagonal(diss, 0.0)
    diss = (diss + diss.T) / 2.0
    dist = squareform(diss, checks=False)
    Z = linkage(dist, method="average")
    heights = np.sort(Z[:, 2])
    if diss_threshold is not None:
        cut = float(diss_threshold)
    elif heights.size:
        # 最大跳跃法：在排序后的合并高度上找**间隙**最大的断点，取该间隙中点。
        # 不用分位数——TOM 相异高度常呈双峰（簇内很近、簇间接近 1），
        # 分位数会整体落进同一峰里，导致「切了个寂寞」（实测 q=0.80/0.90 都切在同一点）。
        # deep_split=k 表示取第 (k+1) 大的间隙：越大切得越碎。
        ds = int(max(0, min(4, deep_split)))
        gaps = np.diff(heights)
        order = np.argsort(gaps)[::-1]
        j = int(order[min(ds, len(order) - 1)])
        cut = float((heights[j] + heights[j + 1]) / 2.0)
    else:
        cut = 0.0
    labels = fcluster(Z, t=max(cut, 1e-12), criterion="distance")
    # 小簇归入 grey（0）
    sizes = pd.Series(labels).value_counts()
    small = set(sizes[sizes < min_module_size].index)
    labels = np.array([0 if l in small else l for l in labels])
    colors = np.array(labels2colors(labels), dtype=object)

    me_res = None
    merged = []
    if len(set(colors) - {_GREY}) > 0:
        me_res = module_eigengenes(e, colors, exclude_grey=True)
        colors, merge_info = merge_close_modules(me_res["eigengenes"], colors,
                                                 cut_height=merge_cut_height)
        merged = merge_info["merged"]
        # 合并后模块成员变了，ME 必须**重算**（拿旧 ME 当结果是典型的口径错误）
        me_res = module_eigengenes(e, colors, exclude_grey=True)
    return {
        "power": float(power),
        "soft_threshold": sft,
        "colors": list(colors),
        "n_modules": int(len(set(colors) - {_GREY})),
        "module_sizes": {c: int((colors == c).sum()) for c in sorted(set(colors))},
        "eigengenes": me_res,
        "merged_modules": merged,
        "tom": tom,
        "adjacency": adj,
        "network_type": network_type,
        "tom_type": tom_type,
        "tom_denom": tom_denom,
        "min_module_size": int(min_module_size),
        "merge_cut_height": float(merge_cut_height),
        "cluster_rule": "scipy 层次聚类(average) + ME 距离合并（非 R dynamicTreeCut 完整 hybrid）",
    }


def merge_close_modules(mes: pd.DataFrame, colors: Sequence[Any],
                        cut_height: float = 0.25) -> tuple:
    """合并 ME 距离过近的模块（上游 mergeCloseModules）。

    距离定义：1 - |cor(ME_i, ME_j)|；低于 cut_height 的模块对合并（保留较大模块名）。
    """
    colors = np.array(colors, dtype=object)
    me_df = mes if isinstance(mes, pd.DataFrame) else pd.DataFrame(mes)
    if me_df.shape[1] < 2:
        return colors, {"merged": [], "eigengenes": me_df}
    cor = np.abs(np.corrcoef(me_df.to_numpy(dtype=float), rowvar=False))
    names = list(me_df.columns)
    dist = 1.0 - cor
    n = len(names)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    merged = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < cut_height:
                ri, rj = find(i), find(j)
                if ri != rj:
                    # 保留成员数多的模块名
                    si = int((colors == names[ri][2:]).sum())
                    sj = int((colors == names[rj][2:]).sum())
                    keep, drop = (ri, rj) if si >= sj else (rj, ri)
                    parent[drop] = keep
                    merged.append({"from": names[drop], "to": names[keep],
                                   "dist": float(dist[i, j])})
    # 应用合并
    mapping = {}
    for idx, nm in enumerate(names):
        root = find(idx)
        mapping[nm] = names[root]
    new_colors = []
    for c in colors:
        key = f"ME{c}"
        new_colors.append(mapping[key][2:] if key in mapping else c)
    return np.array(new_colors, dtype=object), {"merged": merged, "eigengenes": me_df}


# ---------------------------------------------------------------------------
# 模块成员度 kME / hub gene（WGCNA 常规派生量）
# ---------------------------------------------------------------------------
def module_membership(expr, colors: Sequence[Any], mes: Optional[pd.DataFrame] = None,
                      exclude_grey: bool = True) -> pd.DataFrame:
    """kME：基因表达与模块特征基因的相关（WGCNA 的 module membership）。

    |kME| 越大说明该基因越是该模块的核心成员；据此挑 hub gene。
    """
    e = _as_expr(expr)
    colors = np.array(colors, dtype=object)
    me_df = mes if mes is not None else module_eigengenes(e, colors, exclude_grey=exclude_grey)["eigengenes"]
    if me_df is None or me_df.shape[1] == 0:
        return pd.DataFrame()
    rows = []
    mat = e.to_numpy(dtype=float)
    for col in me_df.columns:
        me = me_df[col].to_numpy(dtype=float)
        r = np.array([np.corrcoef(mat[:, g], me)[0, 1] if np.std(mat[:, g]) > 0 else 0.0
                      for g in range(mat.shape[1])])
        p = np.full(r.shape, np.nan)
        rows.append(pd.DataFrame({"gene": list(e.columns), "module": col[2:],
                                  "kME": r, "p": p}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


def hub_genes(kme: pd.DataFrame, top_n: int = 10) -> Dict[str, List[str]]:
    """按 |kME| 取每个模块的前 top_n 个 hub gene。"""
    if kme is None or len(kme) == 0:
        return {}
    out = {}
    d = kme.copy()
    d["abs_kME"] = d["kME"].abs()
    for mod, grp in d.groupby("module"):
        out[str(mod)] = list(grp.sort_values("abs_kME", ascending=False)["gene"].head(top_n))
    return out


def module_trait_correlation(mes: pd.DataFrame, traits,
                             method: str = "pearson") -> Dict[str, Any]:
    """模块-性状相关：ME 与每个性状的相关系数 + p 值（WGCNA 标配下游）。"""
    from scipy import stats as _st

    me_df = mes if isinstance(mes, pd.DataFrame) else pd.DataFrame(mes)
    tr = _as_expr(traits) if not isinstance(traits, pd.DataFrame) else traits
    cor = np.zeros((me_df.shape[1], tr.shape[1]))
    pv = np.zeros((me_df.shape[1], tr.shape[1]))
    for i, mc in enumerate(me_df.columns):
        x = me_df[mc].to_numpy(dtype=float)
        for j, tc in enumerate(tr.columns):
            y = tr[tc].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3:
                cor[i, j], pv[i, j] = np.nan, np.nan
                continue
            if method == "spearman":
                r, p = _st.spearmanr(x[ok], y[ok])
            else:
                r, p = _st.pearsonr(x[ok], y[ok])
            cor[i, j], pv[i, j] = float(r), float(p)
    return {
        "correlation": pd.DataFrame(cor, index=list(me_df.columns), columns=list(tr.columns)),
        "pvalue": pd.DataFrame(pv, index=list(me_df.columns), columns=list(tr.columns)),
        "method": method,
    }
