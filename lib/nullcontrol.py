# -*- coding: utf-8 -*-
"""随机匹配基因集零分布（阴性对照）—— **全项目唯一裁决点**。

为什么必须有它：
  一个"我关心的基因在目标细胞类型里显著富集"的结论，需要回答一个问题 ——
  **换成随便一组同样的基因，会不会一样显著？** 不问这个问题，20 个基因里挑
  10 个说我命中了，和扔骰子没有区别。这套零分布就是把这个"随便一组"真正算出来。

匹配比随机更重要的那一步：
  **基因之间不可比**。高表达基因（晶状体里的 CRYAA/MIP）天然有更高的检出率、
  更大的 fold change、更高的特异性评分；低表达基因的方差被散粒噪声主导。
  直接在整个 gene universe 里均匀随机抽样，会让零分布整体偏弱 →
  **真信号也被稀释成假阳性**，看上去什么都显著。所以默认按协变量（表达水平/
  检出率）**分箱等箱抽样**：把待检基因所在的表达层位一比一复刻到对照组里。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from lib import lm

__all__ = ["sample_matched", "gene_set_null", "multi_set_null"]


def _as_map(values: Mapping[str, float] | Sequence[float],
            universe: Sequence[str]) -> pd.Series:
    """把协变量整理成以 universe 为索引的 Series（缺值留 NaN，等会儿剔除）。"""
    if isinstance(values, Mapping):
        return pd.Series({g: float(values.get(g, np.nan)) for g in universe}, dtype=float)
    v = np.asarray(list(values), dtype=float)
    if v.shape[0] != len(universe):
        raise ValueError(f"协变量长度 {v.shape[0]} 与 universe 长度 {len(universe)} 不一致")
    return pd.Series(v, index=list(universe), dtype=float)


def sample_matched(user_set: Sequence[str], universe: Sequence[str], *,
                  match_var: Mapping[str, float] | Sequence[float],
                  n_iter: int = 1000, n_bins: int = 20,
                  seed: int = 0, exclude: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """抽 `n_iter` 组与待检基因集**同等大小且协变量层位匹配**的随机基因。

    :param user_set: 待检基因集
    :param universe: 背景基因全集（**必须**是真正被测过的基因，不是全基因组）
    :param match_var: **匹配用的协变量**（平均表达 / 检出率 / 基因长度……）
    :param n_bins: 分箱数（按分位数，保证每箱背景基因数量接近）
    :param exclude: 需要从候选中排除的基因（如已用于定义该集合的基因）
    :returns: ``{"samples": [[gene...], ...], "background_used": [...], ...}``

    四条写死的口径：
      1. **按 quantiles 分箱**而不是等宽分箱 —— 表达分布是长尾的，等宽分箱会让
         绝大多数基因落进第 1 箱，"匹配"退化成"全部来自低表达层"；
      2. 箱内**无放回**抽样，箱内候选不足时**报错**而不是有放回凑数：
         有放回会抽出重复基因，这些重复基因的统计量方差被低估，p 值偏小；
      3. 背景与待检基因用**同一套分箱边界**（先算 quantile edges，两边都用
         ``pd.cut``），否则箱号对不上，"配额"会落在错误的层位上而不报错；
      4. 抽样池默认**排除 user_set 自身**，否则对照组会掺进真基因。
    """
    user = [str(g) for g in user_set]
    bg = [str(g) for g in universe]
    if not user:
        raise ValueError("user_set 为空")
    if len(set(user)) != len(user):
        raise ValueError("user_set 内有重复基因 —— 先去重，否则集合大小的语义变了")
    missing = sorted(set(user) - set(bg))
    if missing:
        raise ValueError(f"待检基因不在 universe 里：{missing[:8]}"
                         f"（共 {len(missing)} 个）。背景必须是真正被测过的基因集合")

    sc = _as_map(match_var, bg)
    valid = sc.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.size < len(user) + 2:
        raise ValueError(f"可用背景基因 {valid.size} 个，不足以做匹配抽样")

    ex = set(exclude or []) | set(user)
    pool = [g for g in bg if g in valid.index and g not in ex]
    if len(pool) < len(user):
        raise ValueError(f"排除后仅剩 {len(pool)} 个候选，少于待检基因的 {len(user)} 个")

    sc_pool = valid.loc[pool]
    sc_user = valid.loc[user]          # 待检基因自身的层位 —— 决定每层的配额
    if sc_pool.nunique() < 2:
        raise ValueError("协变量在池中几乎无变化（nunique < 2），无法分箱匹配")

    # ── 背景与待检基因共用同一套分箱边界（口径 3）
    nb = int(min(n_bins, max(2, sc_pool.size // 2)))
    edges = np.unique(sc_pool.quantile(np.linspace(0, 1, nb + 1)).to_numpy())
    if edges.size < 3:
        raise ValueError("协变量值过于集中（分位数退化），无法做层位匹配；"
                         "请换一个分散的匹配变量")
    edges[0], edges[-1] = -np.inf, np.inf

    bg_bins = pd.cut(sc_pool, bins=edges, labels=False, include_lowest=True)
    user_bins = pd.cut(sc_user, bins=edges, labels=False, include_lowest=True)
    if user_bins.isna().any() or bg_bins.isna().any():
        raise ValueError("分箱出现 NaN —— 匹配变量含无法装箱的值（如全同值）")

    quota = {int(b): int(c) for b, c in user_bins.value_counts().items()}
    by_bin: Dict[int, list] = {}
    for g, b in bg_bins.items():
        by_bin.setdefault(int(b), []).append(g)

    bin_keys = sorted(by_bin.keys())

    def _expand(b: int, need: int):
        """第 b 箱的候选不够时，向**相邻层位**依次外扩凑够配额。

        为什么不直接有放回凑数：有放回会抽到重复基因，重复基因的汇总统计量
        方差被系统性低估 → 零分布偏窄 → z 偏大 → p 偏小（假阳性）。
        为什么不是直接报错：某个高表达层位天然只有几十个基因，而待检基因集
        恰好全落在那一层（这在"只挑最显著基因"的用法里很常见），
        直接报错会让这套阴性对照在**最需要它的场景**下不可用。
        折中：优先本箱，不够就按 |Δ箱| 从小到大合并相邻箱的候选，
        并把实际用了几层写进返回值的 ``bins_expanded`` —— 扩得越多，
        匹配就越粗，使用者能看见。
        """
        order = [b] + sorted([x for x in bin_keys if x != b],
                             key=lambda x: (abs(x - b), x))
        cand: list = []
        used = 1
        for x in order:
            cand.extend(by_bin.get(x, []))
            if len(cand) >= need:
                return cand, used
            used += 1
        return cand, used

    avail: Dict[int, list] = {}
    expanded: Dict[int, int] = {}
    for b, q in sorted(quota.items()):
        if q <= 0 or b not in by_bin:
            continue
        cand, used = _expand(b, q)
        if len(cand) < q:
            raise ValueError(
                f"第 {b} 层位即使合并全部相邻层位也只有 {len(cand)} 个候选，"
                f"少于配额 {q} —— 说明背景基因库太小或待检集合太大；"
                "请减少集合规模或降低 n_bins")
        avail[b] = cand
        expanded[b] = used

    if int(n_iter) < 100:
        raise ValueError(f"n_iter={n_iter} 太小 —— 经验 p 的分辨率是 1/n_iter，"
                         "低于 100 时 p 值粒度粗到没法做 FDR 校正")

    rng = np.random.default_rng(int(seed))
    samples = []
    for _ in range(int(n_iter)):
        pick: list = []
        taken: set = set()
        for b, q in sorted(quota.items()):
            if q <= 0 or b not in avail:
                continue
            cand = [g for g in avail[b] if g not in taken]
            if len(cand) < q:
                raise ValueError(
                    f"第 {b} 层位在单次抽样中耗尽候选（需 {q} 余 {len(cand)}）——"
                    "多层配额重叠过多；请降低 n_bins 让每层变宽")
            sel = rng.choice(np.asarray(cand, dtype=object), size=q, replace=False)
            sel = [str(g) for g in sel]
            taken.update(sel)
            pick.extend(sel)
        rng.shuffle(pick)
        samples.append(pick)

    return {
        "samples": samples,
        "n_iter": int(n_iter),
        "background_used": pool,
        "n_background": len(pool),
        "match_variable_bins": int(nb),
        "quota": sorted(quota.items()),
        "bins_expanded": expanded,
        "max_expansion": int(max(expanded.values())) if expanded else 0,
        "seed": int(seed),
        "exclude_applied": True,
        "engine": "lib.nullcontrol.sample_matched",
        "rule": ("在**同一箱的含量配额**下无放回抽样：尽量复刻待检基因集的协变量层位；"
                 "抽样池默认排除待检基因自身"),
    }


def gene_set_null(user_set: Sequence[str],
                  scores: Mapping[str, float] | Sequence[float],
                  *,                   universe: Optional[Sequence[str]] = None,
                  match_var: Optional[Mapping[str, float] | Sequence[float]] = None,
                  statistic: str = "mean", n_iter: int = 1000, n_bins: int = 20,
                  seed: int = 0, alternative: str = "greater") -> Dict[str, Any]:
    """对待检基因集的汇总统计量做置换检验，返回经验 p 值与 z 分数。

    :param scores: **被检验的统计量**（log2FC / 特异性 ES / -log10 p ……）
    :param universe: 背景基因全集，默认取 scores 的全部键（**应**为真正被测过的基因）
    :param match_var: **用于层位匹配的协变量**（通常是平均表达或检出率）。
        强烈建议给出；省略则退化为在整个背景里均匀随机，结果里会标记
        ``matching="unmatched"`` 并给出偏乐观的告警
    :param statistic: ``mean``（默认）或 ``frac_pos``（正值比例）
    :param alternative: ``greater``（默认，分数越高越富集）/ ``less`` / ``two_sided``
    :returns: 观察值、零分布均值标准差、经验 p、z、n_iter 与口径说明

    **``scores`` 与 ``match_var`` 绝不能是同一个量**（这是 API 的硬边界）。
    实测过两种混淆用法，危害的方向不同，但都让 p 值失去意义：

      * **完全相等**（直接把统计量传进 ``match_var``）：抽样被限制在"统计量同层"，
        零分布的**标准差被人为压小**（2026-10-12 实测：正常匹配 null_sd≈0.20、
        z=19；自匹配 null_sd 塌到 ≈0.03、z=32）。
        z 被放大，于是**任何微小差异都会显著** —— 这不是更保守，是更容易假阳性。
        代码里对此直接报错，不留余地。
      * **高度相关但不等**（例如用同一批数据算出的两个量）：不能自动判定为错误，
        但会在结果里给出 ``warning``；此时零分布同样偏窄，p 值偏乐观。

    正确做法：匹配变量必须是**先于本次检验就存在**的基因属性（表达水平、检出率、
    基因长度、GC 含量），**不是**本次检验产生的任何统计量。

    经验 p 的计算（含 +1/+1 校正）：
      ``p = (1 + #{null >= obs}) / (1 + n_iter)``
      加 1 是为了避免 p = 0 —— p = 0 无法做 BH 校正，也会让人误以为证据无穷大。
    """
    if statistic not in ("mean", "frac_pos"):
        raise ValueError(f"statistic 只能是 mean / frac_pos，收到 {statistic!r}")
    if alternative not in ("greater", "less", "two_sided"):
        raise ValueError(f"alternative 只能是 greater/less/two_sided，收到 {alternative!r}")

    if isinstance(scores, Mapping):
        if universe is None:
            universe = list(scores.keys())
        sc = pd.Series({str(g): float(scores[g]) for g in universe}, dtype=float)
    else:
        if universe is None:
            raise ValueError("scores 为数组时必须给出 universe（基因名列表）")
        sc = _as_map(scores, universe)

    bg_list = [str(g) for g in universe]
    valid = sc.replace([np.inf, -np.inf], np.nan).dropna()
    user = [str(g) for g in user_set]
    if not user:
        raise ValueError("user_set 为空 —— 没有基因就没有集合统计量可谈")
    if len(set(user)) != len(user):
        raise ValueError("user_set 内有重复基因")
    missing = sorted(set(user) - set(valid.index))
    if missing:
        raise ValueError(f"待检基因无可用分数（缺失/NaN）：{missing[:8]}")

    # ── 自匹配守卫：把「被检验的统计量」误当匹配变量，是最常见也最难自查的误用
    warning = None
    if match_var is not None:
        mv = _as_map(match_var, list(valid.index))
        both = valid.dropna().index.intersection(mv.dropna().index)
        if len(both) >= 3:
            a = valid.loc[both].to_numpy()
            b = mv.loc[both].to_numpy()
            if np.allclose(a, b, rtol=0, atol=0):
                raise ValueError(
                    "match_var 与 scores 完全相同 —— 这是用「被检验的统计量」去匹配自己。"
                    "零分布的标准差会被压缩到接近 0，z 被放大，任何微小差异都会显著。"
                    "请改用先于本次检验存在的基因属性（平均表达 / 检出率 / 长度）")
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
            if np.isfinite(corr) and abs(corr) > 0.95:
                warning = (f"match_var 与 scores 相关性极高（r={corr:.3f}）：零分布可能被"
                           "人为压窄，p 值偏乐观；请换一个独立于本次检验的匹配变量")


    obs = np.asarray([float(valid[g]) for g in user])
    obs_stat = float(obs.mean()) if statistic == "mean" else float((obs > 0).mean())

    if match_var is None:
        # 退化为均匀随机：不是错误，但要在结果里说清楚代价
        rng = np.random.default_rng(int(seed))
        if int(n_iter) < 100:
            raise ValueError(f"n_iter={n_iter} 太小：经验 p 分辨率是 1/n_iter")
        pool = [g for g in valid.index if g not in set(user)]
        if len(pool) < len(user):
            raise ValueError("背景基因不足以做均匀随机抽样")
        samples = [rng.choice(pool, size=len(user), replace=False).tolist()
                   for _ in range(int(n_iter))]
        matching, n_used = "unmatched", len(pool)
    else:
        samp = sample_matched(user, bg_list, match_var=match_var, n_iter=n_iter,
                              n_bins=n_bins, seed=seed)
        samples = samp["samples"]
        matching, n_used = "matched", int(samp["n_background"])

    null_stats = []
    for picks in samples:
        v = np.asarray([float(valid[g]) for g in picks])
        null_stats.append(float(v.mean()) if statistic == "mean" else float((v > 0).mean()))
    null = np.asarray(null_stats, dtype=float)

    if alternative == "greater":
        ge = int(np.sum(null >= obs_stat))
        p = (1.0 + ge) / (1.0 + len(null))
    elif alternative == "less":
        le = int(np.sum(null <= obs_stat))
        p = (1.0 + le) / (1.0 + len(null))
    else:
        d = np.abs(null - null.mean())
        p = (1.0 + int(np.sum(d >= abs(obs_stat - null.mean())))) / (1.0 + len(null))

    mu, sd = float(null.mean()), float(null.std(ddof=1)) if len(null) > 1 else float("nan")
    z = float((obs_stat - mu) / sd) if np.isfinite(sd) and sd > 0 else float("nan")

    return {
        "observed": obs_stat,
        "null_mean": mu,
        "null_sd": sd,
        "null_p95": float(np.quantile(null, 0.95)),
        "null_p05": float(np.quantile(null, 0.05)),
        "z": z,
        "pvalue": float(p),
        "n_iter": int(len(null)),
        "n_set": int(len(user)),
        "n_background": int(valid.size),
        "statistic": statistic,
        "alternative": alternative,
        "seed": int(seed),
        "null_distribution": null,
        "matching": matching,
        "warning": warning,
        "n_background_used": int(n_used),
        "pvalue_rule": "(1 + #{null 更极端}) / (1 + n_iter)：加 1 保证 p>0，可进 BH",
        "engine": "lib.nullcontrol.gene_set_null",
        "caveat": ("零分布建立在「匹配到了正确的协变量层位」之上；若匹配变量选错"
                   "（例如用表达量去匹配一个只在乎蛋白稳定性的集合），p 值照样会偏"
                   if matching == "matched" else
                   "**未做层位匹配**：背景里不同表达水平的基因被等权抽样，"
                   "高表达基因天然的效应量优势会混进零分布，p 值偏乐观（易假阳性）。"
                   "论文主结论不建议用这个口径，请给出 match_var"),
    }


def multi_set_null(sets: Mapping[str, Sequence[str]],
                   scores: Mapping[str, float] | Sequence[float], *,
                   universe: Optional[Sequence[str]] = None,
                   match_var: Optional[Mapping[str, float] | Sequence[float]] = None,
                   statistic: str = "mean", n_iter: int = 1000, n_bins: int = 20,
                   seed: int = 0, alternative: str = "greater",
                   fdr_cut: float = 0.05) -> Dict[str, Any]:
    """一次检验多个基因集，并按 BH 校正（**复用 lib.lm 的唯一裁决点**，不重写）。

    为什么要 BH 而不是逐个看 p：多个集合同时检验时，做 20 个集合在 α=0.05 下
    期望就有 1 个假阳性 —— 那个"唯一显著的图"往往就是它。
    """
    if not sets:
        raise ValueError("sets 为空")
    rows = []
    for name, gs in sets.items():
        r = gene_set_null(gs, scores, universe=universe, match_var=match_var,
                          statistic=statistic, n_iter=n_iter, n_bins=n_bins,
                          seed=seed, alternative=alternative)
        rows.append({
            "set": name, "n_genes": r["n_set"], "observed": r["observed"],
            "null_mean": r["null_mean"], "null_sd": r["null_sd"],
            "z": r["z"], "pvalue": r["pvalue"],
        })
    table = pd.DataFrame(rows)
    table["padj"] = lm.bh_adjust(table["pvalue"].to_numpy())
    table["significant"] = table["padj"] < float(fdr_cut)
    n_sig = int(table["significant"].sum())
    return {
        "table": table,
        "n_tests": int(len(table)),
        "n_significant": n_sig,
        "fdr_cut": float(fdr_cut),
        "fdr_rule": "BH（lib.lm.bh_adjust）—— 与全项目共用一个校正实现",
        "engine": "lib.nullcontrol.multi_set_null",
        "caveat": ("阳性结论要看 **padj**；只报 p 的集合是从多个集合中挑出来的，"
                   "天然偏乐观"),
    }
