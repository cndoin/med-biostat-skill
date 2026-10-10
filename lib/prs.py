"""分析法 ⑩：多基因风险评分（PRS / PGS）——clumping + P+T + 评分 + 评估。

流程（与 PGS Catalog / PRSice 的口径对齐）：
  1. `clump`      — LD 修剪：每个位点窗口内只留最显著的一个（需 LD 来源或位置窗口退化）
  2. `score`      — 按 P 值阈值（P+T）筛选后，Σ β_i · G_i（可加权）
  3. `evaluate`   — 用 R² / AUC 增量 / OR per SD 评估，**必须在独立验证集上做**
  4. `find_best_threshold` — 在**验证集**上选最优 P 阈值，再到**测试集**上报告

⚠️ 三条最容易翻车的规矩：
  1. **阈值只能在验证集上选**。在测试集上选最优阈值再报性能 = 自欺欺人。
  2. **目标人群与基础 GWAS 的祖源必须匹配**。欧洲 GWAS 的 PRS 用到亚洲人群，
     预测性能会大幅衰减——这不是 bug，是遗传结构差异，必须如实报告。
  3. 评分前必须做**等位基因协调与链翻转处理**，否则 β 的符号会被系统性弄反。
     （MR 侧的协调在 `vendor/_mr/harmonize.py`，PRS 侧这里只做最基本的检查。）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scipy import stats as _st


def clump(snp: Sequence[str], chrom: Sequence[Any], pos: Sequence[int],
          pvalue: Sequence[float], beta: Sequence[float],
          window_kb: int = 250, r2_threshold: Optional[float] = None,
          ld: Optional[Dict[Tuple[str, str], float]] = None,
          p_threshold: float = 5e-8) -> Dict[str, Any]:
    """LD clumping：显著位点里，窗口内保留 P 最小的，剔除与之连锁的。

    :param ld: {(snp_a, snp_b): r2}；**不给就退化成位置窗口**，
               此时只按物理距离去冗余（会留下真实连锁的位点，须如实标注）
    :param r2_threshold: 有 LD 时的 r²阈值（常用 0.1 或 0.2）
    """
    df = pd.DataFrame({"snp": [str(s) for s in snp], "chrom": list(chrom),
                       "pos": [int(p) for p in pos],
                       "p": [float(x) for x in pvalue],
                       "beta": [float(x) for x in beta]})
    sig = df[df["p"] <= p_threshold].sort_values("p").reset_index(drop=True)
    if len(sig) == 0:
        return {"kept": [], "dropped": [], "n_kept": 0,
                "note": f"没有位点通过 P <= {p_threshold}"}
    kept: List[str] = []
    dropped: List[Dict[str, Any]] = []
    for i in range(len(sig)):
        row = sig.iloc[i]
        redundant = False
        for k in kept:
            r = sig[sig["snp"] == k].iloc[0]
            if r["chrom"] != row["chrom"]:
                continue
            if ld is not None and r2_threshold is not None:
                key = (k, row["snp"]) if (k, row["snp"]) in ld else (row["snp"], k)
                r2 = ld.get(key)
                if r2 is not None and r2 >= r2_threshold:
                    redundant = True
                    break
            elif abs(int(r["pos"]) - int(row["pos"])) <= window_kb * 1000:
                redundant = True
                break
        if redundant:
            dropped.append({"snp": row["snp"], "p": float(row["p"])})
        else:
            kept.append(row["snp"])
    sub = sig[sig["snp"].isin(kept)].reset_index(drop=True)
    return {
        "kept": list(sub["snp"]),
        "dropped": dropped,
        "n_kept": len(sub),
        "p_threshold": p_threshold,
        "degenerate": ld is None,
        "rule": ("位置窗口去冗余（未给 LD）——会残留真实连锁位点，须标注"
                 if ld is None else f"LD r² >= {r2_threshold} 剔除"),
        "table": sub,
    }


def score(genotypes: pd.DataFrame, weights: pd.DataFrame,
          snp_col: str = "snp", beta_col: str = "beta",
          center: bool = True, scale: bool = False) -> pd.Series:
    """计算 PRS：score_i = Σ_j β_j · G_ij（默认加性剂量 0/1/2）。

    :param genotypes: 样本×SNP 的基因型剂量矩阵
    :param weights: 含 snp 与 beta 两列（来自 clump 结果）
    """
    common = [s for s in weights[snp_col] if s in set(genotypes.columns)]
    if not common:
        raise ValueError("weights 与 genotypes 没有共同 SNP——检查 ID 体系")
    w = weights.set_index(snp_col).loc[common, beta_col].astype(float)
    G = genotypes[common].to_numpy(dtype=float)
    if center:
        G = G - np.nanmean(G, axis=0)
    raw = np.nan_to_num(G) @ w.to_numpy()
    if scale:
        sd = float(np.std(raw, ddof=1))
        raw = raw / sd if sd > 0 else raw
    return pd.Series(raw, index=genotypes.index, name="PRS")


def evaluate(prs: Sequence[float], outcome, kind: str = "binary",
             covariates: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """PRS 评估。

    kind:
      · binary   — 给出 OR per SD、AUC、以及相对协变量模型的 AUC 增量
      · continuous — 给出 R² 与相对协变量模型的 R² 增量
    ⚠️ 增量必须与**不含 PRS 的基线模型**比较；只报 PRS 单变量的 AUC/R²
       等于把年龄性别等的作用算进了 PRS，是系统性高估。
    """
    p = np.asarray(prs, dtype=float)
    y = np.asarray(outcome)
    base = None if covariates is None else np.asarray(covariates, dtype=float)
    if base is not None and base.ndim == 1:
        base = base.reshape(-1, 1)

    def _design(with_prs: bool) -> np.ndarray:
        cols = [np.ones(len(p))]
        if base is not None:
            cols.append(base)
        if with_prs:
            cols.append(p.reshape(-1, 1))
        return np.column_stack(cols)

    if kind == "binary":
        y = np.asarray(y, dtype=int)
        Z = (p - p.mean()) / (p.std(ddof=1) if p.std(ddof=1) > 0 else 1.0)

        def _logit(Xd, yv):
            b = np.zeros(Xd.shape[1])
            for _ in range(200):
                e = 1 / (1 + np.exp(-np.clip(Xd @ b, -30, 30)))
                g = Xd.T @ (yv - e) / len(yv)
                b = b + 0.5 * g
            return b

        b_full = _logit(_design(True), y)
        b_base = _logit(_design(False), y)
        or_per_sd = float(np.exp(b_full[-1]))
        auc = _auc(y, p)
        auc_base = float("nan")
        if base is not None:
            auc_base = _auc(y, _design(False) @ b_base)
        # 似然比检验
        ll = lambda Xd, yv, b: float(np.sum(yv * (Xd @ b) - np.log1p(np.exp(np.clip(Xd @ b, -30, 30)))))
        lr = 2 * (ll(_design(True), y, b_full) - ll(_design(False), y, b_base))
        return {
            "kind": "binary", "or_per_sd": or_per_sd,
            "auc": auc, "auc_baseline": auc_base,
            "auc_increment": auc - auc_base if base is not None else float("nan"),
            "lr_stat": lr, "pvalue": float(1 - _st.chi2.cdf(lr, 1)),
            "n": int(len(y)), "n_cases": int(y.sum()),
        }
    # 连续
    y = np.asarray(y, dtype=float)
    Xa, Xb = _design(True), _design(False)

    def _r2(Xd):
        b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        pred = Xd @ b
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2_full, r2_base = _r2(Xa), _r2(Xb)
    n, k = len(y), Xa.shape[1]
    f = ((r2_full - r2_base) / max(1 - r2_full, 1e-12)) * ((n - k - 1) / 1)
    return {
        "kind": "continuous", "r2": r2_full, "r2_baseline": r2_base,
        "r2_increment": r2_full - r2_base,
        "f_stat": float(f), "pvalue": float(1 - _st.f.cdf(f, 1, max(n - k - 1, 1))),
        "n": int(n),
    }


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC（Mann-Whitney U / n_pos·n_neg）。"""
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u, _ = _st.mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u / (len(pos) * len(neg)))


def find_best_threshold(genotypes: pd.DataFrame, weights: pd.DataFrame,
                        y_valid, thresholds: Sequence[float] = (5e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.5),
                        kind: str = "binary", snp_col: str = "snp",
                        beta_col: str = "beta", p_col: str = "p") -> pd.DataFrame:
    """在**验证集**上扫 P+T 阈值网格，选最优。

    ⚠️ 选出来的阈值只能用于**独立的测试集**报告性能。
       在验证集上选完再在验证集上报告 = 过拟合的乐观值。
    """
    rows = []
    for t in thresholds:
        sub = weights[weights[p_col] <= t]
        if len(sub) == 0:
            continue
        s = score(genotypes, sub, snp_col=snp_col, beta_col=beta_col)
        ev = evaluate(s.to_numpy(), y_valid, kind=kind)
        rows.append({"p_threshold": t, "n_snp": int(len(sub)),
                     "auc" if kind == "binary" else "r2": ev.get("auc", ev.get("r2")),
                     "pvalue": ev["pvalue"]})
    df = pd.DataFrame(rows)
    if len(df):
        key = "auc" if kind == "binary" else "r2"
        df["best"] = df[key] == df[key].max()
    return df


def quantile_risk(prs: Sequence[float], outcome, n_quantiles: int = 10,
                  kind: str = "binary") -> pd.DataFrame:
    """PRS 分位数的风险梯度（PRS 论文的标准图：十分位 OR 或均值）。"""
    p = np.asarray(prs, dtype=float)
    y = np.asarray(outcome)
    q = pd.qcut(p, n_quantiles, labels=False, duplicates="drop")
    rows = []
    ref = None
    for lev in range(int(np.nanmax(q)) + 1):
        m = q == lev
        if kind == "binary":
            yy = y[m].astype(int)
            prev = float(yy.mean()) if len(yy) else float("nan")
            odds = prev / (1 - prev) if 0 < prev < 1 else float("nan")
            if ref is None:
                ref = odds
            rows.append({"quantile": lev + 1, "n": int(m.sum()), "prevalence": prev,
                         "odds": odds, "or_vs_q1": odds / ref if ref else float("nan")})
        else:
            rows.append({"quantile": lev + 1, "n": int(m.sum()),
                         "mean": float(np.mean(y[m])), "sd": float(np.std(y[m], ddof=1))})
    return pd.DataFrame(rows)
