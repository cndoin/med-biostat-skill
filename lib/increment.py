# SKILL-OWNED —— 本 skill 自有实现（可直接编辑；不要与 vendor/ 的同步副本混淆）
"""increment.py —— 增量预测价值：Harrell C-index / 连续 NRI / IDI + E-value

补齐 PROTOCOL 明确要求、而现有引擎缺失的判据（LPDC 第 9.2 节）：

  * **C-index（Harrell）**：生存资料的区分度，处理删失的可比对
  * **连续 NRI**：新模型相对旧模型，在事件组/非事件组的净重分类改善
  * **IDI**：综合判别改善指数 = 区分度提升 − 校准代价
  * **E-value**：未测混杂需要多强才能解释掉观察到的关联

依据：
  C-index  — Harrell 1982；比较规则见 Pencina & D'Agostino 2004
  NRI/IDI  — Pencina et al. 2008 (Stat Med) / 2011
  E-value  — VanderWeele & Ding 2017 (Ann Intern Med)

依赖：numpy（Tier 1）
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------- C-index

def c_index(time: Sequence[float], event: Sequence[int],
            risk: Sequence[float]) -> Dict[str, float]:
    """Harrell C-index（生存资料区分度）。

    risk 越大表示预测风险越高。并列按 0.5 计。返回 c 与各类对计数（便于审计）。

    **可比对规则**（与 `lifelines.utils.concordance._concordance_summary_statistics`
    逐条对齐；三处曾经写错，偏差最大 0.25）：

    1. `ta == tb`（同刻）→ 仅当**恰好一个发生事件**时可比。
       两个都事件或都删失都无法定先后。
    2. `ta != tb` → 可比当且仅当「**先到的那个人确实发生了事件**」：
       (ea and ta < tb) or (eb and tb < ta)。
    3. 一致方向 = **风险更高者事件更早**。风险相等记 0.5。

    为什么不能用「至少一个发生事件」：那样会把「删失者在前、后面才发生事件」
    的 pair 纳入 —— 该 pair 的先后顺序本质上不可知，纳入后结论会随风险值
    方向翻转（同一份数据 c 从 1.0 变 0.0），是两个无意义数字。
    """
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    r = np.asarray(risk, float)
    n = len(t)
    if n < 2:
        return {"c": np.nan, "n_pairs": 0, "concordant": 0, "discordant": 0, "tied": 0}

    i, j = np.triu_indices(n, k=1)
    ti, tj = t[i], t[j]
    ei, ej = e[i], e[j]
    ri, rj = r[i], r[j]

    same_time = ti == tj
    ei_b, ej_b = ei == 1, ej == 1

    # 规则 1：同刻 → 恰好一个事件才可比
    usable_same = same_time & (ei_b != ej_b)
    # 规则 2：异刻 → 先到者必须是事件
    usable_diff = (~same_time) & ((ei_b & (ti < tj)) | (ej_b & (tj < ti)))
    usable = usable_same | usable_diff

    if not np.any(usable):
        return {"c": np.nan, "n_pairs": 0, "concordant": 0, "discordant": 0, "tied": 0}

    ti, tj, ei_b, ej_b, ri, rj = (ti[usable], tj[usable], ei_b[usable], ej_b[usable],
                                 ri[usable], rj[usable])
    same_time = ti == tj

    # 规则 3：一致 = 风险更高者事件更早。
    # 异刻时「事件更早」= 时间更小；同刻时由事件状态决定先后（恰好一个事件）。
    # 统一表示成：先发生者 high_risk_first ⇔ 其风险值更高。
    #   异刻：i 先发生 ⇔ ti < tj
    #   同刻：i 视作先发生 ⇔ ei_b 为真（此时 ej_b 必为假）
    i_first = np.where(same_time, ei_b, ti < tj)
    r_first = np.where(i_first, ri, rj)      # 先发生者的风险
    r_second = np.where(i_first, rj, ri)     # 后发生者的风险

    concordant = int(np.sum(r_first > r_second))
    discordant = int(np.sum(r_first < r_second))
    tied = int(np.sum(r_first == r_second))
    denom = concordant + discordant + tied
    c = (concordant + 0.5 * tied) / denom if denom else float("nan")
    return {"c": float(c), "n_pairs": int(denom),
            "concordant": concordant, "discordant": discordant, "tied": tied}

    concordant = int(np.sum(r_early > r_late))
    discordant = int(np.sum(r_early < r_late))
    tied = int(np.sum(r_early == r_late))
    denom = concordant + discordant + tied
    c = (concordant + 0.5 * tied) / denom if denom else float("nan")
    return {"c": float(c), "n_pairs": int(denom),
            "concordant": concordant, "discordant": discordant, "tied": tied}


# ---------------------------------------------------------------- NRI / IDI

def continuous_nri(y: Sequence[int], p_old: Sequence[float],
                   p_new: Sequence[float]) -> Dict[str, float]:
    """连续 NRI（Pencina 2008）。y 为 0/1 结局，p 为预测概率。

    NRI_e  = P(p_new > p_old | 事件) − P(p_new < p_old | 事件)
    NRI_ne = P(p_new < p_old | 非事件) − P(p_new > p_old | 非事件)
    NRI    = NRI_e + NRI_ne
    """
    y = np.asarray(y, int)
    po = np.asarray(p_old, float)
    pn = np.asarray(p_new, float)
    ev, ne = y == 1, y == 0
    if ev.sum() == 0 or ne.sum() == 0:
        return {"nri": np.nan, "nri_events": np.nan, "nri_nonevents": np.nan}

    de = pn[ev] - po[ev]
    dn = pn[ne] - po[ne]
    nri_e = float(np.mean(de > 0) - np.mean(de < 0))
    nri_ne = float(np.mean(dn < 0) - np.mean(dn > 0))
    return {"nri": nri_e + nri_ne, "nri_events": nri_e, "nri_nonevents": nri_ne,
            "n_events": int(ev.sum()), "n_nonevents": int(ne.sum())}


def idi(y: Sequence[int], p_old: Sequence[float],
        p_new: Sequence[float]) -> Dict[str, float]:
    """综合判别改善指数 IDI = (IS_new − IS_old) − (IP_new − IP_old)。

    IS = 事件组平均预测概率；IP = 非事件组平均预测概率。
    """
    y = np.asarray(y, int)
    po = np.asarray(p_old, float)
    pn = np.asarray(p_new, float)
    ev, ne = y == 1, y == 0
    if ev.sum() == 0 or ne.sum() == 0:
        return {"idi": np.nan}
    is_old, is_new = float(po[ev].mean()), float(pn[ev].mean())
    ip_old, ip_new = float(po[ne].mean()), float(pn[ne].mean())
    return {"idi": (is_new - is_old) - (ip_new - ip_old),
            "discrimination_gain": is_new - is_old,
            "calibration_cost": ip_new - ip_old}


def auroc(y: Sequence[int], p: Sequence[float]) -> float:
    """AUROC（Mann-Whitney U 形式，含平局 0.5）。"""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    ev, ne = p[y == 1], p[y == 0]
    if ev.size == 0 or ne.size == 0:
        return float("nan")
    diff = ev[:, None] - ne[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


# ---------------------------------------------------------------- E-value

def e_value(estimate: float, ci_low: Optional[float] = None,
            measure: str = "RR") -> Dict[str, float]:
    """E-value：未测混杂需要多强（在 RR 尺度上）才能把关联解释掉。

    公式（VanderWeele & Ding 2017）：
        E-value = RR + sqrt(RR·(RR−1))
    对 RR<1 先取倒数；OR/HR 需先从 OR 近似转换到 RR：
        RR ≈ OR^0.5  (VanderWeele 2017 的常用近似，应在文中声明)
    """
    def _ev(rr: float) -> float:
        rr = float(rr)
        if rr <= 0:
            return float("nan")
        if rr < 1.0:
            rr = 1.0 / rr
        return rr + float(np.sqrt(rr * (rr - 1.0)))

    def _to_rr(x: float) -> float:
        x = float(x)
        if measure == "RR":
            return x
        if measure == "OR":
            return x ** 0.5 if x >= 1 else (1.0 / x) ** 0.5 if x < 1 else x
        if measure == "HR":
            # 罕见结局下 HR≈RR；否则用近似 RR ≈ (1-0.5^sqrt(HR))/(1-0.5^sqrt(1/HR))
            if x <= 0:
                return float("nan")
            return (1 - 0.5 ** np.sqrt(x)) / (1 - 0.5 ** np.sqrt(1.0 / x))
        return x

    point = _ev(_to_rr(estimate))
    out = {"e_value": point, "measure": measure, "estimate_as_rr": _to_rr(estimate)}
    if ci_low is not None:
        out["e_value_ci"] = (0.0 if (_to_rr(ci_low) <= 1.0 <= _to_rr(estimate)
                                     or _to_rr(estimate) <= 1.0 <= _to_rr(ci_low))
                             else _ev(_to_rr(ci_low)))
    return out


# ---------------------------------------------------------------- 汇总

def increment_table(y, p_old, p_new, time=None, event=None,
                    risk_old=None, risk_new=None) -> Dict:
    """一次性给出增量价值的全部指标（有生存信息时附 C-index）。"""
    res = {
        "nri": continuous_nri(y, p_old, p_new),
        "idi": idi(y, p_old, p_new),
        "auroc_old": auroc(y, p_old),
        "auroc_new": auroc(y, p_new),
    }
    res["delta_auroc"] = res["auroc_new"] - res["auroc_old"]
    if time is not None and event is not None and risk_old is not None and risk_new is not None:
        c_old = c_index(time, event, risk_old)["c"]
        c_new = c_index(time, event, risk_new)["c"]
        res["c_index_old"] = c_old
        res["c_index_new"] = c_new
        res["delta_c_index"] = c_new - c_old
    return res
