# SKILL-OWNED —— 本 skill 自有实现（可直接编辑）
"""competing.py —— 竞争风险：累积发生率函数（Aalen–Johansen）+ 组间比较

LPDC 第 9.4 节把竞争风险（以死亡为竞争事件）列为**必做敏感性分析**。
本模块实现 Aalen–Johansen 估计量（非参数 CIF）与两组比较的近似检验。

公式：
    CIF_k(t) = Σ_{t_i ≤ t} S(t_i⁻) · d_k(t_i) / n(t_i)
    其中 S 为全因生存（Aalen–Johansen 的乘积限），d_k 为 k 因事件数，
    n 为风险人数。

依赖：numpy
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def _at_risk(times: np.ndarray, t: float) -> int:
    return int(np.sum(times >= t))


def cif(time: Sequence[float], event: Sequence[int], cause: int = 1) -> Dict:
    """Aalen–Johansen 累积发生率。

    event: 0 = 删失；`cause` = 目标事件；其它非零值 = 竞争事件。
    """
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    order = np.argsort(t)
    t, e = t[order], e[order]

    uniq = np.unique(t)
    surv = 1.0                    # 全因生存 S(t)
    times: List[float] = [0.0]
    cifs: List[float] = [0.0]
    survival: List[float] = [1.0]

    for tt in uniq:
        block = t == tt
        n = _at_risk(t, tt)
        if n == 0:
            continue
        d_all = int(np.sum(e[block] != 0))
        d_k = int(np.sum(e[block] == cause))
        # 先按上一步的 S(t⁻) 累加 CIF_k，再更新 S(t)
        if d_k:
            times.append(float(tt))
            cifs.append(cifs[-1] + surv * d_k / n)
            survival.append(surv)
        surv *= (1.0 - d_all / n)

    return {"cause": cause, "time": times, "cif": cifs, "overall_survival": survival,
            "final_cif": cifs[-1] if cifs else 0.0,
            "final_survival": surv}


def cif_by_group(time, event, group, cause: int = 1) -> Dict:
    """按组分别估 CIF，并对最终 CIF 做两组差的正态近似检验（粗判，非 Gray 检验）。"""
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    g = np.asarray(group)
    groups = sorted(set(g.tolist()))
    out = {}
    for grp in groups:
        m = g == grp
        out[str(grp)] = cif(t[m], e[m], cause=cause)

    if len(groups) == 2:
        a = out[str(groups[0])]["final_cif"]
        b = out[str(groups[1])]["final_cif"]
        na, nb = int(np.sum(g == groups[0])), int(np.sum(g == groups[1]))
        # 二项近似标准误
        se = float(np.sqrt(a * (1 - a) / max(na, 1) + b * (1 - b) / max(nb, 1)))
        z = (a - b) / se if se > 0 else float("nan")
        out["comparison"] = {
            "group_a": str(groups[0]), "group_b": str(groups[1]),
            "cif_a": a, "cif_b": b, "difference": a - b,
            "z_approx": z,
            "note": ("正态近似仅作粗判；正式比较应使用 Gray 检验（cmprsk::cuminc 或 "
                     "survival::finegray 对拍）。竞争风险下不得用 1−KM。"),
        }
    return out
