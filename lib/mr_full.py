# -*- coding: utf-8 -*-
"""mr_full.py —— 孟德尔随机化全套方法（**委托 vendored 开源实现**）

这里是**包装层，不是算法实现**。所有估计量的数学核心来自
`vendor/_mr/` —— 由 `tools/vendor_sync.py` 从 GitHub 项目
**pymr**（maxghenis/pymr，Unlicense / 公有领域）同步而来，原文未改算法。

为什么不自己写：MR 的稳健估计量（MR-Egger 的 I²_GX 衰减校正、
MR-RAPS 的剖面似然、污染混合的网格搜索）都是易于写错、错得又很隐蔽的
东西——写错后依然会输出一个「看起来合理」的 beta。用开源实现 + 对拍
是唯一可靠的做法。

为什么不用 py-twosamplemr：它是目前最完整的 Python MR 移植，但
**许可为 GPL-3.0**，内嵌会传染本 skill；已按红线规则排除，只可用于
外部对拍（`tools/fetch_tools.py`）。

用法
    from lib import mr_full
    res = mr_full.run_all(beta_exp, se_exp, beta_out, se_out)

退出/异常约定
    输入不足（SNP 数少于方法要求）时由开源实现自行抛 ValueError；
    本层不吞异常——样本量不够就是不够，不该给一个假的估计值。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from vendor._mr import methods as _methods
from vendor._mr import sensitivity as _sens

__all__ = [
    "ESTIMATORS", "SENSITIVITY", "run_all", "estimate", "diagnose",
    "harmonize_data", "bidirectional", "two_step_mediation",
]


def _arr(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"需要一维数组，得到 shape={a.shape}")
    return a


def _wrap(fn: Callable[..., Dict[str, Any]], n_min: int = 2):
    """把开源函数包成统一签名 (bx, sx, by, sy, **kw) -> dict。"""
    def inner(beta_exp, se_exp, beta_out, se_out, **kw):
        bx, sx = _arr(beta_exp), _arr(se_exp)
        by, sy = _arr(beta_out), _arr(se_out)
        if not (bx.size == sx.size == by.size == sy.size):
            raise ValueError("四组输入长度必须一致")
        if bx.size < n_min:
            raise ValueError(f"{fn.__name__} 至少需要 {n_min} 个工具变量，现有 {bx.size}")
        out = dict(fn(bx, sx, by, sy, **kw))
        out["method"] = fn.__name__
        out["n_instruments"] = int(bx.size)
        return out
    inner.__name__ = fn.__name__
    inner.__doc__ = (fn.__doc__ or "") + (
        f"\n\n来源：pymr（Unlicense），见 vendor/_mr/methods.py::{fn.__name__}")
    return inner


# ── 估计量：全部为 vendored 开源实现的直接转发 ────────────────────────
ESTIMATORS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "ivw": _wrap(_methods.ivw, 2),
    "weighted_median": _wrap(_methods.weighted_median, 3),
    "mr_egger": _wrap(_methods.mr_egger, 3),
    "simple_mode": _wrap(_methods.simple_mode, 3),
    "weighted_mode": _wrap(_methods.weighted_mode, 3),
    "mr_raps": _wrap(_methods.mr_raps, 3),
    "contamination_mixture": _wrap(_methods.contamination_mixture, 3),
}


def estimate(name: str, beta_exp, se_exp, beta_out, se_out, **kw) -> Dict[str, Any]:
    """按名字调用单个估计量。未知方法名直接 KeyError，不做静默兜底。"""
    if name not in ESTIMATORS:
        raise KeyError(f"未知 MR 方法 {name!r}；可用：{sorted(ESTIMATORS)}")
    return ESTIMATORS[name](beta_exp, se_exp, beta_out, se_out, **kw)


def run_all(beta_exp, se_exp, beta_out, se_out, **kw) -> Dict[str, Any]:
    """跑遍所有估计量，跳过样本量不足的方法并记录**跳过的具体原因**。

    为什么挨个 try：不同方法的最小 SNP 数不同（加权中位数要 ≥3，
    MR-Egger 实际需要 ≥3 才有意义）。整体失败会让人以为「数据有问题」，
    实际只是某个方法不适用。跳过原因必须出现在输出里。
    """
    bx, sx = _arr(beta_exp), _arr(se_exp)
    by, sy = _arr(beta_out), _arr(se_out)
    results, skipped = {}, {}
    for name, fn in ESTIMATORS.items():
        try:
            results[name] = fn(bx, sx, by, sy, **kw)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"
    if not results:
        raise ValueError(f"没有任何估计量可计算（全部跳过）：{skipped}")
    return {
        "n_instruments": int(bx.size),
        "estimates": results,
        "skipped": skipped,
        "primary": results.get("ivw"),  # IVW 为主要结果；敏感性分析用于判断它是否可信
    }


# ── 敏感性分析 ────────────────────────────────────────────────────────
def cochrans_q(beta_exp, se_exp, beta_out, se_out, causal_estimate):
    """Cochran's Q 异质性检验（含 I²）。"""
    return _sens.cochrans_q(_arr(beta_exp), _arr(se_exp), _arr(beta_out),
                            _arr(se_out), float(causal_estimate))


def rucker_q(beta_exp, se_exp, beta_out, se_out):
    """Rucker's Q：MR-Egger 框架的异质性。"""
    return _sens.rucker_q(_arr(beta_exp), _arr(se_exp), _arr(beta_out), _arr(se_out))


def leave_one_out(beta_exp, se_exp, beta_out, se_out):
    """逐一剔除：哪个 SNP 在单独驱动结果。"""
    return _sens.leave_one_out(_arr(beta_exp), _arr(se_exp),
                               _arr(beta_out), _arr(se_out))


def single_snp(beta_exp, se_exp, beta_out, se_out):
    """单 SNP 的 Wald 比。"""
    return _sens.single_snp(_arr(beta_exp), _arr(se_exp),
                            _arr(beta_out), _arr(se_out))


def steiger_filtering(beta_exp, se_exp, beta_out, se_out, n_exp, n_out):
    """Steiger 过滤：剔除方向可疑（反向因果）的工具变量。"""
    return _sens.steiger_filtering(_arr(beta_exp), _arr(se_exp), _arr(beta_out),
                                   _arr(se_out), int(n_exp), int(n_out))


def funnel_asymmetry(beta_exp, se_exp, beta_out, se_out):
    """漏斗图不对称性（Egger 截距检验）。"""
    return _sens.funnel_asymmetry(_arr(beta_exp), _arr(se_exp),
                                  _arr(beta_out), _arr(se_out))


def radial_mr(beta_exp, se_exp, beta_out, se_out):
    """Radial MR：识别离群工具变量。"""
    return _sens.radial_mr(_arr(beta_exp), _arr(se_exp), _arr(beta_out), _arr(se_out))


SENSITIVITY: Dict[str, Callable[..., Any]] = {
    "cochrans_q": cochrans_q,
    "rucker_q": rucker_q,
    "leave_one_out": leave_one_out,
    "single_snp": single_snp,
    "steiger_filtering": steiger_filtering,
    "funnel_asymmetry": funnel_asymmetry,
    "radial_mr": radial_mr,
}


def diagnose(beta_exp, se_exp, beta_out, se_out, causal_estimate=None,
             **kw) -> Dict[str, Any]:
    """跑整套敏感性分析。需要样本量的检查（Steiger）单独传参。"""
    bx, sx, by, sy = (_arr(beta_exp), _arr(se_exp), _arr(beta_out), _arr(se_out))
    if causal_estimate is None:
        causal_estimate = float(_methods.ivw(bx, sx, by, sy)["beta"])
    out: Dict[str, Any] = {"causal_estimate_used": causal_estimate,
                           "checks": {}, "skipped": {}}
    for name in ("cochrans_q", "rucker_q", "leave_one_out", "single_snp",
                 "funnel_asymmetry", "radial_mr"):
        fn = SENSITIVITY[name]
        args = (bx, sx, by, sy, causal_estimate) if name == "cochrans_q" else (bx, sx, by, sy)
        try:
            out["checks"][name] = fn(*args)
        except (ValueError, TypeError, FloatingPointError,
                np.linalg.LinAlgError) as exc:
            out["skipped"][name] = f"{type(exc).__name__}: {exc}"
    return out


def harmonize_data(exposure, outcome):
    """等位基因协调（链翻转 / 回文剔除）——MR 前置步骤，不做则结果无意义。

    委托 pymr 的实现。传入两个 DataFrame（含 SNP、效应等位基因等列）。
    """
    from vendor._mr.harmonize import harmonize
    return harmonize(exposure, outcome)


# ══════════════════════════════════════════════════════════════════════════════
# 扩展分析：双向 MR 与两步法中介 MR
# ══════════════════════════════════════════════════════════════════════════════
# 这两个是**组合分析**，本身不含新的估计量——正向/反向都复用上面的 run_all
# （源 pymr, Unlicense），中介的间接效应用 delta 法合成。方法学依据：
#   · 双向 MR        —— Davey Smith G, Hemani G. *Hum Mol Genet* 2014;23(R1):R89-98
#   · 两步法中介 MR  —— Burgess S, Daniel RM, Butterworth AS, Thompson SG.
#                       *Eur J Epidemiol* 2015;30:543-52
#                       （间接效应必须用 delta 法或 bootstrap 求 CI）


def bidirectional(forward: Dict[str, Any], reverse: Dict[str, Any],
                  **kw) -> Dict[str, Any]:
    """双向 MR：正向 + 反向各跑一次，并提示是否存在**反向因果 / 水平多效性**信号。

    :param forward: ``{"bx","sx","by","sy"}`` —— 暴露的工具变量对结局的效应
    :param reverse: ``{"bx","sx","by","sy"}`` —— **结局的工具变量**对暴露的效应
    :return: dict，含 ``forward`` / ``reverse`` 的 ``run_all`` 结果与 ``reverse_hint``

    ⚠️ **反向 MR 不是「把数据反过来算一遍」**：必须用**结局的独立工具变量**
       （在结局 GWAS 里达到全基因组显著、且满足 IV 假设的位点）。
       拿同一批暴露 IV 反着算，只是把同一个估计换个写法，得不到任何新信息。

    ⚠️ **双向都显著 ≠ 反向因果**。它同样可能来自**水平多效性**（一个 SNP 同时
       影响两个性状）。要区分，得看反向分析里 IV 假设是否成立
       （异质性、留一、与暴露的关联强度）。
    """
    # 允许用简写键 bx/sx/by/sy，内部映射到 run_all 的参数名
    alias = {"bx": "beta_exp", "sx": "se_exp", "by": "beta_out", "sy": "se_out"}

    def _norm(d: Dict[str, Any]) -> Dict[str, Any]:
        return {alias.get(k, k): v for k, v in d.items()}

    fwd = run_all(**_norm(forward), **kw)
    rev = run_all(**_norm(reverse), **kw)

    def _p(res: Dict[str, Any]):
        """取主要结果（IVW）的 p 值。"""
        pr = res.get("primary") or {}
        return pr.get("pval", pr.get("pvalue"))

    pf, pr = _p(fwd), _p(rev)
    hint, flag = None, "undetermined"
    if pf is not None and pr is not None:
        if pf < 0.05 and pr < 0.05:
            flag = "both_significant"
            hint = ("双向均显著 —— 可能是反向因果，也可能是水平多效性；"
                    "需用反向分析的异质性 / 留一 / 工具强度区分")
        elif pf < 0.05:
            flag = "forward_only"
            hint = "仅正向显著 —— 未见反向因果提示"
        else:
            flag = "neither"
            hint = "正向不显著 —— 主假设未获支持"
    return {
        "forward": fwd,
        "reverse": rev,
        "p_forward": pf,
        "p_reverse": pr,
        # 结构化标志：调用方应按它分支，不要去解析下面那句自然语言提示
        "reverse_flag": flag,
        "reverse_hint": hint,
        "engine": "lib.mr_full.bidirectional",
    }


def two_step_mediation(b_step1: float, se_step1: float,
                       b_step2: float, se_step2: float,
                       b_total: Optional[float] = None,
                       se_total: Optional[float] = None) -> Dict[str, Any]:
    """两步法中介 MR：暴露 →（中介）→ 结局。

    :param b_step1, se_step1: 第 1 步 MR —— 暴露对中介的效应
    :param b_step2, se_step2: 第 2 步 MR —— 中介对结局的效应（**须调整暴露**）
    :param b_total, se_total: 暴露对结局的**总效应**（给了才能算直接效应与中介比例）
    :return: dict
        - ``indirect`` / ``se`` / ``ci95`` / ``pvalue``：间接效应（delta 法）
        - ``direct`` / ``proportion_mediated``：需要总效应
        - ``method``：``"two-step MR, delta method"``

    ⚠️ **间接效应必须用 delta 法或 bootstrap 求 CI**：``a×b`` 是两个估计的乘积，
       **不服从正态分布**；用「a 的 CI 与 b 的 CI 是否含 0」来判断中介是错的
       （这也是 Sobel 检验在小样本失真的原因）。
    ⚠️ **两步法默认不存在暴露–中介交互**。若怀疑有交互（有调节的中介），
       两步法会低估中介效应，需换用中介–调节拆解。
    """
    import numpy as _np
    from scipy import stats as _st

    indirect = float(b_step1) * float(b_step2)
    se_ind = float(_np.sqrt((float(b_step2) ** 2) * float(se_step1) ** 2
                            + (float(b_step1) ** 2) * float(se_step2) ** 2))
    z = indirect / se_ind if se_ind > 0 else float("nan")
    pval = float(2 * _st.norm.sf(abs(z))) if _np.isfinite(z) else float("nan")
    out: Dict[str, Any] = {
        "indirect": indirect,
        "se": se_ind,
        "ci95": (indirect - 1.96 * se_ind, indirect + 1.96 * se_ind),
        "pvalue": pval,
        "step1": float(b_step1),
        "step2": float(b_step2),
        "method": "two-step MR, delta method",
        "engine": "lib.mr_full.two_step_mediation",
    }
    if b_total is not None:
        direct = float(b_total) - indirect
        out["total"] = float(b_total)
        out["direct"] = direct
        out["proportion_mediated"] = (indirect / float(b_total)
                                      if float(b_total) != 0 else float("nan"))
        if se_total is not None:
            out["direct_ci95"] = (direct - 1.96 * float(se_total),
                                  direct + 1.96 * float(se_total))
    return out
