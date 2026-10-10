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

from typing import Any, Callable, Dict, Sequence

import numpy as np

from vendor._mr import methods as _methods
from vendor._mr import sensitivity as _sens

__all__ = [
    "ESTIMATORS", "SENSITIVITY", "run_all", "estimate", "diagnose",
    "harmonize_data",
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
