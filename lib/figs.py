# -*- coding: utf-8 -*-
"""Fig3 失守曲线：GAM 拟合 + 95% 置信带 + 失守点标注，**纯 SVG、确定性**。

为什么不直接用 matplotlib / ggplot2：
  * **可复现**：同一份输入必得同一个字节串，能进 git diff、能当快照断言的对象；
  * **零绘图依赖**：不依赖字体/后端配置，CI 里也能出图；
  * **矢量可直接进论文**：SVG 能被 Inkscape/Illustrator 打开改字号描边，不用重跑统计。

这个模块**只画图，不算任何统计量** —— 曲线与标注值全部来自
``lib.age_onset.fit_onset`` 的输出，这里不判断、不重算、不外推。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from vendor.viz import svg_charts as _svg

__all__ = ["onset_curve"]

_INK = _svg.INK
_MUTED = _svg.MUTED
_GRID = _svg.GRID
_BLUE = _svg.BLUE
_BLUE_FILL = _svg.BLUE_FILL
_RED = _svg.RED
_AMBER = _svg.AMBER
_DOT = "#9AA7B4"
_MAX_POINTS = 900          # 散点上限：超过就抽稀，保证 SVG 体积可控且联网可传


def _ticks(lo: float, hi: float, n: int = 5) -> list:
    """等距刻度（自己实现，不依赖 vendored 模块的私有函数）。"""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [0.0, 1.0]
    step = (hi - lo) / float(n)
    return [lo + i * step for i in range(n + 1)]


def _fmt(v: float) -> str:
    """刻度文本：定宽输出，避免不同数据导致 SVG 长度漂移（不利于快照比对）。"""
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 100 or a < 0.01:
        return f"{v:.3g}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def onset_curve(onset: Dict[str, Any], *,
                ages: Optional[Sequence[float]] = None,
                expr: Optional[Sequence[float]] = None,
                title: str = "年龄失守曲线",
                ylabel: str = "表达（GAM 拟合值）",
                xlabel: str = "年龄",
                gene_label: Optional[str] = None,
                width: int = 760, height: int = 470,
                footer: Optional[Sequence[str]] = None) -> str:
    """把 ``lib.age_onset.fit_onset`` 的结果画成 Fig3 那样的失守曲线。

    :param onset: ``fit_onset`` 的返回值（用 ``grid_ages/grid_fitted/grid_lo/grid_hi/
                  onset_age/baseline/detected/degenerate/direction/pvalue``）
    :param ages: 可选，原始观测点的年龄（画散点，用来展示样本在年龄轴上的覆盖）
    :param expr: 可选，原始观测点的表达
    :param footer: 可选的额外注记行（追加在右下角口径说明下面）
    :returns: SVG 字符串

    三处口径写死在这里（改之前先看这三行）：
      1. **阴影带就是 fit_onset 给的 grid_lo/grid_hi** —— GAM 的**逐点** 95% CI，
         不是同时置信带（simultaneous band）。后者更宽，论文里不得混称；
      2. **失守点竖线只在 ``detected=True`` 时画**。趋势不显著或 degenerate 时改为
         在图上直接写明原因，不画一条"看起来好像有个失守点"的线；
      3. 纵轴范围由**所有要画的元素**共同决定（曲线 ± CI + 散点），
         只取拟合值范围会把散点和 CI 裁掉 —— 那样图会"看起来拟合得更好"。
    """
    required = ("grid_ages", "grid_fitted", "grid_lo", "grid_hi", "detected")
    missing = [k for k in required if k not in onset]
    if missing:
        raise ValueError(f"onset 缺少字段 {missing}；应传入 lib.age_onset.fit_onset 的输出")

    ga = np.asarray(onset["grid_ages"], dtype=float).ravel()
    gf = np.asarray(onset["grid_fitted"], dtype=float).ravel()
    glo = np.asarray(onset["grid_lo"], dtype=float).ravel()
    ghi = np.asarray(onset["grid_hi"], dtype=float).ravel()
    if not (ga.size == gf.size == glo.size == ghi.size):
        raise ValueError("grid_ages/grid_fitted/grid_lo/grid_hi 长度必须一致")

    detected = bool(onset.get("detected"))
    degenerate = bool(onset.get("degenerate", False))
    base_raw = onset.get("baseline")
    baseline = float(base_raw) if base_raw is not None and np.isfinite(float(base_raw)) else None

    xs, ys = None, None
    if ages is not None and expr is not None:
        xs = np.asarray(list(ages), dtype=float).ravel()
        ys = np.asarray(list(expr), dtype=float).ravel()
        if xs.size != ys.size:
            raise ValueError(f"ages({xs.size}) 与 expr({ys.size}) 长度不一致")
        keep = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[keep], ys[keep]

    has_curve = bool(ga.size > 1)
    if not has_curve and xs is None:
        raise ValueError("既没有可用的 grid 也没有散点，无从作图")

    parts = []
    if has_curve:
        parts += [ga, gf, glo, ghi]
    if xs is not None and xs.size:
        parts += [xs, ys]
    allv = np.concatenate([np.asarray(p).ravel() for p in parts])
    allv = allv[np.isfinite(allv)]
    if allv.size == 0:
        raise ValueError("grid 与散点里都没有可用数值（全 NaN），无法作图")
    if baseline is not None:
        allv = np.concatenate([allv, [baseline]])

    xall = np.concatenate([ga] + ([xs] if xs is not None and xs.size else []))
    xall = xall[np.isfinite(xall)]
    xlo, xhi = float(xall.min()), float(xall.max())
    ylo, yhi = float(allv.min()), float(allv.max())
    if not np.isfinite(xlo) or not np.isfinite(xhi) or xhi <= xlo:
        xlo, xhi = (xlo if np.isfinite(xlo) else 0.0) - 0.5, (xlo if np.isfinite(xlo) else 0.0) + 0.5
    if not np.isfinite(ylo) or not np.isfinite(yhi) or yhi <= ylo:
        ylo, yhi = (ylo if np.isfinite(ylo) else 0.0) - 0.5, (ylo if np.isfinite(ylo) else 0.0) + 0.5
    padx, pady = 0.04 * (xhi - xlo), 0.06 * (yhi - ylo)
    xlo, xhi, ylo, yhi = xlo - padx, xhi + padx, ylo - pady, yhi + pady

    L, R, T, B = 74, 24, 56, 96
    pw, ph = width - L - R, height - T - B

    def sx(v: float) -> float:
        return L + (float(v) - xlo) / (xhi - xlo) * pw

    def sy(v: float) -> float:
        return T + ph - (float(v) - ylo) / (yhi - ylo) * ph

    head = title + (f" · {gene_label}" if gene_label else "")
    c = _svg.Canvas(width, height, title=head,
                    desc="年龄失守曲线：GAM 拟合 + 95% 逐点置信带（未检出时不画竖线）")
    c.rect(0, 0, width, height, fill="#FFFFFF", stroke="none")

    # ── 网格与刻度
    for t in _ticks(ylo, yhi, 5):
        c.line(L, sy(t), L + pw, sy(t), color=_GRID, sw=1.0)
        c.text(L - 8, sy(t), _fmt(t), size=11, color=_MUTED, anchor="end",
               baseline="central")
    for t in _ticks(xlo, xhi, 5):
        c.line(sx(t), T, sx(t), T + ph, color=_GRID, sw=1.0)
        c.text(sx(t), T + ph + 16, _fmt(t), size=11, color=_MUTED, anchor="middle")
    c.rect(L, T, pw, ph, fill="none", stroke=_INK, sw=1.0)

    # ── 标题与轴标题
    c.text(L, T - 16, head, size=13, color=_INK, weight=600)
    c.text(L + pw / 2, T + ph + 38, xlabel, size=12, color=_INK, anchor="middle")
    c.add(f'<text x="20" y="{T + ph / 2:.1f}" font-size="12" fill="{_INK}" '
          f'text-anchor="middle" transform="rotate(-90 20 {T + ph / 2:.1f})" '
          f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">'
          f'{_svg._esc(ylabel)}</text>')

    # ── 散点（画在最下层）
    if xs is not None and xs.size:
        step = max(1, int(np.ceil(xs.size / _MAX_POINTS)))
        for i in range(0, xs.size, step):
            c.circle(sx(xs[i]), sy(ys[i]), r=1.9, fill=_DOT, stroke="none")

    # ── 置信带 + 拟合线 + 基线
    if has_curve:
        band = [(sx(a), sy(lo)) for a, lo in zip(ga, glo)]
        band += [(sx(a), sy(hi)) for a, hi in zip(ga[::-1], ghi[::-1])]
        c.polyline(band, color="none", fill=_BLUE_FILL, sw=0)
        c.polyline([(sx(a), sy(v)) for a, v in zip(ga, gf)], color=_BLUE, sw=2.2)
        if baseline is not None:
            c.line(L, sy(baseline), L + pw, sy(baseline), color=_AMBER, sw=1.4, dash="6 4")
            c.text(L + pw - 6, sy(baseline) - 6, "基线 = 最年轻段均值", size=10,
                   color=_AMBER, anchor="end")

    # ── 失守点：只在 detected 时画
    if detected and onset.get("onset_age") is not None:
        ox = sx(float(onset["onset_age"]))
        c.line(ox, T, ox, T + ph, color=_RED, sw=1.6, dash="5 3")
        c.circle(ox, T + 12, r=3.4, fill=_RED, stroke="none")
        right_side = ox < L + pw * 0.72
        c.text(ox + (6 if right_side else -6), T + 16,
               f"失守点 {float(onset['onset_age']):.1f}", size=11, color=_RED,
               anchor="start" if right_side else "end")
    elif degenerate:
        c.text(L + pw / 2, T + ph / 2, "样本不足以拟合（degenerate）", size=13,
               color=_MUTED, anchor="middle")
    else:
        p = onset.get("pvalue")
        ptxt = ("趋势 p 不可用" if p is None or not np.isfinite(float(p))
                else f"趋势不显著 p={float(p):.3g}")
        # 这里写"未检出失守"而不是"未检出失守点"：后者含"失守点"三字，
        # 人在扫图时容易读成"有个失守点"，属于文案级别的误导。
        c.text(L + pw / 2, T + ph / 2, f"未检出失守 · {ptxt}", size=13,
               color=_MUTED, anchor="middle")

    # ── 右下角口径注记：图本身就是实验记录的一部分
    notes = []
    if has_curve:
        notes.append("阴影 = GAM 逐点 95% CI（非同时置信带）")
    if onset.get("rule"):
        notes.append(f"失守判据 = {onset['rule']}")
    if onset.get("edof") is not None and np.isfinite(float(onset["edof"])):
        notes.append(f"edof = {float(onset['edof']):.2f}")
    if onset.get("r2") is not None and np.isfinite(float(onset["r2"])):
        notes.append(f"R² = {float(onset['r2']):.3f}")
    if onset.get("direction"):
        notes.append(f"方向 = {onset['direction']}")
    if footer:
        notes.extend(list(footer))
    for i, t in enumerate(notes):
        c.text(L + pw, T + ph + 52 + 15 * i, t, size=10, color=_MUTED, anchor="end")
    return c.to_svg()
