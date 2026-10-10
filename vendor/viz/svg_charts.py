# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/viz/charts.py
# 许可: 自建/内部
# 说明: 纯 SVG 图表（无 matplotlib）
# 源 sha256: 3f9ac53a98179925
# VENDORED-FILE-END
"""程序化 SVG 图表库。

全部图形由代码直接绘制为矢量，不使用任何图像生成模型，不依赖 matplotlib。
这样做的三个理由：结果可复现（同一数据必得同一图）、文件极小、离线可看。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

INK = "#2C2C2A"
MUTED = "#5F5E5A"
GRID = "#D3D1C7"
BLUE = "#378ADD"
BLUE_FILL = "#E6F1FB"
RED = "#A32D2D"
GREEN = "#639922"
AMBER = "#BA7517"
TEAL = "#1D9E75"


class Canvas:
    """极简 SVG 画布。"""

    def __init__(self, width: int, height: int, title: str = "", desc: str = ""):
        self.w = width
        self.h = height
        self.title = title
        self.desc = desc
        self.body: List[str] = []

    def add(self, s: str) -> "Canvas":
        self.body.append(s)
        return self

    def text(self, x: float, y: float, s: str, size: int = 12,
             color: str = INK, anchor: str = "start", weight: int = 400,
             baseline: str = "auto") -> "Canvas":
        s = _esc(s)
        return self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="system-ui,-apple-system,Segoe UI,sans-serif" '
            + (f'dominant-baseline="{baseline}" ' if baseline != "auto" else "")
            + f'>{s}</text>')

    def line(self, x1: float, y1: float, x2: float, y2: float,
             color: str = GRID, sw: float = 1.0, dash: str = "") -> "Canvas":
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                        f'stroke="{color}" stroke-width="{sw}"{d}/>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none",
             stroke: str = GRID, sw: float = 0.5, rx: float = 0) -> "Canvas":
        return self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>')

    def circle(self, cx: float, cy: float, r: float, fill: str = BLUE,
               stroke: str = "none", sw: float = 0.5) -> "Canvas":
        return self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def polyline(self, pts: Sequence[Tuple[float, float]], color: str = BLUE,
                 sw: float = 1.8, dash: str = "", fill: str = "none") -> "Canvas":
        if not pts:
            return self
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return self.add(f'<polyline points="{d}" fill="{fill}" stroke="{color}" '
                        f'stroke-width="{sw}"{dash_attr} stroke-linejoin="round"/>')

    def path(self, d: str, color: str = BLUE, sw: float = 1.8,
             fill: str = "none", dash: str = "") -> "Canvas":
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return self.add(f'<path d="{d}" fill="{fill}" stroke="{color}" '
                        f'stroke-width="{sw}"{dash_attr}/>')

    def to_svg(self) -> str:
        head = (f'<svg viewBox="0 0 {self.w} {self.h}" width="100%" '
                f'role="img" xmlns="http://www.w3.org/2000/svg">')
        t = (f"<title>{_esc(self.title)}</title><desc>{_esc(self.desc)}</desc>") if self.title else ""
        return head + t + "".join(self.body) + "</svg>"


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [0.0, 1.0]
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


# ------------------------------------------------------------------ 森林图

def forest_plot(items: Sequence[dict], title: str = "森林图",
                xlabel: str = "效应量 (95% CI)", log_scale: bool = True,
                null_value: float = 1.0 if True else 0.0) -> str:
    """items: [{label, estimate, low, high}]，log_scale 时按对数轴绘制（适用于 OR/HR）。"""
    nv = 1.0 if log_scale else 0.0
    rows = []
    for it in items:
        e, lo, hi = it["estimate"], it["low"], it["high"]
        if log_scale and e > 0 and lo > 0 and hi > 0:
            e, lo, hi = np.log(e), np.log(lo), np.log(hi)
        rows.append((it["label"], e, lo, hi))

    finite = [v for r in rows for v in r[1:] if np.isfinite(v)]
    if not finite:
        return Canvas(600, 120, title).text(20, 40, "无可绘制的效应量").to_svg()

    lo_all, hi_all = min(finite), max(finite)
    pad = (hi_all - lo_all) * 0.15 or 0.5
    xmin, xmax = lo_all - pad, hi_all + pad
    if np.isfinite(null_value if not log_scale else np.log(null_value)):
        nvp = np.log(null_value) if log_scale else null_value
        xmin, xmax = min(xmin, nvp), max(xmax, nvp)

    W = 660
    label_w = 210
    plot_x0, plot_x1 = label_w + 20, W - 90
    top, rowh = 56, 30
    H = top + rowh * len(rows) + 60

    c = Canvas(W, H, title, "效应量与 95% 置信区间")
    c.text(20, 26, title, size=14, weight=500)
    c.text(plot_x0, top + rowh * len(rows) + 34, xlabel, size=11, color=MUTED)

    def sx(v):
        return plot_x0 + (v - xmin) / max(xmax - xmin, 1e-9) * (plot_x1 - plot_x0)

    for tk in _nice_ticks(xmin, xmax, 4):
        c.line(sx(tk), top - 8, sx(tk), top + rowh * len(rows) + 6)
        lab = "%.2f" % (np.exp(tk) if log_scale else tk)
        c.text(sx(tk), top + rowh * len(rows) + 22, lab, size=10, color=MUTED, anchor="middle")

    nvp = np.log(null_value) if log_scale else null_value
    c.line(sx(nvp), top - 8, sx(nvp), top + rowh * len(rows) + 6,
           color=RED, sw=1.2, dash="3 3")

    for i, (lab, e, lo, hi) in enumerate(rows):
        y = top + i * rowh + rowh // 2
        c.text(20, y + 4, lab[:26], size=12)
        if np.isfinite(lo) and np.isfinite(hi):
            c.line(sx(lo), y, sx(hi), y, color=BLUE, sw=1.6)
            c.line(sx(lo), y - 4, sx(lo), y + 4, color=BLUE, sw=1.6)
            c.line(sx(hi), y - 4, sx(hi), y + 4, color=BLUE, sw=1.6)
        if np.isfinite(e):
            c.rect(sx(e) - 3.5, y - 3.5, 7, 7, fill=BLUE, stroke="none")
        val = "%.3f (%.3f–%.3f)" % (
            np.exp(e) if log_scale else e, np.exp(lo) if log_scale else lo,
            np.exp(hi) if log_scale else hi)
        c.text(plot_x1 + 8, y + 4, val, size=10, color=MUTED)
    return c.to_svg()


# ------------------------------------------------------------------ KM 曲线

def km_plot(curves: Dict[str, dict], title: str = "Kaplan-Meier 生存曲线",
            xlabel: str = "随访时间", ylabel: str = "生存概率") -> str:
    """curves: {name: {"time": [...], "survival": [...], "ci_low":..., "ci_high":...}}"""
    W, H = 660, 380
    x0, x1, y0, y1 = 80, W - 40, H - 70, 60
    c = Canvas(W, H, title, "分组生存曲线与置信带")
    c.text(20, 30, title, size=14, weight=500)

    allt = [max(v["time"]) for v in curves.values() if len(v["time"])]
    tmax = max(allt) if allt else 1.0
    colors = [BLUE, RED, GREEN, AMBER, TEAL]

    for tk in _nice_ticks(0, tmax, 5):
        x = x0 + tk / tmax * (x1 - x0)
        c.line(x, y0, x, y1, color=GRID, sw=0.5)
        c.text(x, y0 + 18, "%.1f" % tk, size=10, color=MUTED, anchor="middle")
    for tk in _nice_ticks(0, 1, 5):
        y = y0 - tk * (y0 - y1)
        c.line(x0, y, x1, y, color=GRID, sw=0.5)
        c.text(x0 - 8, y + 4, "%.1f" % tk, size=10, color=MUTED, anchor="end")

    c.line(x0, y0, x1, y0, color=INK, sw=1.0)
    c.line(x0, y0, x0, y1, color=INK, sw=1.0)
    c.text((x0 + x1) / 2, H - 24, xlabel, size=11, color=MUTED, anchor="middle")
    c.text(20, (y0 + y1) / 2, ylabel, size=11, color=MUTED, anchor="middle",
           baseline="central")

    for i, (name, v) in enumerate(curves.items()):
        col = colors[i % len(colors)]
        t = np.asarray(v["time"], float)
        s = np.asarray(v["survival"], float)
        pts = [(x0 + tv / tmax * (x1 - x0), y0 - sv * (y0 - y1)) for tv, sv in zip(t, s)]
        c.polyline(pts, color=col, sw=1.8)
        if "ci_low" in v and "ci_high" in v:
            lo = np.asarray(v["ci_low"], float)
            hi = np.asarray(v["ci_high"], float)
            up = [(x0 + tv / tmax * (x1 - x0), y0 - hv * (y0 - y1))
                  for tv, hv in zip(t, hi)]
            dn = [(x0 + tv / tmax * (x1 - x0), y0 - lv * (y0 - y1))
                  for tv, lv in zip(t, lo)]
            c.polyline(up, color=col, sw=0.8, dash="3 3")
            c.polyline(dn, color=col, sw=0.8, dash="3 3")
        c.rect(x1 - 150, y1 + i * 20, 12, 8, fill=col, stroke="none")
        c.text(x1 - 132, y1 + i * 20 + 8, name, size=11)
    return c.to_svg()


# ------------------------------------------------------------------ ROC / 校准 / DCA

def roc_plot(roc: dict, auc: Optional[float] = None,
             title: str = "ROC 曲线") -> str:
    W, H = 420, 360
    x0, y0 = 70, H - 60
    size = W - 110
    c = Canvas(W, H, title, "受试者工作特征曲线")
    c.text(20, 26, title, size=14, weight=500)
    c.rect(x0, y0 - size, size, size, fill="none", stroke=GRID, sw=0.5)
    c.path(f"M{x0} {y0} L{x0 + size} {y0 - size}", color=MUTED, sw=1.0, dash="4 4")
    fpr = np.asarray(roc.get("fpr", [0, 1]), float)
    tpr = np.asarray(roc.get("tpr", [0, 1]), float)
    pts = [(x0 + f * size, y0 - t * size) for f, t in zip(fpr, tpr)]
    c.polyline(pts, color=BLUE, sw=2.0)
    for tk in _nice_ticks(0, 1, 5):
        c.text(x0 + tk * size, y0 + 16, "%.1f" % tk, size=10, color=MUTED, anchor="middle")
        c.text(x0 - 8, y0 - tk * size + 4, "%.1f" % tk, size=10, color=MUTED, anchor="end")
    c.text(x0 + size / 2, H - 18, "假阳性率 (1 − 特异度)", size=10, color=MUTED,
           anchor="middle")
    if auc is not None and np.isfinite(auc):
        c.text(x0 + 12, y0 - size + 22, "AUROC = %.3f" % auc, size=12, weight=500)
    return c.to_svg()


def calibration_plot(bins: Sequence[dict], slope: Optional[float] = None,
                     title: str = "校准曲线") -> str:
    W, H = 420, 360
    x0, y0 = 70, H - 60
    size = W - 110
    c = Canvas(W, H, title, "预测概率分箱与实测事件率")
    c.text(20, 26, title, size=14, weight=500)
    c.rect(x0, y0 - size, size, size, fill="none", stroke=GRID, sw=0.5)
    c.path(f"M{x0} {y0} L{x0 + size} {y0 - size}", color=MUTED, sw=1.0, dash="4 4")
    xs, ys, ws = [], [], []
    for b in bins:
        xs.append(b["mean_predicted"])
        ys.append(b["observed_rate"])
        ws.append(b.get("n", 1))
    if xs:
        pts = [(x0 + x * size, y0 - y * size) for x, y in zip(xs, ys)]
        for (px, py), w in zip(pts, ws):
            r = min(3 + np.sqrt(w) * 0.6, 9)
            c.circle(px, py, r, fill=BLUE, stroke="none")
        c.polyline(pts, color=BLUE, sw=1.6)
    for tk in _nice_ticks(0, 1, 5):
        c.text(x0 + tk * size, y0 + 16, "%.1f" % tk, size=10, color=MUTED, anchor="middle")
        c.text(x0 - 8, y0 - tk * size + 4, "%.1f" % tk, size=10, color=MUTED, anchor="end")
    c.text(x0 + size / 2, H - 18, "预测概率", size=10, color=MUTED, anchor="middle")
    if slope is not None and np.isfinite(slope):
        c.text(x0 + 12, y0 - size + 22, "校准斜率 = %.3f（1 为理想）" % slope,
               size=11, weight=500)
    return c.to_svg()


def dca_plot(curve: Sequence[dict], title: str = "决策曲线分析") -> str:
    W, H = 420, 360
    x0, y0 = 70, H - 60
    pw, ph = W - 110, H - 130
    c = Canvas(W, H, title, "不同阈值概率下的净获益")
    c.text(20, 26, title, size=14, weight=500)
    c.rect(x0, y0 - ph, pw, ph, fill="none", stroke=GRID, sw=0.5)
    nb = [d["net_benefit"] for d in curve]
    ta = [d["treat_all"] for d in curve]
    lo, hi = min(nb + ta + [0.0]), max(nb + ta + [0.0])
    rng = hi - lo or 1.0

    def sy(v):
        return y0 - (v - lo) / rng * ph

    pts = [(x0 + i / max(len(curve) - 1, 1) * pw, sy(v)) for i, v in enumerate(nb)]
    pts_all = [(x0 + i / max(len(curve) - 1, 1) * pw, sy(v)) for i, v in enumerate(ta)]
    c.polyline(pts_all, color=MUTED, sw=1.2, dash="4 4")
    c.polyline([(x0, y0), (x0 + pw, y0)], color=AMBER, sw=1.2)
    c.polyline(pts, color=BLUE, sw=2.0)
    c.text(x0 + 12, y0 - ph + 20, "蓝：模型", size=11, color=BLUE, weight=500)
    c.text(x0 + 12, y0 - ph + 36, "灰：全部治疗", size=11, color=MUTED)
    c.text(x0 + 12, y0 - ph + 52, "橙：全部不治疗", size=11, color=AMBER)
    c.text(x0 + pw / 2, H - 18, "阈值概率", size=10, color=MUTED, anchor="middle")
    return c.to_svg()


# ------------------------------------------------------------------ 火山图 / 热图

def volcano(df, logfc_col: str = "log2FC", q_col: str = "q",
            title: str = "火山图", lfc: float = 1.0, alpha: float = 0.05) -> str:
    W, H = 660, 400
    x0, y0 = 90, H - 60
    pw, ph = W - 150, H - 120
    c = Canvas(W, H, title, "效应量与显著性")
    c.text(20, 26, title, size=14, weight=500)

    fc = df[logfc_col].values.astype(float)
    q = df[q_col].values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = -np.log10(np.clip(q, 1e-300, 1.0))
    ok = np.isfinite(fc) & np.isfinite(lp)
    fc, lp = fc[ok], lp[ok]
    if fc.size == 0:
        return c.text(20, 60, "无可绘制数据").to_svg()

    xr = max(np.nanmax(np.abs(fc)) * 1.1, lfc * 1.5)
    yr = max(np.nanmax(lp) * 1.1, 2.0)

    def sx(v):
        return x0 + (v + xr) / (2 * xr) * pw

    def sy(v):
        return y0 - v / yr * ph

    c.line(sx(-lfc), y0 - ph, sx(-lfc), y0, color=AMBER, sw=0.8, dash="3 3")
    c.line(sx(lfc), y0 - ph, sx(lfc), y0, color=AMBER, sw=0.8, dash="3 3")
    thr = -np.log10(alpha)
    c.line(x0, sy(thr), x0 + pw, sy(thr), color=AMBER, sw=0.8, dash="3 3")
    c.line(x0, y0, x0 + pw, y0, color=INK, sw=1.0)

    up = (fc >= lfc) & (lp >= thr)
    dn = (fc <= -lfc) & (lp >= thr)
    c.add("".join(
        f'<circle cx="{sx(v):.1f}" cy="{sy(w):.1f}" r="2.6" fill="{GREEN}" fill-opacity="0.75"/>'
        for v, w in zip(fc[up], lp[up])))
    c.add("".join(
        f'<circle cx="{sx(v):.1f}" cy="{sy(w):.1f}" r="2.6" fill="{RED}" fill-opacity="0.75"/>'
        for v, w in zip(fc[dn], lp[dn])))
    ns = ~(up | dn)
    c.add("".join(
        f'<circle cx="{sx(v):.1f}" cy="{sy(w):.1f}" r="2.2" fill="{MUTED}" fill-opacity="0.45"/>'
        for v, w in zip(fc[ns], lp[ns])))

    for tk in _nice_ticks(-xr, xr, 4):
        c.text(sx(tk), y0 + 16, "%.1f" % tk, size=10, color=MUTED, anchor="middle")
    for tk in _nice_ticks(0, yr, 4):
        c.text(x0 - 8, sy(tk) + 4, "%.0f" % tk, size=10, color=MUTED, anchor="end")
    c.text(x0 + pw / 2, H - 20, "log2 倍数变化", size=10, color=MUTED, anchor="middle")
    c.text(20, y0 - ph / 2, "−log10(q)", size=10, color=MUTED, anchor="middle",
           baseline="central")
    return c.to_svg()


def heatmap(matrix: np.ndarray, row_labels: Optional[Sequence[str]] = None,
            col_labels: Optional[Sequence[str]] = None,
            title: str = "热图") -> str:
    M = np.asarray(matrix, float)
    if M.ndim != 2 or M.size == 0:
        return Canvas(400, 100).text(20, 40, "空矩阵").to_svg()
    n, p = M.shape
    cell = 16
    W = min(660, 120 + p * cell + 20)
    H = min(560, 80 + n * cell + 20)
    c = Canvas(W, H, title, "数值矩阵热图")
    c.text(20, 26, title, size=14, weight=500)

    lo, hi = np.nanmin(M), np.nanmax(M)
    rng = hi - lo or 1.0
    for i in range(n):
        for j in range(p):
            v = M[i, j]
            if not np.isfinite(v):
                col = "#F1EFE8"
            else:
                t = (v - lo) / rng
                # 蓝→白→红 发散配色
                if t < 0.5:
                    a = (0.5 - t) * 2
                    col = "#%02X%02X%02X" % (int(230 - 100 * a), int(241 - 80 * a), int(251 - 40 * a))
                else:
                    a = (t - 0.5) * 2
                    col = "#%02X%02X%02X" % (int(252), int(235 - 120 * a), int(235 - 150 * a))
            c.rect(110 + j * cell, 50 + i * cell, cell - 1, cell - 1, fill=col, stroke="none")
    if row_labels:
        for i, lab in enumerate(row_labels[:n]):
            c.text(104, 50 + i * cell + cell / 2 + 4, str(lab)[:16], size=9,
                   color=MUTED, anchor="end")
    c.text(110, H - 14, "低 ← 数值 → 高", size=10, color=MUTED)
    return c.to_svg()
