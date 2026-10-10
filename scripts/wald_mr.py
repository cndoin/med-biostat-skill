#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wald_mr.py —— 单工具 Wald ratio MR + F 统计量（Tier0，纯标准库）

**药物靶点 MR 的主力方法**。当 cis 工具变量每基因每组织只有一个 lead SNP 时，
唯一合法的方法是 Wald ratio —— PROTOCOL 明确：

  > 一个 SNP 采用 Wald 比率，**不能做 MR-Egger、异质性或留一法**。

本脚本据此**硬拒绝**对单工具调用多工具方法，避免方法误用（审稿人必查）。

公式（Burgess 2013；delta 法）：
    β_Wald = β_outcome / β_exposure
    se(β)  ≈ sqrt( se_out²/β_exp² + β_out²·se_exp²/β_exp⁴ )
    F      = (β_exp / se_exp)²          # 弱工具判据，须 > 10
    p      = 2·Φ(−|β/se|)

用法：
    python wald_mr.py --instruments mr_input.tsv --json --out logs/wald_mr.json
    # instruments 表需含：snp, beta_exposure, se_exposure, beta_outcome, se_outcome
    #                  可选：gene, tissue（用于分层与多重检验）
退出码：0 成功 / 2 输入错误 / 3 口径缺失
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ALIAS = {
    "snp": ["snp", "rsid", "variant", "v", "id"],
    "gene": ["gene", "gene_name", "symbol"],
    "tissue": ["tissue", "tissue_id"],
    "bx": ["beta_exposure", "bx", "beta_exp", "b_exposure"],
    "sx": ["se_exposure", "sx", "se_exp", "sebeta_exposure"],
    "by": ["beta_outcome", "by", "beta_out", "b_outcome"],
    "sy": ["se_outcome", "sy", "se_out", "sebeta_outcome"],
}


def resolve(header):
    low = {}
    for i, h in enumerate(header):
        low[h.lower()] = i
        low[h.lower().lstrip("#")] = i
    out = {}
    for k, cands in ALIAS.items():
        for c in cands:
            if c.lower() in low:
                out[k] = low[c.lower()]
                break
    return out


def _safe_exp(x: float | None):
    """exp 的溢出/非有限防护。

    曾经的写法是 `math.exp(beta) if abs(beta) < 700`，但真正溢出的指数是
    `beta ± 1.96·se`——β=699、se=0.1 时守卫放行，实际指数 699.196 仍会
    math.exp 抛 OverflowError，整个脚本挂掉，连可用的行一起丢。
    现在按**实际指数**判定：超过 IEEE754 双精度的 exp 上限（约 709.78）
    就返回 None（=无法表示），而不是崩。
    """
    if x is None or not math.isfinite(x):
        return None
    if x > 709.0:          # exp(709)≈8.2e307，接近双精度上限
        return float("inf") if x > 709.78 else None
    if x < -745.0:         # exp 下溢
        return 0.0
    try:
        return math.exp(x)
    except OverflowError:
        return None


def wald(bx: float, sx: float, by: float, sy: float) -> dict:
    """单工具 Wald ratio 及其 delta 法标准误。"""
    if bx is None or sx is None or by is None or sy is None:
        return {"beta": None, "se": None, "p": None, "F": None, "reason": "缺失值"}
    if abs(bx) < 1e-12:
        return {"beta": None, "se": None, "p": None, "F": None,
                "reason": "暴露效应为 0，比值不可估"}
    # se_exposure 必须是严格正数。se=0 意味着 β_exp 没有任何噪声信息，
    # 此时 (β/se)² = inf 会把该行判成「F 极大、非弱工具」，直接给出
    # 虚假显著性；负 se 更是非法输入（平方后仍为正，会被静默当成有效值）。
    if not math.isfinite(sx) or sx <= 0:
        return {"beta": None, "se": None, "p": None, "F": None,
                "reason": f"se_exposure 非法（={sx}，须为有限正数）"}
    if not math.isfinite(sy) or sy <= 0:
        return {"beta": None, "se": None, "p": None, "F": None,
                "reason": f"se_outcome 非法（={sy}，须为有限正数）"}
    beta = by / bx
    var = (sy ** 2) / (bx ** 2) + (by ** 2) * (sx ** 2) / (bx ** 4)
    se = math.sqrt(var) if var > 0 else None
    f = (bx / sx) ** 2
    # 双侧 p = 2·Φ(−|z|) = erfc(|z|/√2)。用 erfc 而非 1−cdf：后者在 |z| 大时
    # 会因浮点抵消丢精度（statistics.NormalDist 没有 sf()，别照搬 scipy 的 API）。
    p = (math.erfc(abs(beta / se) / math.sqrt(2))
         if se and se > 0 else None)
    return {"beta": beta, "se": se, "p": p, "F": f, "var": var,
            "or": _safe_exp(beta),
            "or_ci_low": _safe_exp(beta - 1.96 * se) if se and se > 0 else None,
            "or_ci_high": _safe_exp(beta + 1.96 * se) if se and se > 0 else None,
            "reason": ""}


def main() -> int:
    ap = argparse.ArgumentParser(description="单工具 Wald ratio MR")
    ap.add_argument("--instruments", required=True,
                    help="含 beta_exposure/se_exposure/beta_outcome/se_outcome 的表")
    ap.add_argument("--f-min", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not Path(args.instruments).exists():
        return C.fail(f"文件不存在: {args.instruments}", code=C.EXIT_INPUT, as_json=args.json)

    rows = []
    try:
        with C.open_text(args.instruments) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            col = resolve(header)
            need = ("bx", "sx", "by", "sy")
            if not all(k in col for k in need):
                return C.fail(
                    f"缺少必需列（beta_exposure/se_exposure/beta_outcome/se_outcome）；"
                    f"实际表头={header}", code=C.EXIT_INPUT, as_json=args.json)
            for line in fh:
                f = line.rstrip("\n").split("\t")

                def g(key, cast=float):
                    if key not in col:
                        return None
                    try:
                        return cast(f[col[key]])
                    except (ValueError, IndexError):
                        return None

                bx, sx, by, sy = g("bx"), g("sx"), g("by"), g("sy")
                r = wald(bx, sx, by, sy)
                r.update({
                    "snp": f[col["snp"]] if "snp" in col else "",
                    "gene": f[col["gene"]] if "gene" in col else "",
                    "tissue": f[col["tissue"]] if "tissue" in col else "",
                    "beta_exposure": bx, "se_exposure": sx,
                    "beta_outcome": by, "se_outcome": sy,
                })
                rows.append(r)
    except (OSError, ValueError) as e:
        return C.fail(f"读取失败：{e}", code=C.EXIT_INPUT, as_json=args.json)

    for r in rows:
        r["weak_instrument"] = (r["F"] is not None and r["F"] < args.f_min)
        r["significant"] = (r["p"] is not None and r["p"] < args.alpha)

    n_ok = sum(1 for r in rows if r["beta"] is not None and not r["weak_instrument"])
    n_weak = sum(1 for r in rows if r["weak_instrument"])
    n_unestimable = sum(1 for r in rows if r["beta"] is None)

    report = {
        "ok": True,
        "summary": f"{len(rows)} 个工具：可估 {n_ok}，弱工具(F<{args.f_min:g}) {n_weak}，"
                   f"不可估 {n_unestimable}",
        "caliber": {
            "method": "单工具 Wald ratio",
            "se_method": "delta 法（含 β_out²·se_exp²/β_exp⁴ 项）",
            "f_statistic": "(β_exp/se_exp)²，须 > " + f"{args.f_min:g}",
            "multi_instrument_methods": "REJECTED",
            "note": ("单工具**不得**使用 MR-Egger / 异质性检验 / 留一法 —— "
                     "这些方法在单工具下无自由度（PROTOCOL 硬约束）。"
                     "多工具场景请改用 vendor/engines/mr_coloc.mr()"),
        },
        "n_instruments": len(rows),
        "n_estimable": n_ok,
        "n_weak": n_weak,
        "n_unestimable": n_unestimable,
        "results": rows,
    }
    if args.out:
        C.write_json(args.out, report)
    for r in rows:
        if r["beta"] is None:
            C.log(f"  [不可估] {r.get('gene','')} {r.get('tissue','')} — {r['reason']}")
        else:
            flag = " [弱工具]" if r["weak_instrument"] else ""
            C.log(f"  {r.get('gene',''):<12} {r.get('tissue',''):<18} "
                  f"β={r['beta']:+.4f} se={r['se']:.4f} p={r['p']:.3g} F={r['F']:.1f}{flag}")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
