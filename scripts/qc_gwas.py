#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""qc_gwas.py —— S2 全量质控重算（口径先于数值）

铁律：**每个指标输出前必须先声明口径**。缺口径默认报错（exit 3），
只有显式 --allow-missing-caliber 才降级为 UNSPECIFIED。

计算项：
  * 变异数、可用变异数（按 MAF/INFO 过滤后）
  * **λGC（必须指定 MAF 分档）** —— λGC = median(χ²₁) / 0.4549364
  * 全基因组显著 **变异数**（P<5e-8；注意 ≠ 独立位点数，见 loci_count.py）
  * 技术性健全性：AF 越界、sebeta 非正、p 越界、重复变异

用法：
    python qc_gwas.py --input raw/x.gz --bucket maf_0.01_0.5 --json
    python qc_gwas.py --input raw/x.gz --maf-min 0.01 --maf-max 0.5 --json
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

# 原生加速层（可选）：tools/native_accel.py 只依赖标准库，导入失败即视为不可用。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
try:  # noqa: E402
    import native_accel as _NA
except Exception:  # noqa: BLE001
    _NA = None

CHI2_DF1_MEDIAN = 0.4549364231195732  # χ²₁ 的中位数，λGC 的分母
NORM = NormalDist()


def chi2_df1(p: float) -> float:
    """p 值 → χ²₁；用标准库 NormalDist.inv_cdf 实现，无需 scipy。

    inv_cdf 要求入参严格落在 (0,1) 开区间；p 极小时 1-p/2 会舍入到 1.0，
    必须回退到有限上限，否则抛 StatisticsError（实测 p=1e-300 会崩）。
    """
    p = min(max(float(p), 1e-300), 1.0)
    q = 1.0 - p / 2.0
    q = min(max(q, 1e-15), 1.0 - 1e-15)
    z = NORM.inv_cdf(q)
    return z * z


ALIAS = {
    "chrom": ["#chrom", "chrom", "chr", "chromosome"],
    "pos": ["pos", "position", "bp"],
    "p": ["pval", "p", "p_value"],
    "sebeta": ["sebeta", "se", "standard_error"],
    "af": ["af_alt", "af", "maf", "eaf"],
    "info": ["info", "rsq", "r2"],
}


def resolve(header):
    low = {h.lower(): h for h in header}
    out = {}
    for k, cands in ALIAS.items():
        for c in cands:
            if c.lower() in low:
                out[k] = header.index(low[c.lower()]); break
    return out


def _native_engine(args, caliber, maf_min, maf_max):
    """原生加速层。已处理则返回退出码；不可用/降级返回 None（回退 Tier0）。

    ⚠️ 口径纪律：原生层算的是 **Wald 均值口径** λGC
    （mean((β/se)²)/0.4549364），与 Tier0 的 **p 值中位数口径** **不是同一个数**。
    因此本函数：
      · 只在用户显式 `--native` 时启用（不静默替换）；
      · 结果必须带 `lambda_caliber` 与 `engine` 两个可核字段；
      · 质控未通过时**原样透传**原生层的诊断，不重算、不掩盖。
    """
    if _NA is None:
        C.log("[WARN] 原生加速层模块不可导入，回退 Tier0")
        return None
    if not _NA.available():
        C.log("[WARN] 原生二进制不存在（vendor/native），回退 Tier0 纯标准库路径")
        return None

    payload, code = _NA.sumstats(args.input, threads=args.threads,
                                 max_rows=args.max_variants)
    if payload.get("status") == "SKIP":
        C.log(f"[WARN] {payload.get('reason')}，回退 Tier0")
        return None

    report = payload.get("report") or {}
    if payload.get("status") != "OK":
        status = payload.get("status")
        return C.fail(
            f"原生层质控未通过（{status}）："
            f"{(payload.get('error') or {}).get('message', status)}",
            code=C.EXIT_INPUT if status in ("BLOCKED", "CONFIG") else C.EXIT_ERROR,
            as_json=args.json,
            native_status=status,
            native_exit_code=payload.get("exit_code"),
            diagnostics=payload.get("diagnostics"),
        )

    lam = report.get("lambda_gc")
    out = {
        "ok": True,
        "engine": "native:biostat-vcf/sumstats",
        "summary": f"变异 {report.get('rows_read', 0):,}／可用 "
                   f"{report.get('rows_used', 0):,}；λGC="
                   f"{None if lam is None else round(lam, 6)}"
                   f"（原生层 · {payload.get('lambda_caliber')}）",
        "caliber": {
            **caliber,
            "lambda_gc_definition": "mean((β/se)²)/0.4549364231195727（Wald 均值口径，原生层）",
            "lambda_gc_maf_bucket": caliber["bucket"],
            "lambda_gc_caliber_tag": payload.get("lambda_caliber"),
            "note": "与 Tier0（p 值中位数口径）不是同一个数，不可混用或相互替换",
        },
        "n_variants": report.get("rows_read"),
        "n_variants_usable": report.get("rows_used"),
        "lambda_gc": lam,
        "maf_binned": report.get("maf_binned"),
        "sanity_issues": report.get("issues"),
        "diagnostics": payload.get("diagnostics"),
        "truncated": report.get("truncated", False),
        "implementation": payload.get("implementation"),
    }
    if args.out:
        C.write_json(args.out, out)
    C.log(f"  原生层：变异 {report.get('rows_read', 0):,}／可用 "
          f"{report.get('rows_used', 0):,}／λGC={lam}")
    return C.emit(out, as_json=args.json, code=C.EXIT_OK)


def main() -> int:
    ap = argparse.ArgumentParser(description="GWAS 质控重算（口径必填）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--bucket", default=None,
                    help="MAF 分档口径名（如 maf_0.01_0.5），或与下一组二选一")
    ap.add_argument("--maf-min", type=float, default=None)
    ap.add_argument("--maf-max", type=float, default=None)
    ap.add_argument("--info-min", type=float, default=None)
    ap.add_argument("--n-case", type=int, default=None)
    ap.add_argument("--n-control", type=int, default=None)
    ap.add_argument("--gw-threshold", type=float, default=5e-8)
    ap.add_argument("--allow-missing-caliber", action="store_true")
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--native", action="store_true",
                    help="优先走原生加速层（biostat-vcf sumstats，Rust）。"
                         "⚠️ 会切换 λGC 口径为 Wald 均值口径，见输出 lambda_caliber")
    ap.add_argument("--threads", type=int, default=None, help="原生层线程数")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # ---- 口径守卫：λGC 必须同报 MAF 分档 ----
    cfg = {"bucket": args.bucket}
    if not args.bucket and args.maf_min is not None and args.maf_max is not None:
        cfg["bucket"] = f"maf_{args.maf_min}_{args.maf_max}"
    try:
        caliber = C.require_caliber(
            {"bucket": cfg["bucket"],
             "gw_threshold": args.gw_threshold,
             "variant_unit": "variant（非独立位点）"},
            ["bucket", "gw_threshold", "variant_unit"],
            allow_missing=args.allow_missing_caliber)
    except C.CaliberError as e:
        return C.fail(f"口径缺失：{e.missing}（λGC 必须同报 MAF 分档）",
                      code=C.EXIT_CALIBER, as_json=args.json,
                      missing_caliber=e.missing)

    maf_min = args.maf_min if args.maf_min is not None else 0.0
    maf_max = args.maf_max if args.maf_max is not None else 0.5

    if not Path(args.input).exists():
        return C.fail(f"文件不存在: {args.input}", code=C.EXIT_INPUT, as_json=args.json)

    # ---- 原生加速层（显式 --native 才走；绝不静默替换口径） ----
    if args.native:
        native = _native_engine(args, caliber, maf_min, maf_max)
        if native is not None:
            return native

    header = None
    n = 0
    n_usable = 0
    chi2_sum = 0.0
    pvals_usable = []
    n_gw = 0
    issues = {"af_out_of_range": 0, "sebeta_nonpositive": 0,
              "p_out_of_range": 0, "p_not_finite": 0, "dup_keys": 0}
    seen = set()
    t0 = time.time()
    watch = C.ProgressWatch(timeout=180)

    with C.open_text(args.input) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = resolve(header)
        if "p" not in col:
            return C.fail(f"无法识别 p 值列；表头={header}",
                          code=C.EXIT_INPUT, as_json=args.json)
        for line in fh:
            n += 1
            if args.max_variants and n > args.max_variants:
                break
            parts = line.rstrip("\n").split("\t")
            try:
                p = float(parts[col["p"]])
            except (ValueError, IndexError):
                continue
            # NaN/Inf 必须显式拦下：float("NaN") 能解析成功，而
            # `p < 0 or p > 1` 对 NaN 恒为 False —— NaN 会静默混入 λGC，
            # 让 median(χ²) 整体变 NaN，且产出 `NaN` 字面量（RFC 8259 非法 JSON）。
            # 真实数据里 NA/NaN/. 是常见缺失写法，这个洞很容易踩到。
            if not math.isfinite(p):
                issues["p_not_finite"] += 1
                continue
            if p < 0 or p > 1:
                issues["p_out_of_range"] += 1
                continue
            if p < args.gw_threshold:
                n_gw += 1
            # AF 口径与过滤
            keep = True
            if "af" in col:
                try:
                    af = float(parts[col["af"]]); maf = C.as_maf(af)
                    if not (0 <= af <= 1):
                        issues["af_out_of_range"] += 1
                    if maf < maf_min or maf > maf_max:
                        keep = False
                except (ValueError, IndexError):
                    keep = False
            if "info" in col and args.info_min is not None:
                try:
                    if float(parts[col["info"]]) < args.info_min:
                        keep = False
                except (ValueError, IndexError):
                    keep = False
            if "sebeta" in col:
                try:
                    if float(parts[col["sebeta"]]) <= 0:
                        issues["sebeta_nonpositive"] += 1
                except (ValueError, IndexError):
                    pass
            # 重复键
            if "chrom" in col and "pos" in col:
                k = (parts[col["chrom"]], parts[col["pos"]])
                if k in seen:
                    issues["dup_keys"] += 1
                else:
                    seen.add(k)
            if keep:
                n_usable += 1
                pvals_usable.append(p)
            if n % 2_000_000 == 0:
                watch.tick(n)

    # λGC：只在 MAF 过滤后的集合上算
    lam = None
    if pvals_usable:
        chi2 = [chi2_df1(p) for p in pvals_usable]
        chi2.sort()
        med = chi2[len(chi2) // 2]
        lam = med / CHI2_DF1_MEDIAN

    truncated = bool(args.max_variants and n > args.max_variants)
    report = {
        "ok": True,
        "engine": "stdlib:qc_gwas.py",
        "summary": f"变异 {n:,}；可用 {n_usable:,}；λGC={lam if lam is None else round(lam,4)}"
                   f"（口径 {caliber['bucket']}）；P<{args.gw_threshold:g} 变异 {n_gw:,}",
        "caliber": {
            **caliber,
            "lambda_gc_definition": "median(χ²₁)/0.4549364，仅在 MAF 过滤后集合上计算",
            "lambda_gc_maf_bucket": caliber["bucket"],
            "n_case": args.n_case,
            "n_control": args.n_control,
            "info_min": args.info_min,
        },
        "n_variants": n,
        "n_variants_usable": n_usable,
        "lambda_gc": lam,
        "n_gw_significant_variants": n_gw,
        "unit_warning": "n_gw_significant_variants 是变异数，不等于独立位点数（P08）",
        "sanity_issues": issues,
        "truncated": truncated,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if args.out:
        C.write_json(args.out, report)
    if truncated:
        C.log("[WARN] 结果来自截断数据（--max-variants），不得作为最终结论（P02）")
    C.log(f"  变异 {n:,} / 可用 {n_usable:,} / λGC={lam}")
    for k, v in issues.items():
        if v:
            C.log(f"  [健全性] {k}={v}")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
