#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""reference_sumstats.py —— 原生层（biostat-vcf sumstats）的**同口径** Python 参考实现。

存在意义：对拍需要一个「独立于被测对象」的裁判。但这个裁判必须
**与 Rust 实现同口径**，否则比出来的差异是口径差异、不是实现差异。

⚠️ 口径声明（极其重要，不可省略，否则数字不可比）
--------------------------------------------------
本参考实现复刻的是 `biostat-vcf sumstats` 的 **Wald 口径**：

    λGC = mean( χ² ) / 0.4549364231195727 ,   χ² = (β/se)²

注意它与本 skill 的 `scripts/qc_gwas.py` **不是同一个口径**：

    qc_gwas.py : λGC = median( χ²(p) ) / 0.4549364231195727 ,  χ² 由 p 值反推

两者在真实数据上**数值不同**，都合法，但**绝不可混用或相互替换**。
`native_accel.sumstats()` 返回的 `implementation` 字段就是为了让下游
能区分数字来自哪条口径 —— 混用会让「可复现」失效（P21）。

本文件只依赖标准库，任何裸 Python 都能跑。
"""

from __future__ import annotations

import math

THETA = 0.4549364231195727               # χ²₁ 的理论中位数
MAF_EDGES = [0.0, 0.01, 0.05, 0.2, 0.4, 0.51]   # 右开区间

_AF_ALIASES = {"af", "maf", "allele_frequency", "freq", "afreq", "eaf",
               "af_alt", "alt_af", "alt_freq", "freq_alt",
               "alt_allele_freq", "effect_allele_frequency"}
_CHROM_ALIASES = {"chr", "chrom", "chromosome", "#chrom"}
_POS_ALIASES = {"pos", "position", "bp", "base_pair_position"}
_BETA_ALIASES = {"beta", "b", "effect", "es", "effect_size"}
_SE_ALIASES = {"se", "sebeta", "std_err", "standard_error"}
_P_ALIASES = {"p", "pval", "p_value", "pvalue", "pvalue_mantissa"}


def _norm_chrom(raw: str) -> int | None:
    text = raw.strip()
    if text[:3].lower() == "chr":
        text = text[3:]
    if not text:
        return None
    up = text.upper()
    if up == "X":
        return 23
    if up == "Y":
        return 24
    if up in ("MT", "M"):
        return 25
    try:
        n = int(up)
    except ValueError:
        return None
    return n if 0 <= n <= 25 else None


def _resolve(header: list[str]) -> dict:
    lowered = [h.strip().lstrip("#").lower() for h in header]
    out = {}
    for key, aliases in (("chrom", _CHROM_ALIASES), ("pos", _POS_ALIASES),
                         ("af", _AF_ALIASES), ("beta", _BETA_ALIASES),
                         ("se", _SE_ALIASES), ("p", _P_ALIASES)):
        for i, name in enumerate(lowered):
            if name in aliases:
                out[key] = i
                break
    return out


def _to_float(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def reference(path: str) -> dict:
    """对 `path`（.tsv 或 .tsv.gz）跑同口径汇总，返回与 Rust 对齐的字典。"""
    import gzip

    opener = gzip.open if path.endswith(".gz") else open
    bins_n = [0] * (len(MAF_EDGES) - 1)
    bins_chi2 = [0.0] * (len(MAF_EDGES) - 1)
    issues = {"nonfinite_p": 0, "p_out_of_range": 0, "nonfinite_beta": 0,
              "nonpositive_se": 0, "missing_af": 0, "duplicate_keys": 0,
              "missing_required": 0}
    rows_read = rows_used = 0
    seen: set = set()

    with opener(path, "rt", encoding="utf-8-sig", newline="") as fh:
        header = None
        for line in fh:
            line = line.rstrip("\n")
            if header is None:
                stripped = line.lstrip()
                # 与 Rust 同规则：`##` 是注释；单 `#` 行只有解析出已知列名才是表头
                if not stripped:
                    continue
                if stripped.startswith("##"):
                    continue
                candidate = line.split("\t")
                col = _resolve(candidate)
                if stripped.startswith("#") and not col:
                    continue
                header = candidate
                if len([k for k in ("chrom", "pos", "af", "beta", "se", "p")
                        if k in col]) < 6:
                    return {"rows_read": 0, "rows_used": 0, "lambda_gc": None,
                            "issues": issues, "missing_columns": True}
                continue
            if not line.strip():
                continue
            rows_read += 1
            f = line.split("\t")

            def get(key):
                i = col.get(key)
                return f[i] if i is not None and i < len(f) else None

            chrom = _norm_chrom(get("chrom") or "")
            try:
                pos = int((get("pos") or "").strip())
            except ValueError:
                pos = None
            if chrom is None or pos is None:
                issues["missing_required"] += 1
                continue

            key = (chrom, pos)
            if key in seen:
                issues["duplicate_keys"] += 1
            else:
                seen.add(key)

            p = _to_float(get("p"))
            if p is None or not math.isfinite(p):
                issues["nonfinite_p"] += 1
                continue
            if not (0.0 <= p <= 1.0):
                issues["p_out_of_range"] += 1
                continue

            beta = _to_float(get("beta"))
            if beta is None or not math.isfinite(beta):
                issues["nonfinite_beta"] += 1
                continue

            se = _to_float(get("se"))
            if se is None or not math.isfinite(se) or se <= 0.0:
                issues["nonpositive_se"] += 1
                continue

            af = _to_float(get("af"))
            if af is None or not math.isfinite(af):
                issues["missing_af"] += 1
                continue
            # 与 Rust 同口径：先折算 MAF = min(f, 1-f)，越界单独计数
            if not (0.0 <= af <= 1.0):
                issues["missing_af"] += 1
                continue
            maf = min(af, 1.0 - af)

            bin_index = None
            for i in range(len(MAF_EDGES) - 1):
                if MAF_EDGES[i] <= maf < MAF_EDGES[i + 1]:
                    bin_index = i
                    break
            if bin_index is None:
                issues["missing_af"] += 1
                continue

            z = beta / se
            bins_n[bin_index] += 1
            bins_chi2[bin_index] += z * z
            rows_used += 1

    lam = None
    if rows_used:
        lam = (sum(bins_chi2) / rows_used) / THETA
    return {
        "rows_read": rows_read,
        "rows_used": rows_used,
        "lambda_gc": lam,
        "maf_binned": [
            {"af_low": MAF_EDGES[i], "af_high": MAF_EDGES[i + 1],
             "n": bins_n[i],
             "lambda_gc": (bins_chi2[i] / bins_n[i] / THETA) if bins_n[i] else None}
            for i in range(len(bins_n))
        ],
        "issues": issues,
        "truncated": False,
        "implementation": "python:reference-sumstats/wald",
    }


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(reference(sys.argv[1]), ensure_ascii=False, indent=2))
