#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""loci_backbone.py —— S3① 位点骨架：可信集纯度过滤（红线 R6）

实证背景：存在 cs_size 达 21,398 的低纯度可信集（cs_avg_r2 极低），
不过滤会把上万个 SNP 当成一个信号，污染全部下游。
**本脚本默认过滤，并强制记录剔除行数。**

纯度判据（可配）：
    good_cs  ⇔  cs_size <= max-cs-size  且  cs_min_r2 >= min-r2
（若不提供该二列，则退化为按 cs_size 上限过滤并标注口径不完整）

用法：
    python loci_backbone.py --finemap finemap/x.snp.filter.tsv --good-cs --json
退出码：0 成功
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ALIAS = {
    "chrom": ["#chrom", "chromosome", "chrom", "chr"],
    "pos": ["position", "pos", "bp"],
    "region": ["region"],
    "v": ["v", "variant", "rsid"],
    "cs_size": ["cs_size"],
    "cs_avg_r2": ["cs_avg_r2"],
    "cs_min_r2": ["cs_min_r2"],
    "p": ["p", "pval"],
    "beta": ["beta", "b"],
}


def resolve(header):
    low = {h.lower(): h for h in header}
    out = {}
    for k, cands in ALIAS.items():
        for c in cands:
            if c.lower() in low:
                out[k] = header.index(low[c.lower()]); break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="可信集纯度过滤 + 位点骨架")
    ap.add_argument("--finemap", required=True, help="snp.filter / 可信集 TSV")
    ap.add_argument("--good-cs", action="store_true",
                    help="启用 good_cs 纯度过滤（关闭则仅统计，不剔除）")
    ap.add_argument("--max-cs-size", type=int, default=1000)
    ap.add_argument("--min-r2", type=float, default=0.8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-out", default=None, help="保留（good_cs）的变异清单输出路径")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    header = None
    rows = []
    with C.open_text(args.finemap) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = resolve(header)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if not p or p == [""]:
                continue
            rec = {}
            for k, i in col.items():
                try:
                    rec[k] = p[i]
                except IndexError:
                    rec[k] = None
            # 保留原始行：--keep-out 要按**原始表头**原样写回，
            # 而 rec 的键是归一化名（chrom/pos…）。曾用 r.get(h.lower()) 回取，
            # 遇到 chromosome/position 这类表头必然取不到 → 产物里
            # 染色体与位置两列全空，而报告照样说 n_kept=2。
            rec["__raw__"] = p
            rows.append(rec)

    if "cs_size" not in col:
        return C.fail("缺少 cs_size 列，无法做纯度过滤（口径不完整，拒绝静默放行）",
                      code=C.EXIT_CALIBER, as_json=args.json)

    def fnum(rec, key):
        try:
            return float(rec[key])
        except (TypeError, ValueError):
            return None

    kept, dropped = [], []
    for rec in rows:
        size = fnum(rec, "cs_size")
        min_r2 = fnum(rec, "cs_min_r2") if "cs_min_r2" in col else None
        good = (size is not None and size <= args.max_cs_size)
        if min_r2 is not None:
            good = good and (min_r2 >= args.min_r2)
        (kept if (good or not args.good_cs) else dropped).append(rec)

    by_region = defaultdict(int)
    for rec in kept:
        by_region[str(rec.get("region", "NA"))] += 1

    sizes = [fnum(r, "cs_size") for r in rows]
    sizes = [s for s in sizes if s is not None]

    report = {
        "ok": True,
        "summary": f"读入 {len(rows)} 行；good_cs 保留 {len(kept)}，剔除 {len(dropped)}"
                   + ("（未启用过滤）" if not args.good_cs else ""),
        "caliber": {
            "good_cs_rule": f"cs_size <= {args.max_cs_size} 且 cs_min_r2 >= {args.min_r2}",
            "filter_enabled": args.good_cs,
            "has_min_r2_column": "cs_min_r2" in col,
            "note": "低纯度大可信集必须剔除并记录剔除行数（红线 R6）",
        },
        "n_rows": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "dropped_sample": [
            {"region": r.get("region"), "v": r.get("v"), "cs_size": r.get("cs_size"),
             "cs_avg_r2": r.get("cs_avg_r2")} for r in dropped[:20]
        ],
        "n_regions": len(by_region),
        "variants_per_region": dict(sorted(by_region.items())),
        "cs_size_stats": {
            "max": max(sizes) if sizes else None,
            "median": sorted(sizes)[len(sizes) // 2] if sizes else None,
        },
    }

    if args.keep_out:
        Path(args.keep_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.keep_out, "w", encoding="utf-8", newline="") as fh:
            fh.write("\t".join(header) + "\n")
            for r in kept:
                raw = r.get("__raw__")
                if raw is not None:
                    # 原样写回原始列（缺列补空，保证列数与表头一致）
                    vals = list(raw)[:len(header)]
                    vals += [""] * (len(header) - len(vals))
                    fh.write("\t".join(vals) + "\n")
                else:  # pragma: no cover - 兜底
                    fh.write("\t".join(str(r.get(h.lower(), "")) for h in header) + "\n")

    if args.out:
        C.write_json(args.out, report)
    C.log(f"  保留 {len(kept)} / 剔除 {len(dropped)}（判据：{report['caliber']['good_cs_rule']}）")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
