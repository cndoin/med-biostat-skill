#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cis_instrument.py —— cis-eQTL / cis-pQTL 工具变量选择（Tier0，纯标准库）

药物靶点 MR 的第一步。判据全部来自项目 PROTOCOL（dhp / ptgds 共有）：

  * **cis 窗口**：基因边界 ± window（默认 100 kb）；--full-cis 用基因体全范围；
    敏感性可用 --window-kb 500
  * **显著性**：P < 5e-8
  * **弱工具排除**：F = (beta/se)² > 10
  * **非回文双等位**：排除 (A,T)/(T,A)/(C,G)/(G,C)，因其链方向无法确定
  * **一基因一组织一工具**：取最强 P 的 lead；--all-cis 可保留窗口内全部合格变异
  * **无合格工具就报告不可估计**，不更换主组织、不静默替换

用法：
    python cis_instrument.py --eqtl eqtl.tsv.gz --genes genes.tsv \
        --window-kb 100 --p-threshold 5e-8 --f-min 10 --json --out logs/instruments.json
退出码：0 成功 / 2 输入错误 / 3 口径缺失
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PALINDROMIC = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}

ALIAS = {
    "chrom": ["#chrom", "chrom", "chr", "chromosome"],
    "pos": ["pos", "position", "bp", "variant_pos"],
    "gene": ["gene", "gene_id", "gene_name", "symbol", "ensembl"],
    "tissue": ["tissue", "tissue_id", "cell_type", "condition"],
    "p": ["pval", "p", "p_value", "pval_nominal"],
    "beta": ["beta", "slope", "b", "effect"],
    "se": ["se", "sebeta", "stderr", "standard_error", "se_nominal"],
    "ref": ["ref", "reference_allele", "other_allele", "a2", "nea"],
    "alt": ["alt", "alt_allele", "effect_allele", "a1", "ea"],
    "af": ["af", "af_alt", "maf", "eaf", "freq"],
}


def resolve(header):
    """列名归一化查表：同时登记原样与去掉 '#' 前缀两种小写键。

    注意：不能先建一个 dict 再整体覆盖——那样 "#chrom" 会匹配不上 "chrom"。
    """
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


def load_gene_bounds(path: str) -> dict:
    """基因边界表：chrom/start/end/gene（或 gene/start/end 缺 chrom 时按同名基因匹配）。"""
    bounds = {}
    with C.open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        low = {h.lower(): i for i, h in enumerate(header)}
        ci, si, ei = low.get("chrom"), low.get("start"), low.get("end")
        gi = low.get("gene", low.get("gene_name", low.get("symbol")))
        if gi is None or si is None or ei is None:
            raise ValueError(f"基因边界表需含 gene/start/end 列，实际表头={header}")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            try:
                gene = f[gi]
                key = (C.normalize_chrom(f[ci]) if ci is not None else "", gene)
                bounds[key] = (int(f[si]), int(f[ei]))
                bounds[gene] = (int(f[si]), int(f[ei]))   # 无 chrom 时的退化键
            except (ValueError, IndexError):
                continue
    return bounds


def main() -> int:
    ap = argparse.ArgumentParser(description="cis 工具变量选择")
    ap.add_argument("--eqtl", required=True, help="eQTL/pQTL 汇总统计")
    ap.add_argument("--genes", required=True, help="基因边界表 gene/chrom/start/end")
    ap.add_argument("--window-kb", type=float, default=100.0,
                    help="基因边界两侧窗口（kb），PROTOCOL 主值 100")
    ap.add_argument("--full-cis", action="store_true",
                    help="只用基因体全范围（不给 ±window），用于敏感性")
    ap.add_argument("--p-threshold", type=float, default=5e-8)
    ap.add_argument("--f-min", type=float, default=10.0)
    ap.add_argument("--all-cis", action="store_true",
                    help="保留窗口内全部合格变异（默认只留最强 lead）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not Path(args.eqtl).exists():
        return C.fail(f"eQTL 文件不存在: {args.eqtl}", code=C.EXIT_INPUT, as_json=args.json)
    if args.genes and not Path(args.genes).exists():
        # --genes 与 --eqtl 同为必需输入；只校验前者会让缺文件时
        # FileNotFoundError 崩在 open() 处（exit 1），与「正常失败」撞车。
        return C.fail(f"基因坐标文件不存在: {args.genes}",
                      code=C.EXIT_INPUT, as_json=args.json)

    bounds = load_gene_bounds(args.genes)
    win = int(args.window_kb * 1000)

    header = None
    col = None
    # (gene, tissue) -> 候选列表
    cand: dict = defaultdict(list)
    n_scanned = n_in_window = n_palindromic = 0

    try:
        with C.open_text(args.eqtl) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            col = resolve(header)
            if not all(k in col for k in ("chrom", "pos", "p", "beta", "se")):
                return C.fail(f"eQTL 缺少必需列（chrom/pos/p/beta/se）；表头={header}",
                              code=C.EXIT_INPUT, as_json=args.json)
            for line in fh:
                f = line.rstrip("\n").split("\t")
                n_scanned += 1
                try:
                    chrom = C.normalize_chrom(f[col["chrom"]])
                    pos = int(f[col["pos"]])
                    p = float(f[col["p"]]); beta = float(f[col["beta"]]); se = float(f[col["se"]])
                except (ValueError, IndexError):
                    continue
                gene = f[col["gene"]] if "gene" in col else ""
                tissue = f[col["tissue"]] if "tissue" in col else "ALL"

                # 基因边界（优先 chrom+gene，退化到 gene）
                b = bounds.get((chrom, gene)) or bounds.get(gene)
                if b is None:
                    continue
                start, end = b
                lo = start if args.full_cis else start - win
                hi = end if args.full_cis else end + win
                if not (lo <= pos <= hi):
                    continue
                n_in_window += 1

                # 非回文双等位
                ref = (f[col["ref"]] if "ref" in col else "").upper()
                alt = (f[col["alt"]] if "alt" in col else "").upper()
                if ref and alt:
                    if (ref, alt) in PALINDROMIC:
                        n_palindromic += 1
                        continue
                    if len(ref) != 1 or len(alt) != 1:   # 仅双等位 SNV
                        continue
                cand[(gene, tissue)].append({
                    "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                    "p": p, "beta": beta, "se": se,
                    "af": float(f[col["af"]]) if "af" in col else None,
                })
    except (OSError, ValueError) as e:
        return C.fail(f"读取 eQTL 失败：{e}", code=C.EXIT_INPUT, as_json=args.json)

    # 逐 gene×tissue 选工具
    instruments, unavailable = [], []
    for (gene, tissue), rows in sorted(cand.items()):
        rows.sort(key=lambda r: r["p"])
        lead = rows[0]
        fstat = (lead["beta"] / lead["se"]) ** 2 if lead["se"] else float("inf")
        qualified = (lead["p"] < args.p_threshold) and (fstat > args.f_min)
        record = {
            "gene": gene, "tissue": tissue,
            "n_cis_variants": len(rows),
            "lead": {**lead,
                     "F": fstat,
                     "r2_approx": fstat / (fstat + max(len(rows) - 2, 1))},
            "qualified": qualified,
            "reason": ("" if qualified else
                       f"未通过（P={lead['p']:.3g}, F={fstat:.1f}）"),
        }
        if args.all_cis:
            record["all_cis"] = [
                {**r, "F": (r["beta"] / r["se"]) ** 2 if r["se"] else None}
                for r in rows if r["p"] < args.p_threshold
                and r["se"] and (r["beta"] / r["se"]) ** 2 > args.f_min
            ]
        (instruments if qualified else unavailable).append(record)

    report = {
        "ok": True,
        "summary": f"扫描 {n_scanned:,} 行；窗口内 {n_in_window:,}；"
                   f"合格工具 {len(instruments)} 个（gene×tissue），"
                   f"不可估计 {len(unavailable)} 个",
        "caliber": {
            "window_kb": None if args.full_cis else args.window_kb,
            "full_cis": args.full_cis,
            "p_threshold": args.p_threshold,
            "f_min": args.f_min,
            "palindromic_excluded": True,
            "biallelic_snv_only": True,
            "selection": "strongest_P_lead" if not args.all_cis else "all_cis_qualified",
            "note": "无合格工具须报告不可估计，不得更换主组织或静默替换（PROTOCOL 硬约束）",
        },
        "n_scanned": n_scanned,
        "n_in_window": n_in_window,
        "n_palindromic_excluded": n_palindromic,
        "instruments": instruments,
        "unavailable": unavailable,
    }
    if args.out:
        C.write_json(args.out, report)
    for r in instruments:
        C.log(f"  [OK] {r['gene']:<12} {r['tissue']:<20} "
              f"P={r['lead']['p']:.3g} F={r['lead']['F']:.1f}")
    for r in unavailable:
        C.log(f"  [不可估计] {r['gene']:<12} {r['tissue']:<20} {r['reason']}")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
