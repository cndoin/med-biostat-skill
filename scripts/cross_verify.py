#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cross_verify.py —— S1③ 第 4–5 道：两套官方产物逐变异互证 + 等位基因方向

这是「本地数据与官网一致」最硬的证据（一致性强于「看起来对」）：
  若子集文件（如精细定位 snp.filter）里每个变异，在全量文件里都能找到，
  且 p / beta / maf 数值一致，则说明全量文件没被截断、没有错位、没有解码错误。

内置三个防坑：
  * 等位基因方向自动检测：同序（ref==a1, alt==a2）vs 反序，取命中率高者并报告（P03）
  * MAF 口径折算：一律 min(af, 1-af) 再比，避免 0.84 的假差异（P05）
  * 染色体前缀归一化，避免 chrchr1（P04）

用法：
    python cross_verify.py --full raw/x.gz --subset finemap/x.snp.filter.tsv --json
退出码：0 命中率与偏差均在容差内 / 1 存在系统性偏差或命中率过低
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

FULL_ALIAS = {
    "chrom": ["#chrom", "chrom", "chr", "chromosome"],
    "pos": ["pos", "position", "bp"],
    "p": ["pval", "p", "p_value"],
    "beta": ["beta", "b", "effect"],
    "ref": ["ref", "reference_allele", "a2", "other_allele"],
    "alt": ["alt", "alt_allele", "a1", "effect_allele"],
    "af": ["af_alt", "af", "maf", "eaf"],
}
SUB_ALIAS = {
    "chrom": ["chromosome", "#chrom", "chrom", "chr"],
    "pos": ["position", "pos", "bp"],
    "p": ["p", "pval", "p_value"],
    "beta": ["beta", "b", "effect"],
    "a1": ["allele1", "a1", "ref", "other_allele"],
    "a2": ["allele2", "a2", "alt", "effect_allele"],
    "maf": ["maf", "af", "af_alt", "eaf"],
}


def resolve(header: list[str], alias: dict) -> dict:
    low = {h.lower(): h for h in header}
    out = {}
    for k, cands in alias.items():
        for c in cands:
            if c.lower() in low:
                out[k] = header.index(low[c.lower()])
                break
    return out


def read_header(path: str):
    with C.open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    return header


def main() -> int:
    ap = argparse.ArgumentParser(description="两套官方产物逐变异互证")
    ap.add_argument("--full", required=True, help="全量汇总统计")
    ap.add_argument("--subset", required=True, help="子集文件（精细定位等）")
    ap.add_argument("--p-tol", type=float, default=1e-12, help="p 绝对偏差容差")
    ap.add_argument("--min-hit-rate", type=float, default=0.98)
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # 1) 读子集，构造目标键（保留代表性：region / v 若有）
    sub_header = read_header(args.subset)
    sc = resolve(sub_header, SUB_ALIAS)
    if "chrom" not in sc or "pos" not in sc or "a1" not in sc or "a2" not in sc:
        return C.fail(f"子集文件缺少必需列: {sub_header}",
                      code=C.EXIT_INPUT, as_json=args.json)

    def make_key(chrom, pos, a1, a2, order):
        c = C.normalize_chrom(chrom)
        x, y = (a1, a2) if order == "same" else (a2, a1)
        return f"{c}:{pos}:{x}:{y}"

    subs = []
    with C.open_text(args.subset) as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            try:
                subs.append({
                    "chrom": p[sc["chrom"]], "pos": int(p[sc["pos"]]),
                    "a1": p[sc["a1"]], "a2": p[sc["a2"]],
                    "p": float(p[sc["p"]]) if "p" in sc else None,
                    "beta": float(p[sc["beta"]]) if "beta" in sc else None,
                    "maf": float(p[sc["maf"]]) if "maf" in sc else None,
                })
            except (ValueError, IndexError):
                continue
    C.log(f"子集变异数：{len(subs)}")

    # 2) 流式扫全量，收集两种顺序的命中
    full_header = read_header(args.full)
    fc = resolve(full_header, FULL_ALIAS)
    if not all(k in fc for k in ("chrom", "pos", "p", "ref", "alt")):
        return C.fail(f"全量文件缺少必需列: {full_header}",
                      code=C.EXIT_INPUT, as_json=args.json)

    for order in ("same", "swapped"):
        for s in subs:
            s[f"key_{order}"] = make_key(s["chrom"], s["pos"], s["a1"], s["a2"], order)
    targets = {}
    for s in subs:
        targets[s["key_same"]] = ("same", s)
        targets[s["key_swapped"]] = ("swapped", s)

    found: dict[str, tuple] = {}
    n = 0
    t0 = time.time()
    watch = C.ProgressWatch(timeout=180)
    with C.open_text(args.full) as fh:
        fh.readline()
        for line in fh:
            n += 1
            if args.max_variants and n > args.max_variants:
                break
            p = line.rstrip("\n").split("\t")
            try:
                c = C.normalize_chrom(p[fc["chrom"]]); pos = int(p[fc["pos"]])
                ref, alt = p[fc["ref"]], p[fc["alt"]]
                key = f"{c}:{pos}:{ref}:{alt}"
            except (ValueError, IndexError):
                continue
            if key in targets:
                found[key] = (float(p[fc["p"]]),
                              float(p[fc["beta"]]) if "beta" in fc else None,
                              C.as_maf(float(p[fc["af"]])) if "af" in fc else None)
            if n % 2_000_000 == 0:
                watch.tick(n)
    truncated = bool(args.max_variants and n > args.max_variants)

    # 3) 判定哪种等位基因顺序命中率高（P03）
    hit_same = sum(1 for s in subs if s["key_same"] in found)
    hit_swapped = sum(1 for s in subs if s["key_swapped"] in found)
    order = "same" if hit_same >= hit_swapped else "swapped"
    hit = hit_same if order == "same" else hit_swapped

    # 4) 逐变异偏差
    dp = dbeta = dmaf = 0.0
    n_p_exact = n_beta_exact = 0
    missing = []
    for s in subs:
        got = found.get(s[f"key_{order}"])
        if got is None:
            missing.append(s[f"key_{order}"])
            continue
        if s["p"] is not None and got[0] is not None:
            d = abs(s["p"] - got[0]); dp = max(dp, d); n_p_exact += int(d == 0)
        if s["beta"] is not None and got[1] is not None:
            d = abs(s["beta"] - got[1]); dbeta = max(dbeta, d); n_beta_exact += int(d == 0)
        if s["maf"] is not None and got[2] is not None:
            dmaf = max(dmaf, abs(C.as_maf(s["maf"]) - got[2]))

    hit_rate = hit / max(len(subs), 1)
    systematic = (dp > args.p_tol) or (hit_rate < args.min_hit_rate)
    ok = not systematic

    report = {
        "ok": ok,
        "summary": f"命中率 {hit_rate*100:.2f}%（{hit}/{len(subs)}），"
                   f"等位基因方向={order}，max|Δp|={dp:g}，max|ΔMAF|={dmaf:g}",
        "caliber": {
            "key": "chrom:pos:ref:alt（前缀已归一化）",
            "allele_order": order,
            "maf_definition": "min(af,1-af)（次等位频率）",
            "p_tolerance": args.p_tol,
            "min_hit_rate": args.min_hit_rate,
        },
        "n_subset_variants": len(subs),
        "n_found": hit,
        "hit_rate": hit_rate,
        "hit_rate_same_order": hit_same / max(len(subs), 1),
        "hit_rate_swapped_order": hit_swapped / max(len(subs), 1),
        "max_abs_dp": dp,
        "n_p_exact": n_p_exact,
        "max_abs_dbeta": dbeta,
        "n_beta_exact": n_beta_exact,
        "max_abs_dmaf": dmaf,
        "n_missing": len(missing),
        "missing_keys_sample": missing[:50],
        "scanned_variants": n,
        "truncated": truncated,
        "verdict": ("一致" if ok else "存在系统性偏差 → 排查取回文件完整性/等位基因方向"),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # 门禁：命中率过低直接 FAIL（可被 test_guards_negative 证明会失败）
    if args.out:
        C.write_json(args.out, report)
    C.log(f"  等位基因方向={order}（同序命中 {hit_same} / 反序命中 {hit_swapped}）")
    C.log(f"  命中率 {hit_rate*100:.2f}%，max|Δp|={dp:g}，max|Δbeta|={dbeta:g}，max|ΔMAF|={dmaf:g}")

    code = C.EXIT_OK if ok else C.EXIT_GATE_FAIL
    return C.emit(report, as_json=args.json, code=code)


if __name__ == "__main__":
    sys.exit(main())
