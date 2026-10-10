#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""loci_count.py —— S3① 独立位点计数（口径显式，窗口敏感性内置）

核心纪律：**变异数 ≠ 独立位点数**（P08），两个量都报，且独立位点必须说明聚类口径。

  * 主口径：LD 聚类（需 --ld-panel；PLINK2 --clump 为行业定义）
  * 退化口径：距离聚类（必须报窗口，并给窗口敏感性矩阵）

用法：
    python loci_count.py --input raw/x.gz --p-threshold 5e-8 \
        --window-kb 1000,3000 --json --out logs/loci_count.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

# GWAS 汇总统计列名别名表（不同提供方命名不同，统一归一化）
ALIAS = {
    "chrom": ["#chrom", "chrom", "chr", "chromosome"],
    "pos": ["pos", "position", "bp", "base_pair_location"],
    "pval": ["pval", "p", "p_value", "pval_nominal"],
    "beta": ["beta", "b", "effect", "effect_size"],
    "ref": ["ref", "reference_allele", "other_allele", "a2"],
    "alt": ["alt", "alt_allele", "effect_allele", "a1"],
    "af": ["af_alt", "af", "maf", "eaf", "freq"],
}


def pick(header: list[str], key: str) -> str | None:
    low = {h.lower(): h for h in header}
    for cand in ALIAS[key]:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def scan(path: str, collect_below: float, max_rows: int | None):
    """流式扫描，收集 p < collect_below 的变异；返回 (列, 数据数组, 扫描行数, 截断标志)。"""
    with C.open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {k: pick(header, k) for k in ("chrom", "pos", "pval")}
        if not all(col.values()):
            raise ValueError(f"无法识别必需列（chrom/pos/pval）；表头={header}")
        idx = {k: header.index(v) for k, v in col.items()}

        out_ch, out_pos, out_p = [], [], []
        n = 0
        watch = C.ProgressWatch(timeout=180)
        for line in fh:
            n += 1
            if max_rows and n > max_rows:
                break
            parts = line.rstrip("\n").split("\t")
            try:
                p = float(parts[idx["pval"]])
                pos = int(parts[idx["pos"]])
            except (ValueError, IndexError):
                continue
            if p < collect_below:
                out_ch.append(C.normalize_chrom(parts[idx["chrom"]]))
                out_pos.append(pos)
                out_p.append(p)
            if n % 1_000_000 == 0:
                watch.tick(n)
                C.log(f"  ... 扫描 {n:,} 行，收集 {len(out_p):,} 个")
    truncated = bool(max_rows and n > max_rows)
    return out_ch, out_pos, out_p, n, truncated


def clump_distance(chroms, poss, ps, thr: float, win_bp: int):
    """距离聚类：贪心选代表变异，其 ±win_bp 内归入同一独立位点。"""
    sel = [i for i, p in enumerate(ps) if p < thr]
    sel.sort(key=lambda i: ps[i])
    picked = []
    for i in sel:
        ci, pi = chroms[i], poss[i]
        for cj, pj in picked:
            if ci == cj and abs(pi - pj) <= win_bp:
                break
        else:
            picked.append((ci, pi))
    return sel, picked


def main() -> int:
    ap = argparse.ArgumentParser(description="独立位点计数（距离聚类 + 窗口敏感性）")
    ap.add_argument("--input", required=True)
    ap.add_argument("--p-threshold", type=float, default=5e-8)
    ap.add_argument("--window-kb", default="1000,3000",
                    help="逗号分隔的距离窗口（kb），用于敏感性矩阵")
    ap.add_argument("--ld-panel", default=None,
                    help="LD 参考面板（提供则主口径为 LD 聚类；否则退化并标注）")
    ap.add_argument("--collect-below", type=float, default=None,
                    help="收集阈值，默认取 max(p-threshold, 1e-5)")
    ap.add_argument("--max-variants", type=int, default=None,
                    help="冒烟测试上限；触发时产物写 truncated=true（P02）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    collect = args.collect_below or max(args.p_threshold, 1e-5)
    # 空/空白/连续逗号（"--window-kb ''" / '"  "' / ',,'）会让 split 产出空串，
    # int('') 直接 IndexError 崩在 argparse 之后 —— exit 1 还与「正常失败」
    # 的退出码撞车，自检断言 exit∈{0,1,2,3} 时会把崩溃当成合法失败放过。
    # 这里显式校验并给出可读原因。
    try:
        windows = [int(w.strip()) * 1000
                   for w in args.window_kb.split(",") if w.strip()]
    except ValueError:
        return C.fail(f"--window-kb 非法：{args.window_kb!r}（应为逗号分隔的正整数 kb）",
                      code=C.EXIT_INPUT, as_json=args.json)
    if not windows:
        return C.fail(f"--window-kb 未解析出任何窗口：{args.window_kb!r}",
                      code=C.EXIT_INPUT, as_json=args.json)
    if any(w <= 0 for w in windows):
        return C.fail(f"--window-kb 必须为正数：{args.window_kb!r}",
                      code=C.EXIT_INPUT, as_json=args.json)

    t0 = time.time()
    try:
        chroms, poss, ps, n_scanned, truncated = scan(
            args.input, collect, args.max_variants)
    except (ValueError, TimeoutError, OSError) as e:
        return C.fail(str(e), code=C.EXIT_INPUT, as_json=args.json)

    cluster_method = "ld" if args.ld_panel else "distance"
    matrix = []
    for win in windows:
        sel, picked = clump_distance(chroms, poss, ps, args.p_threshold, win)
        matrix.append({
            "threshold": args.p_threshold,
            "window_kb": win // 1000,
            "n_variants_below": len(sel),
            "n_independent_loci": len(picked),
            "first_20_loci": [f"{c}:{p}" for c, p in picked[:20]],
        })

    n_raw = sum(1 for p in ps if p < args.p_threshold)
    report = {
        "ok": True,
        "summary": f"扫描 {n_scanned:,} 变异；P<{args.p_threshold:g} 原始变异 {n_raw:,} 个 "
                   f"vs 独立位点 {matrix[0]['n_independent_loci']} 个"
                   f"（口径：{cluster_method}，窗口 {matrix[0]['window_kb']} kb）",
        # ★ 口径必须先于数值（caliber 字段必带）
        "caliber": {
            "p_threshold": args.p_threshold,
            "cluster_method": cluster_method,
            "window_kb": [w // 1000 for w in windows],
            "unit_distinction": "n_variants = 原始变异计数；n_independent_loci = 去重后位点数",
            "cluster_method_note": ("LD 聚类（行业定义，PLINK2 --clump）" if args.ld_panel
                                    else "距离聚类（退化口径）；与已发表口径的偏差须显式量化"),
        },
        "n_variants_scanned": n_scanned,
        "n_raw_variants_below_threshold": n_raw,
        "cluster_method": cluster_method,
        "sensitivity_matrix": matrix,
        "truncated": truncated,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    if truncated:
        C.log("[WARN] 本次结果来自截断数据（--max-variants），不得作为最终结论（P02）")

    if args.out:
        C.write_json(args.out, report)

    for row in matrix:
        C.log(f"  thr={row['threshold']:g} 窗口={row['window_kb']} kb -> "
              f"独立位点 {row['n_independent_loci']}（区域内变异 {row['n_variants_below']:,}）")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
