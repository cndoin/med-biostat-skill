#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""replication_check.py —— 外部复现检查（Tier0，纯标准库）

PROTOCOL 要求（dhp / ptgds 共有）：

  > 外部结局复核……公开 OR/P 舍入使重构 SE 为近似，明确标注，不强行合并。
  > 独立复现须检查 **ancestry** 与**暴露-结局样本重叠**。

本脚本给四件事，全部是「可检验的陈述」而非结论性判断：

  1. **方向一致性**：复现的效应方向与发现期是否一致（含 symbol 对齐）
  2. **显著性复现**：复现在名义 α 下是否达到显著（分「复现显著 / 方向一致但不显著 / 翻转」）
  3. **样本重叠**：若给了 n_case/n_control，比对两阶段是否可能共享对照（相同对照数 = 高度可疑）
  4. **ancestry 一致性**：两阶段人群是否同源；不同源须显式声明外推限制

**不合并**跨阶段效应量（PROTOCOL 明令），只做逐条对照。

用法：
    python replication_check.py --discovery d.tsv --replication r.tsv --json
    # 两表列：id, beta, se, p（可选 ancestry, n_case, n_control）
退出码：0 成功 / 2 输入错误
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

ALIAS = {
    "id": ["id", "snp", "rsid", "variant", "trait", "gene", "exposure", "test_id"],
    "beta": ["beta", "b", "effect", "slope", "or_log"],
    "se": ["se", "sebeta", "stderr", "standard_error"],
    "p": ["p", "pval", "p_value"],
    "ancestry": ["ancestry", "population", "ethnicity", "cohort_pop"],
    "n_case": ["n_case", "ncase", "cases"],
    "n_control": ["n_control", "ncontrol", "controls"],
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


def load(path):
    rows = {}
    meta = {}
    with C.open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = resolve(header)
        if "id" not in col or "beta" not in col or "p" not in col:
            raise ValueError(f"需要 id/beta/p 列；表头={header}")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            rid = f[col["id"]]
            try:
                rows[rid] = {"beta": float(f[col["beta"]]),
                             "p": float(f[col["p"]]),
                             "se": (float(f[col["se"]]) if "se" in col else None)}
            except (ValueError, IndexError):
                continue
            meta[rid] = {
                "ancestry": f[col["ancestry"]] if "ancestry" in col else None,
                "n_case": (int(float(f[col["n_case"]])) if "n_case" in col else None),
                "n_control": (int(float(f[col["n_control"]])) if "n_control" in col else None),
            }
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="外部复现检查")
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--replication", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (args.discovery, args.replication):
        if not Path(p).exists():
            return C.fail(f"文件不存在: {p}", code=C.EXIT_INPUT, as_json=args.json)

    try:
        d, dm = load(args.discovery)
        r, rm = load(args.replication)
    except (OSError, ValueError) as e:
        return C.fail(f"读取失败：{e}", code=C.EXIT_INPUT, as_json=args.json)

    common = sorted(set(d) & set(r))
    if not common:
        return C.fail("两表无共同 id，无法对照", code=C.EXIT_INPUT, as_json=args.json)

    per = []
    n_same_dir = n_sig_rep = n_flip = 0
    for rid in common:
        bd, br = d[rid]["beta"], r[rid]["beta"]
        pd_, pr = d[rid]["p"], r[rid]["p"]
        same_dir = (bd > 0) == (br > 0)
        sig_rep = pr < args.alpha
        flip = not same_dir
        n_same_dir += int(same_dir)
        n_sig_rep += int(sig_rep and same_dir)
        n_flip += int(flip)
        per.append({
            "id": rid,
            "discovery": {"beta": bd, "p": pd_},
            "replication": {"beta": br, "p": pr},
            "same_direction": same_dir,
            "replication_significant": sig_rep,
            "verdict": ("复现成功（同向且显著）" if same_dir and sig_rep else
                        "同向但未达显著" if same_dir else "方向翻转"),
        })

    # 样本重叠与 ancestry
    overlap_flags, ancestry_flags = [], []
    for rid in common:
        a, b = dm[rid], rm[rid]
        if a.get("n_control") is not None and b.get("n_control") is not None:
            if a["n_control"] == b["n_control"]:
                overlap_flags.append({"id": rid, "n_control": a["n_control"],
                                      "warning": "两阶段对照数完全相同 → 高度可疑共享对照"})
        if a.get("ancestry") and b.get("ancestry"):
            if a["ancestry"] != b["ancestry"]:
                ancestry_flags.append({"id": rid, "discovery": a["ancestry"],
                                       "replication": b["ancestry"]})

    n = len(common)
    report = {
        "ok": True,
        "summary": f"共同 id {n} 个：同向 {n_same_dir}/{n}（{100*n_same_dir/n:.0f}%），"
                   f"复现显著 {n_sig_rep}/{n}，方向翻转 {n_flip}",
        "caliber": {
            "alpha": args.alpha,
            "method": "逐条对照（方向 / 显著性 / 重叠 / ancestry）",
            "no_pooling": True,
            "note": "跨阶段效应量**不合并**；公开值舍入导致的 SE 近似须显式标注（PROTOCOL）",
        },
        "counts": {"n_common": n, "n_same_direction": n_same_dir,
                   "n_replication_significant": n_sig_rep, "n_flipped": n_flip},
        "sample_overlap_flags": overlap_flags,
        "ancestry_mismatch": ancestry_flags,
        "per_test": per,
    }
    if args.out:
        C.write_json(args.out, report)

    C.log(f"同向 {n_same_dir}/{n}；复现显著 {n_sig_rep}；翻转 {n_flip}")
    for f in overlap_flags:
        C.log(f"  [样本重叠?] {f['id']} 对照数 {f['n_control']}")
    for f in ancestry_flags:
        C.log(f"  [ancestry 不一致] {f['id']} {f['discovery']} vs {f['replication']}")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
