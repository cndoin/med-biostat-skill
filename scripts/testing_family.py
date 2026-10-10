#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""testing_family.py —— 检验家族分层校正（Tier0，纯标准库）

PROTOCOL 硬约束（dhp / ptgds 共有）：

  > 主检验 Bonferroni（跨预先指定的主检验数）；次要 / 探索性检验 BH。

「家族」不是一个可以事后组装的概念——必须在动手前把检验分好族。本脚本据此：

  * family = `main`         → **Bonferroni**（控 FWER，主结论）
  * family = `secondary`    → **BH-FDR**
  * family = `exploratory`  → **BH-FDR**
  * 每族**独立**校正，并报告该族的检验次数

用法：
    python testing_family.py --input tests.tsv --json --out logs/testing_family.json
    # tests.tsv 列：test_id, p, family（family 取 main/secondary/exploratory）
退出码：0 成功 / 2 输入错误
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

MAIN = "main"
SECONDARY = "secondary"
EXPLORATORY = "exploratory"


def bonferroni(pvals: list) -> list:
    m = len(pvals)
    return [min(p * m, 1.0) for p in pvals]


def bh(pvals: list) -> list:
    """Benjamini-Hochberg，含单调化（保证 q 随秩次不减）。"""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [1.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q


METHOD = {MAIN: ("Bonferroni", bonferroni),
          SECONDARY: ("BH-FDR", bh),
          EXPLORATORY: ("BH-FDR", bh)}


def main() -> int:
    ap = argparse.ArgumentParser(description="检验家族分层校正")
    ap.add_argument("--input", required=True, help="TSV：test_id, p, family")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not Path(args.input).exists():
        return C.fail(f"文件不存在: {args.input}", code=C.EXIT_INPUT, as_json=args.json)

    tests = []
    try:
        with C.open_text(args.input) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            low = {h.lower().lstrip("#"): i for i, h in enumerate(header)}
            for h in header:
                low[h.lower()] = low.get(h.lower(), len(low))
            pi = low.get("p", low.get("pval", low.get("p_value")))
            fi = low.get("family", low.get("category"))
            ti = low.get("test_id", low.get("id", low.get("name", 0)))
            if pi is None:
                return C.fail(f"缺少 p 列；表头={header}", code=C.EXIT_INPUT, as_json=args.json)
            for line in fh:
                f = line.rstrip("\n").split("\t")
                try:
                    p = float(f[pi])
                except (ValueError, IndexError):
                    continue
                fam = (f[fi].strip().lower() if fi is not None else MAIN)
                if fam not in METHOD:
                    fam = EXPLORATORY
                tests.append({"test_id": f[ti] if ti is not None else "",
                              "p": p, "family": fam})
    except (OSError, ValueError) as e:
        return C.fail(f"读取失败：{e}", code=C.EXIT_INPUT, as_json=args.json)

    # 分族独立校正
    families = {}
    for fam in (MAIN, SECONDARY, EXPLORATORY):
        idx = [i for i, t in enumerate(tests) if t["family"] == fam]
        if not idx:
            continue
        ps = [tests[i]["p"] for i in idx]
        name, fn = METHOD[fam]
        qs = fn(ps)
        for k, i in enumerate(idx):
            tests[i]["correction"] = name
            tests[i]["p_adjusted"] = qs[k]
            tests[i]["significant"] = qs[k] < args.alpha
        families[fam] = {
            "method": name,
            "n_tests": len(idx),
            "n_significant": sum(1 for k in idx if tests[k]["significant"]),
            "min_p_adjusted": min(qs) if qs else None,
        }

    report = {
        "ok": True,
        "summary": f"{len(tests)} 个检验，{len(families)} 个家族："
                   + "；".join(f"{k}({v['method']}, n={v['n_tests']}, "
                               f"显著 {v['n_significant']})"
                               for k, v in families.items()),
        "caliber": {
            "main": "Bonferroni（控 FWER）",
            "secondary": "BH-FDR",
            "exploratory": "BH-FDR",
            "alpha": args.alpha,
            "rule": "每族独立校正；家族须在动手前分好，禁止事后重组",
        },
        "families": families,
        "tests": tests,
    }
    if args.out:
        C.write_json(args.out, report)
    for fam, v in families.items():
        C.log(f"  [{fam}] {v['method']}: {v['n_tests']} 个检验，"
              f"显著 {v['n_significant']}")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
