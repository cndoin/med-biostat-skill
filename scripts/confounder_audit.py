#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""confounder_audit.py —— S4 混杂与偏倚十二项审计

能自动判定的自动判定，其余输出 manual_required 并列出所需输入。
**无法查的必须写「无法查」（cannot_check + reason），不能写「未见异常」。**

用法：
    python confounder_audit.py --project <项目dir> --json
    python confounder_audit.py --project <dir> --endpoints ep.tsv --json
退出码：0 完成（含存在 cannot_check 的情形）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

# 十二项定义（与 references/confounders.md 对齐）
ITEMS = [
    (1, "群体分层", "lambda_gc"),
    (2, "年龄与性别", None),
    (3, "样本重叠", "endpoints"),
    (4, "共病与转诊", "comorbid_loci"),
    (5, "表型界定", None),
    (6, "效应量膨胀", None),
    (7, "组织错配", None),
    (8, "多重检验", "multiplicity"),
    (9, "窗口与阈值", "sensitivity"),
    (10, "技术性伪影", "qc_issues"),
    (11, "信息偏倚", None),
    (12, "研究者自由度", "gates_predefined"),
]


def check_endpoints(path):
    """S4③ 样本重叠：比对端点间对照数，完全相同的对照组标记为疑似共享。

    **缺列不得当作"通过"**。曾经的写法在 ni（n_control 列索引）为 None 时
    执行 f[None] 抛 TypeError，被 `except: continue` 吞掉 → dup 恒为空 →
    报告输出「未见对照数完全相同的端点组」。而本脚本第 6 行的红线明令禁止
    把「无法查」写成「未见异常」：缺列时必须返回 status=cannot_check，
    交由报告层决定如何披露，而不是给一个虚假的安心结论。
    """
    with C.open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        low = {h.lower(): i for i, h in enumerate(header)}
        ci = low.get("endpoint", low.get("name"))
        ni = low.get("n_control", low.get("controls"))

        missing = [n for n, v in (("endpoint", ci), ("n_control", ni)) if v is None]
        if missing:
            return {"status": "cannot_check",
                    "reason": f"端点表缺少必需列：{missing}",
                    "header": header,
                    "duplicate_control_groups": {}}

        n_by_ctrl: dict = {}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            try:
                n_by_ctrl.setdefault(f[ni], []).append(f[ci])
            except IndexError:
                continue
        dup = {k: v for k, v in n_by_ctrl.items() if len(v) > 1}
        return {"status": "checked",
                "duplicate_control_groups": dup,
                "n_endpoints": sum(len(v) for v in n_by_ctrl.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description="混杂十二项审计")
    ap.add_argument("--project", default=".", help="项目目录（读取 logs/ 与 tables/）")
    ap.add_argument("--endpoints", default=None, help="端点表（endpoint + n_control）")
    ap.add_argument("--sensitivity", default=None, help="多窗口结果 JSON（来自 loci_count）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    proj = Path(args.project)
    logs = proj / "logs"

    qc = C.read_json(logs / "qc_gwas.json")
    gates = C.read_json(logs / "gates.json")
    sens = C.read_json(args.sensitivity) if args.sensitivity else None

    results = []
    for num, name, dep in ITEMS:
        rec = {"id": num, "name": name, "status": "manual_required",
               "evidence": None, "conclusion": None, "reason": None}

        if name == "群体分层" and qc:
            lam = qc.get("lambda_gc")
            rec.update(status="checked",
                       evidence=f"λGC={lam}（口径 {qc.get('caliber', {}).get('lambda_gc_maf_bucket')}）",
                       conclusion=("λGC 未见膨胀" if (lam or 0) < 1.1
                                   else "λGC 偏高，须排查群体分层/技术伪影"))
        elif name == "样本重叠":
            if args.endpoints and Path(args.endpoints).exists():
                res = check_endpoints(args.endpoints)
                if res["status"] == "cannot_check":
                    # 查不了就说查不了 —— 不得输出「未见异常」（脚本红线）
                    rec.update(status="cannot_check",
                               reason=res["reason"],
                               evidence=None,
                               conclusion="无法判定（缺少必需列），不得读作无重叠")
                else:
                    dup = res["duplicate_control_groups"]
                    rec.update(status="checked",
                               evidence=f"对照数相同的端点组：{dup}",
                               conclusion=("存在共享对照，不得当独立证据合并" if dup
                                           else "未见对照数完全相同的端点组"))
            else:
                rec.update(status="cannot_check", reason="未提供 --endpoints（端点表）")
        elif name == "窗口与阈值" and sens:
            mx = sens.get("sensitivity_matrix", [])
            counts = sorted({m["n_independent_loci"] for m in mx})
            rec.update(status="checked" if mx else "cannot_check",
                       evidence=f"多窗口独立位点数：{counts}",
                       conclusion=("结论随窗口变化，须报告敏感性" if len(counts) > 1
                                   else "各窗口结果一致"),
                       reason=None if mx else "未提供 --sensitivity")
        elif name == "多重检验" and qc:
            rec.update(status="manual_required",
                       reason="需在分析配置中提供检验次数（区域数 × 组织数）后计入")
        elif name == "技术性伪影" and qc:
            issues = qc.get("sanity_issues", {})
            nonzero = {k: v for k, v in issues.items() if v}
            rec.update(status="checked",
                       evidence=f"健全性异常计数：{nonzero or '无'}",
                       conclusion=("存在技术性异常，须逐项排查" if nonzero
                                   else "未发现技术性异常"))
        elif name == "研究者自由度":
            rec.update(status="checked" if gates else "cannot_check",
                       evidence=f"gates.json 存在={bool(gates)}",
                       conclusion=("判定门已在动手前落盘" if gates
                                   else "未发现预置判定门"),
                       reason=None if gates else "logs/gates.json 缺失（判定门未前置）")
        else:
            rec.update(reason=f"需人工判断（依赖：{dep or '领域输入'}）")
        results.append(rec)

    n_checked = sum(1 for r in results if r["status"] == "checked")
    n_cannot = sum(1 for r in results if r["status"] == "cannot_check")
    n_manual = sum(1 for r in results if r["status"] == "manual_required")

    report = {
        "ok": True,
        "summary": f"已查 {n_checked} / 无法查 {n_cannot} / 待人工 {n_manual}（共 12 项）",
        "rule": "无法查的必须写「无法查」，不能写「未见异常」",
        "counts": {"checked": n_checked, "cannot_check": n_cannot,
                   "manual_required": n_manual},
        "items": results,
    }
    out = args.out or str(logs / "confounder_audit.json")
    C.write_json(out, report)
    for r in results:
        C.log(f"  [{r['status']:15s}] {r['id']:2d} {r['name']}"
              + (f" — {r['reason']}" if r["reason"] else ""))
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
