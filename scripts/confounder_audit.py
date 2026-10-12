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
#
# ⚠️ 这十二项是**为 GWAS / MR / 队列**设计的。对单细胞与蛋白组分析，它漏掉了
# 三类该领域特有的主要混杂（2026-10-12 审查发现，故显式写出来而不是假装覆盖）：
#   A. **伪重复**——把细胞当独立样本，样本量虚增两三个数量级，p 值必然假阳性；
#      （`lib.pseudobulk` 是闸门，但闸门本身要有人确认过了）
#   B. **整合伪影**——批次校正过头会把年龄/处理信号一起抹掉，且**不报错**；
#      （`lib.integration.fidelity` 是判据）
#   C. **细胞组成变化**——表观"某基因上调"可能只是某细胞类型比例变了；
#      （`lib.composition` 是判据）
# 补的处置方式见 EXTENDED_ITEMS：与十二项**并列**的领域扩展项，
# 默认不计入 12（避免改变既有口径），用 --extended 打开。
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

# 领域扩展项（v1.10 新增）。刻意**不并进上面十二项**：
# 十二项的编号与 references/confounders.md 已对齐多年，插新项会打断既有对照，
# 而审计表的"共 N 项"口径一旦变了，历年报告就不可比。
EXTENDED_ITEMS = [
    (13, "伪重复（细胞当独立样本）", "pseudobulk_gate"),
    (14, "整合伪影（信号被批次校正抹掉）", "integration_fidelity"),
    (15, "细胞组成变化", "composition"),
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
    ap.add_argument("--extended", action="store_true",
                    help="追加 3 项单细胞/蛋白组特有的混杂（伪重复 / 整合伪影 / 细胞组成）")
    ap.add_argument("--pseudobulk", default=None,
                    help="pseudobulk 聚合产物 JSON（供体×基因矩阵摘要）")
    ap.add_argument("--fidelity", default=None,
                    help="lib.integration.fidelity 的输出 JSON")
    ap.add_argument("--composition", default=None,
                    help="lib.composition.test 的输出 JSON")
    args = ap.parse_args()

    proj = Path(args.project)
    logs = proj / "logs"

    qc = C.read_json(logs / "qc_gwas.json")
    gates = C.read_json(logs / "gates.json")
    sens = C.read_json(args.sensitivity) if args.sensitivity else None
    pb = C.read_json(args.pseudobulk) if args.pseudobulk else None
    fid = C.read_json(args.fidelity) if args.fidelity else None
    comp = C.read_json(args.composition) if args.composition else None

    items = list(ITEMS)
    if args.extended:
        items = items + EXTENDED_ITEMS

    results = []
    for num, name, dep in items:
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
        # ── 领域扩展项（--extended）────────────────────────────────────
        elif name == "伪重复（细胞当独立样本）":
            # 判据是**分析单位**而不是细胞数：n_cells > n_donors 且下游检验
            # 直接用细胞当 n —— 这时样本量被虚增两三个数量级，p 必然假阳性。
            if not pb:
                rec.update(status="cannot_check",
                           reason="未提供 --pseudobulk；细胞级分析若未先做供体聚合，"
                                  "结论一律不得读作独立证据")
            else:
                n_donor = int(pb.get("n_donors", 0) or 0)
                n_cell = int(pb.get("n_cells", 0) or 0)
                n_low = int(pb.get("n_donors_below_min", 0) or 0)
                rec.update(status="checked",
                           evidence=f"供体 {n_donor} / 细胞 {n_cell}"
                                    + (f"；细胞数不足的供体 {n_low}" if n_low else ""),
                           conclusion=("已按供体聚合，分析单位是供体"
                                       if n_donor > 0 else
                                       "聚合产物里供体数为 0，不得据此宣称做了供体层级检验"))
        elif name == "整合伪影（信号被批次校正抹掉）":
            if not fid:
                rec.update(status="cannot_check",
                           reason="未提供 --fidelity；批次校正不保证保留任何协变量信号，"
                                  "缺这一步就无法排除「整合把年龄/处理效应一起抹掉」")
            else:
                v = fid.get("verdict")
                mr = fid.get("min_retention")
                rec.update(status="checked",
                           evidence=f"verdict={v}，min_retention={mr}",
                           conclusion=("整合保留了关键协变量的可解释方差"
                                       if v == "PASS" else
                                       "整合**未**通过保真诊断：下游差异分析的结果"
                                       "不能归因于目标变量（FAIL/WARN/UNINFORMATIVE）"))
        elif name == "细胞组成变化":
            if not comp:
                rec.update(status="cannot_check",
                           reason="未提供 --composition；表观表达变化可能只来自亚群比例改变")
            else:
                n_sig = comp.get("n_significant")
                rec.update(status="checked",
                           evidence=f"亚群比例显著项数={n_sig}",
                           conclusion=("已检验亚群比例随协变量的变化，"
                                       "表达差异须与组成变化分开解释"
                                       if n_sig else
                                       "未检出亚群比例变化（不等于不存在，"
                                       "取决于检验功效）"))
        else:
            rec.update(reason=f"需人工判断（依赖：{dep or '领域输入'}）")
        results.append(rec)

    n_checked = sum(1 for r in results if r["status"] == "checked")
    n_cannot = sum(1 for r in results if r["status"] == "cannot_check")
    n_manual = sum(1 for r in results if r["status"] == "manual_required")

    report = {
        "ok": True,
        "summary": (f"已查 {n_checked} / 无法查 {n_cannot} / 待人工 {n_manual}"
                    f"（共 {len(items)} 项"
                    + ("，含 3 项单细胞/蛋白组扩展" if args.extended else "") + "）"),
        "rule": "无法查的必须写「无法查」，不能写「未见异常」",
        "extended": bool(args.extended),
        "n_items": len(items),
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
