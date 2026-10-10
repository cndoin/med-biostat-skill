#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""trigger_check.py —— 自动触发自检（Tier0，纯标准库）

Agent 的**自动触发唯一取决于 `SKILL.md` 的 `description`**：agent 在启动时读到
所有 skill 的 name + description，据此判断是否加载。所以「能不能自动触发」
等价于「description 是否覆盖了用户真实的问法」。

本脚本用**真实场景问法**做两向验证：
  * 正向样本（应当触发）：问法的核心概念是否被 description 覆盖 → 漏触发是可量化的缺陷
  * 负向样本（不应触发）：完全无关的任务是否被误覆盖 → 误触发
  * 边界声明：description 必须显式写出「不触发」的条件

用法：
    python scripts/trigger_check.py --json
    python scripts/trigger_check.py --skill <SKILL.md> --samples <samples.json> --json
    # 换 skill 时必须同时换 samples —— 内置样本是**本 skill 领域专用**的，
    # 套到别的领域 skill 上会把「本就不该覆盖的概念」误判为漏触发。
退出码：0 触发可靠 / 1 存在漏触发或误触发
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
DEFAULT_SKILL = PKG / "SKILL.md"

# 正向样本：(用户会怎么问, [同义词组…]，任一命中即算覆盖)
POSITIVE = [
    ("帮我分析这份 GWAS 汇总统计", [["gwas"], ["汇总统计", "sumstats", "汇总"]]),
    ("这个基因和白内障有没有因果关系", [["因果"], ["孟德尔随机化", "mr", "mendelian"]]),
    ("跑一下共定位看看是不是同一个位点", [["共定位", "coloc"], ["位点", "locus"]]),
    ("核对一下我这组结果对不对", [["核对", "复核", "审计", "audit"]]),
    ("这个方向有没有人做过、能不能做", [["方向", "命题"], ["查重", "能不能做", "有没有人"]]),
    ("帮我做生存分析和预测模型", [["生存分析", "survival"], ["预测模型", "模型"]]),
    ("eQTL 工具变量应该怎么选", [["eqtl", "pqtl", "qtl"], ["工具变量", "instrument"]]),
    ("做一下差异表达和通路富集", [["差异表达", "deg"], ["富集", "通路", "enrichment"]]),
    ("这个药有没有因果证据", [["药物", "drug"], ["因果", "靶点"]]),
    ("帮我把它写成可投稿的报告", [["报告"], ["投稿", "strobe", "tripod", "规范"]]),
    ("我的数据质量和混杂怎么查", [["质控", "质量", "qc"], ["混杂", "confounder", "偏倚"]]),
    ("帮我核验数据是不是和官方一致", [["一致", "核验", "完整性"], ["哈希", "互证", "官方"]]),
    # 眼科 / 白内障：本项目重点方向，若 description 少了这块就**完全不会自动加载**
    ("帮我分析白内障的 GWAS 数据", [["白内障", "cataract"], ["gwas", "遗传", "位点"]]),
    ("这批白内障患者的人工晶体度数怎么算", [["人工晶体", "iol", "晶体"], ["屈光", "度数", "计算"]]),
    ("眼科随访数据的视力和眼压怎么统计", [["眼科", "眼压", "视力"], ["logmar", "统计", "随访"]]),
    # v1.7 新增的主流分析法：用户很可能**只用方法名**提问，
    # description 里没有对应词就完全不会自动加载（这是自动触发的唯一机制）
    ("帮我做 WGCNA 找共表达模块", [["wgcna", "共表达"], ["模块", "module", "网络"]]),
    ("这批单细胞数据怎么聚类分群", [["单细胞", "scrna", "single cell"], ["聚类", "分群", "cluster"]]),
    ("做 GSEA 和 GSVA 通路活性分析", [["gsea", "gsva"], ["通路", "富集", "活性"]]),
    ("用 LASSO 筛一下预后变量", [["lasso", "岭回归", "ridge", "elastic"], ["变量", "筛选", "预后"]]),
    ("这几项研究的结果做个 meta 分析", [["meta", "荟萃", "meta-analysis"], ["合并", "异质性", "i2"]]),
    ("两组基线不齐，做倾向性评分匹配", [["倾向性评分", "psm", "propensity"], ["匹配", "加权", "iptw"]]),
    ("算一下多基因风险评分 PRS", [["prs", "多基因风险", "polygenic"], ["风险评分", "评分"]]),
    ("这几个 SNP 哪个是因果变异，做精细定位", [["精细定位", "fine-mapping", "finemap"], ["因果变异", "可信集", "susie"]]),
    ("肿瘤样本的突变特征和 TMB", [["突变特征", "signature", "sbs"], ["tmb", "突变负荷"]]),
    ("空间转录组哪些基因有空间模式", [["空间转录组", "spatial"], ["空间", "moran", "邻域"]]),
    ("帮我算样本量够不够", [["样本量", "sample size"], ["功效", "power", "把握度"]]),
    ("免疫浸润比例怎么算", [["免疫浸润", "infiltration"], ["去卷积", "deconvolve", "浸润"]]),
    ("我的样本来自不同批次，怎么做批次校正", [["批次", "batch"], ["校正", "combat", "效应"]]),
    ("分析一下不同细胞类型之间的通讯", [["细胞通讯", "通讯", "配受体", "ligand"], ["细胞类型", "细胞"]]),
]

# 负向样本：完全无关，不应被覆盖
NEGATIVE = [
    "今天天气怎么样",
    "帮我把这段文字翻译成英文",
    "写一个贪吃蛇小游戏",
    "推荐几家附近的餐厅",
]

# 边界声明关键词（至少命中一个）
BOUNDARY = ["不触发", "不适用", "不要加载", "not triggered", "不加载"]


def load_description(skill_path: Path) -> str:
    text = skill_path.read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", text, re.S)
    if not m:
        return ""
    fm = m.group(1)
    # description 可能跨多行（续行缩进），这里按行收集
    lines = fm.splitlines()
    out, capturing = [], False
    for ln in lines:
        if ln.startswith("description:"):
            out.append(ln[len("description:"):].strip())
            capturing = True
        elif capturing and (ln.startswith(" ") or ln.startswith("\t")):
            out.append(ln.strip())
        elif capturing:
            capturing = False
    return " ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="自动触发自检")
    ap.add_argument("--skill", default=str(DEFAULT_SKILL))
    ap.add_argument("--samples", default=None,
                    help="外部样本 JSON：{\"positive\": [[问法,[关键词组…]],…], \"negative\":[…]}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    positive, negative = POSITIVE, NEGATIVE
    if args.samples:
        try:
            data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
            positive = [(q, g) for q, g in data.get("positive", [])] or positive
            negative = data.get("negative", []) or negative
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"样本读取失败: {e}"}, ensure_ascii=False))
            return 1

    skill_path = Path(args.skill)
    if not skill_path.exists():
        print(json.dumps({"ok": False, "error": f"找不到 {skill_path}"}, ensure_ascii=False))
        return 1

    desc = load_description(skill_path)
    if not desc:
        print(json.dumps({"ok": False, "error": "未从 frontmatter 解析到 description"},
                         ensure_ascii=False))
        return 1
    low = desc.lower()

    # 正向：每个样本的每个关键词组都要命中
    positive_results, missed = [], []
    for query, groups in positive:
        hit_groups = []
        for grp in groups:
            ok = any(kw.lower() in low for kw in grp)
            hit_groups.append(ok)
            if not ok:
                missed.append({"query": query, "missing_concept": grp})
        positive_results.append({
            "query": query, "covered": all(hit_groups),
            "groups": [{"concepts": g, "hit": h} for g, h in zip(groups, hit_groups)],
        })

    # 负向：不应命中任何「生信/医学分析」的核心词
    core = ["gwas", "生信", "孟德尔", "共定位", "eqtl", "富集分析", "位点", "临床统计"]
    negative_results = [{"query": q, "false_positive": any(k in q.lower() for k in core)}
                        for q in negative]

    has_boundary = any(b.lower() in low for b in BOUNDARY)
    n_pos = sum(1 for r in positive_results if r["covered"])
    n_fp = sum(1 for r in negative_results if r["false_positive"])
    ok = (n_pos == len(positive)) and (n_fp == 0) and has_boundary

    report = {
        "ok": ok,
        "summary": (f"正向覆盖 {n_pos}/{len(positive)}，误触发 {n_fp}，"
                    f"边界声明 {'有' if has_boundary else '缺'}"),
        "skill": str(skill_path),
        "description_chars": len(desc),
        "positive": {"n": len(positive), "covered": n_pos, "missing": missed,
                     "detail": positive_results},
        "negative": {"n": len(negative), "false_positives": n_fp, "detail": negative_results},
        "boundary_declared": has_boundary,
        "guidance": ("自动触发完全由 description 决定：漏掉的用户问法要补进 description，"
                     "无关任务靠「不触发」边界排除。"),
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"正向覆盖 {n_pos}/{len(POSITIVE)}；误触发 {n_fp}；"
              f"边界声明 {'有' if has_boundary else '缺'}")
        for m in missed:
            print(f"  [漏] 「{m['query']}」缺概念 {m['missing_concept']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
