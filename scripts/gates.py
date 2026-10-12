#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gates.py —— 判定门的前置落盘（S0/S1，Tier0 纯标准库）

## 为什么必须有这个脚本（2026-10-12 逻辑框架审查发现的缺口）

`SKILL.md` 一直写着"判定门要**动手前**写好，落盘 `logs/gates.json`"，
但审查发现：**仓库里没有任何脚本会写这个文件**。
后果不是"少个功能"，而是门禁可以被无声绕过 ——
AI 直接手写一个 gates.json（或干脆不写），审计时看不出来，
而"研究者自由度"那一项恰恰是靠这个文件的存在与否来判的。
也就是说：门 ⑫ 的输入本身不可审计，门就成了摆设。

所以这个脚本只做一件事：**在分析开始之前，把判定条件写成机器可校验的文件**。
它不改任何数值、不参与统计，只负责让"事前定门"这件事留下可审计的痕迹。

## 三条写死的口径

1. **判定条件必须是结构化的、可判定的**，不接受自由文本。
   「创新性足够高」不是判定条件；「≥60% 的风险基因在目标细胞类型可检出」
   才是（operator + threshold + 口径说明齐全）。
2. **事后不许改门**：`--amend` 只允许**收紧**（提高阈值或缩小通过范围），
   并且必须写 `--reason`。放宽一律拒绝 —— 放宽阈值让结果"通过"是最常见的
   事后合理化，不给它一个方便的命令行。
3. **`gates.json` 里带 `sha256` 指纹**：任何后续改动都会改变指纹，
   `confounder_audit` 与 CI 都能据此发现"门被改过"。改了就必须显式 `--amend`，
   而 `--amend` 留痕。

用法：
    python scripts/gates.py --init --project . --json
    python scripts/gates.py --check --project . --json     # 校验结构 + 指纹
    python scripts/gates.py --amend --gate G2 --min 0.8 --reason "..." --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PKG = Path(__file__).resolve().parents[1]

# 六道判定门（与 SKILL.md「判定门」小节一一对应）。
# `direction`：'higher_better' = 指标越大越通过；'lower_better' 反之。
GATE_SPECS = [
    {"id": "G1", "name": "命题门", "when": "S0 完成",
     "question": "概念/技术/实体三路径任一命中「命题已被正面回答」？",
     "action": "停止，不得换措辞继续"},
    {"id": "G2", "name": "数据门", "when": "S1 完成",
     "question": "关键源不可得，或三级源承担核心暴露且无法落实？",
     "action": "换源或降级，不得用近似数据顶替"},
    {"id": "G3", "name": "质控门", "when": "S2 完成",
     "question": "可用变异低于下限，或指标与官方基准偏差超阈值且原因未查明？",
     "action": "先排查口径，未查明不得进 S3"},
    {"id": "G4", "name": "阳性对照门", "when": "S3 首轮",
     "question": "阳性对照不成立？", "action": "停止主分析，先修参考数据"},
    {"id": "G5", "name": "增量门", "when": "S3 主结论后",
     "question": "新方法相对已有方法一致性高于上限（无新信息）？",
     "action": "停止，方向无增量"},
    {"id": "G6", "name": "阴性门", "when": "全程",
     "question": "得到 P>0.05 且无解释力？", "action": "不得包装成阳性，按阴性写"},
]

OPERATORS = {
    "gt": lambda v, t: v > t,
    "ge": lambda v, t: v >= t,
    "lt": lambda v, t: v < t,
    "le": lambda v, t: v <= t,
}

DIRECTIONS = {
    # 指标越大越该"通过"（如保真保留率、检出率）
    "higher_better": True,
    # 指标越小越该"通过"（如 λGC、混合度）
    "lower_better": False,
}


def is_tighter(new_op: str, new_th: float, old_th: float) -> bool:
    """新阈值是否比旧阈值**更严**。

    判据只认 operator 指向的方向，不看 `direction` 字段 ——
    那个字段可能没填或填错，用它判会漏放"放宽"。这里刻意保守：
    无法判定方向时一律视为**不收紧**（即拒绝），因为误放一次就等于门失效。
    """
    if new_op in ("gt", "ge"):
        return new_th >= old_th
    if new_op in ("lt", "le"):
        return new_th <= old_th
    return False


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def fingerprint(doc: dict) -> str:
    """指纹只覆盖**判定条件本身**，不覆盖 status/observed 这些事后填的结果。

    这是关键设计：结果字段每轮分析都会变，若把它算进指纹，
    每跑一次分析指纹就变一次，"门被改过"就永远无法判定。
    """
    core = {g["id"]: {k: g.get(k) for k in ("name", "metric", "operator",
                                             "threshold", "direction", "unit")}
            for g in doc.get("gates", [])}
    return hashlib.sha256(_canon(core).encode("utf-8")).hexdigest()[:16]


def build(args) -> dict:
    doc = {
        "schema": "biostat-copilot/gates@1",
        "project": str(Path(args.project).resolve().name),
        "note": ("判定门在分析开始前落盘。事后只允许**收紧**（--amend 提高阈值或"
                 "缩小通过范围），放宽一律拒绝。criteria 里 metric/operator/threshold"
                 "缺一不可 —— 「创新性足够高」不是判定条件，机器无法判定。"),
        "gates": [],
    }
    for spec in GATE_SPECS:
        g = dict(spec)
        g.update(status="pending", metric=None, operator=None,
                 threshold=None, direction=None, unit=None,
                 observed=None, amended=None)
        doc["gates"].append(g)
    return doc


def apply_thresholds(doc: dict, args) -> dict:
    """把命令行给的判定条件写进门。缺的保持 None（= 待定，不是"通过"）。"""
    by_id = {g["id"]: g for g in doc["gates"]}
    given = {
        "G2": ("min_detectable_rate", "ge"),
        "G3": ("min_lambda_gc_headroom", "le"),
        "G4": ("min_positive_control_hits", "ge"),
        "G5": ("max_agreement_with_existing", "le"),
        "G6": ("max_p_value", "le"),
    }
    for gid, (metric, default_op) in given.items():
        if gid not in by_id:
            continue
        val = getattr(args, metric, None)
        if val is None:
            continue
        g = by_id[gid]
        g["metric"] = metric
        g["operator"] = args.operator or default_op
        g["threshold"] = float(val)
        g["direction"] = ("higher_better" if g["operator"] in ("gt", "ge")
                           else "lower_better")
        g["unit"] = "ratio" if "rate" in metric or "control" in metric else "value"
    if getattr(args, "min_age_trends", None) is not None:
        by_id["G2"]["secondary"] = {"metric": "min_age_trend_genes",
                                    "operator": "ge",
                                    "threshold": float(args.min_age_trends),
                                    "direction": "higher_better",
                                    "unit": "genes"}
    return doc


def validate(doc: dict) -> list:
    errs = []
    for g in doc.get("gates", []):
        crit = [g.get(k) for k in ("metric", "operator", "threshold", "direction")]
        if all(v is None for v in crit):
            continue                      # 待定，允许
        if any(v is None for v in crit):
            errs.append(f"{g['id']}：metric/operator/threshold/direction 必须同时给全，"
                        f"不能只给一部分")
            continue
        if g["operator"] not in OPERATORS:
            errs.append(f"{g['id']}：operator={g['operator']!r} 不支持"
                        f"（可用：{sorted(OPERATORS)}）")
        if g["direction"] not in DIRECTIONS:
            errs.append(f"{g['id']}：direction={g['direction']!r} 不支持")
        if not isinstance(g["threshold"], (int, float)):
            errs.append(f"{g['id']}：threshold 必须是数字")
    if not doc.get("gates"):
        errs.append("gates 为空 —— 判定门必须在动手前落盘")
    return errs


def check(doc: dict) -> dict:
    errs = validate(doc)
    fp = fingerprint(doc)
    stored = doc.get("fingerprint")
    amended = [g["id"] for g in doc.get("gates", []) if g.get("amended")]

    # 指纹不匹配 = 落盘之后有人绕过 --amend 直接改了文件。
    # 这**必须**让 ok=False：否则"门被改过"这件事只写在输出里、退出码仍是 0，
    # CI 与上游脚本会照样放行 —— 门禁自己失效。（实测踩到过。）
    if stored is not None and stored != fp:
        errs.append(
            f"指纹不匹配（记录 {stored} vs 实际 {fp}）：判定门在落盘后被直接改过。"
            f"请用 --amend 改（只允许收紧，且必须留 --reason），"
            f"或删掉 gates.json 重新 --init（若尚未开始分析）")

    return {
        # 结构错 or 被手改 → 一律 ok=False
        "ok": not errs,
        "n_gates": len(doc.get("gates", [])),
        "fingerprint": fp,
        "fingerprint_matches": (stored == fp) if stored else None,
        "amended": amended,
        "pending": [g["id"] for g in doc.get("gates", [])
                    if g.get("status") == "pending"],
        "errors": errs,
        "rule": ("指纹不匹配 = 判定门被绕过 --amend 直接改过（ok=False，退出码 1）；"
                 "事后只允许 --amend 收紧，放宽一律拒绝"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="判定门前置落盘与校验")
    ap.add_argument("--project", default=".")
    ap.add_argument("--init", action="store_true", help="新建 gates.json")
    ap.add_argument("--check", action="store_true", help="只校验结构与指纹")
    ap.add_argument("--amend", action="store_true", help="事后收紧（不放宽）")
    ap.add_argument("--gate", default=None, help="--amend 时指定门 id（如 G2）")
    ap.add_argument("--min-detectable-rate", type=float, default=None)
    ap.add_argument("--min-age-trends", type=float, default=None,
                    help="G2 次要条件：FDR<0.05 的年龄趋势基因数下限")
    ap.add_argument("--min-lambda-gc-headroom", type=float, default=None)
    ap.add_argument("--min-positive-control-hits", type=float, default=None)
    ap.add_argument("--max-agreement-with-existing", type=float, default=None)
    ap.add_argument("--max-p-value", type=float, default=None)
    ap.add_argument("--operator", default=None, choices=sorted(OPERATORS))
    ap.add_argument("--reason", default=None, help="--amend 必填")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    proj = Path(args.project)
    logs = proj / "logs"
    path = Path(args.out) if args.out else (logs / "gates.json")

    if args.init:
        if path.exists() and not args.amend:
            sys.stderr.write(
                f"[WARN] {path} 已存在。判定门在分析前落盘后不该重建 —— "
                f"要改请用 --amend（只允许收紧）。\n")
        doc = build(args)
        doc = apply_thresholds(doc, args)
        errs = validate(doc)
        if errs:
            return sys.exit(C.emit({"ok": False, "errors": errs},
                                    as_json=args.json, code=C.EXIT_ERROR))
        doc["fingerprint"] = fingerprint(doc)
        C.write_json(str(path), doc)
        return sys.exit(C.emit({"ok": True, "written": str(path),
                                "summary": f"已落盘 {len(doc['gates'])} 道判定门",
                                "fingerprint": doc["fingerprint"]},
                               as_json=args.json))

    if args.amend:
        if not path.exists():
            return sys.exit(C.emit({
                "ok": False,
                "errors": [f"{path} 不存在 —— 事后才想起定门已经晚了，"
                           f"请回到 S0 重走"]}, as_json=args.json, code=C.EXIT_ERROR))
        if not args.reason:
            return sys.exit(C.emit({
                "ok": False,
                "errors": ["--amend 必须带 --reason（改门要留痕）"]},
                as_json=args.json, code=C.EXIT_ERROR))
        doc = C.read_json(path) or {}
        gates = doc.get("gates", [])
        target = [g for g in gates if g.get("id") == args.gate] if args.gate else gates
        if not target:
            return sys.exit(C.emit({
                "ok": False, "errors": [f"找不到门 {args.gate!r}"]},
                as_json=args.json, code=C.EXIT_ERROR))

        # 每道门对应一个"新阈值"的来源。显式列出而不是用一堆 getattr 猜，
        # 免得某个参数名对不上时静默什么都不改（那会读成"改门成功"）。
        THRESHOLD_ARG = {
            "G2": ("min_detectable_rate", "min_age_trends"),
            "G3": ("min_lambda_gc_headroom",),
            "G4": ("min_positive_control_hits",),
            "G5": ("max_agreement_with_existing",),
            "G6": ("max_p_value",),
        }
        changed, refused = [], []
        for g in target:
            keys = THRESHOLD_ARG.get(g["id"], ())
            new_th = next((getattr(args, k) for k in keys
                           if getattr(args, k, None) is not None), None)
            if new_th is None:
                continue
            new_th = float(new_th)
            new_op = args.operator or g.get("operator") or "ge"
            old_th = g.get("threshold")
            if old_th is not None and not is_tighter(new_op, new_th, old_th):
                refused.append(f"{g['id']}：{old_th} → {new_th} 是**放宽**，"
                               f"拒绝（放宽阈值让结果通过是最常见的事后合理化）")
                continue
            g["operator"] = new_op
            g["threshold"] = new_th
            g["direction"] = ("higher_better" if new_op in ("gt", "ge")
                              else "lower_better")
            g["amended"] = {"reason": args.reason, "previous_threshold": old_th}
            changed.append(g["id"])

        if not changed:
            return sys.exit(C.emit({
                "ok": False,
                "errors": refused or ["没有任何门被修改："
                                      + f"{args.gate or '指定门'} 没有可用的阈值参数"]},
                as_json=args.json, code=C.EXIT_ERROR))
        doc["fingerprint"] = fingerprint(doc)
        C.write_json(str(path), doc)
        return sys.exit(C.emit({"ok": True, "amended": changed, "refused": refused,
                                "fingerprint": doc["fingerprint"]},
                               as_json=args.json))

    # 默认与 --check 都是校验
    if not path.exists():
        return sys.exit(C.emit({
            "ok": False,
            "errors": [f"{path} 不存在 —— 判定门未在动手前落盘，"
                       f"请先跑 scripts/gates.py --init"]},
            as_json=args.json, code=C.EXIT_ERROR))
    doc = C.read_json(path) or {}
    report = check(doc)
    return sys.exit(C.emit(report, as_json=args.json,
                           code=C.EXIT_OK if report["ok"] else C.EXIT_ERROR))


if __name__ == "__main__":
    sys.exit(main())