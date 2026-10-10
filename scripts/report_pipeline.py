#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""report_pipeline.py —— S6/S7 确定性文档流水线：Markdown → 模板 HTML → 门禁 → DOCX

**硬约束（红线 R10）**：含数字的报告绝不能让写作代理改写。本流水线是**确定性转换**：
输入字符串进、HTML 字符串出，逐字节可复现；数字只从 Markdown 原样搬运。

门禁（可被 test_guards_negative 证明会 FAIL）：
  * 结构门禁：报告契约的必需章节齐全
  * 数字门禁：正文中每个数字必须能在 source-data 目录中找到，或带 `[src:key]` 标注

用法：
    python report_pipeline.py --md report/draft.md --contract academic_paper \
        --sourcedata tables/ --out output/report.html --json
退出码：0 全过 / 4 门禁 FAIL
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
TEMPLATE = PKG / "assets" / "html_template" / "head_only.html"
CONTRACTS = PKG / "assets" / "report_contracts"

NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _strip_sep(tok: str) -> str:
    """去掉千分位逗号，让 1,234 与 1234 视为同一个数字。"""
    return tok.replace(",", "") if "," in tok else tok


# ---- 极简 Markdown → HTML（确定性，无第三方） ----------------------------
def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def inline(s: str) -> str:
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    return s


def md_to_html(md: str) -> tuple[str, list[str]]:
    """返回 (html, headings)。仅支持本 skill 用到的子集：标题/段落/表格/列表/代码/引用。"""
    lines = md.splitlines()
    out, headings = [], []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            lvl = len(ln) - len(ln.lstrip("#"))
            txt = ln[lvl:].strip()
            headings.append(txt)
            out.append(f"<h{min(lvl,6)}>{inline(txt)}</h{min(lvl,6)}>")
        elif ln.startswith("|") and i + 1 < len(lines) and set(lines[i+1].replace("|", "").strip()) <= set("-: "):
            # 表格
            header = [c.strip() for c in ln.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            i -= 1
            th = "".join(f"<th>{inline(c)}</th>" for c in header)
            trs = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                          for r in rows)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
        elif ln.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            out.append("<pre><code>" + esc("\n".join(buf)) + "</code></pre>")
        elif ln.startswith(("- ", "* ")):
            buf = []
            while i < len(lines) and lines[i].startswith(("- ", "* ")):
                buf.append(f"<li>{inline(lines[i][2:])}</li>"); i += 1
            i -= 1
            out.append("<ul>" + "".join(buf) + "</ul>")
        elif ln.startswith(">"):
            out.append(f'<div class="callout">{inline(ln[1:].strip())}</div>')
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out), headings


def load_contract(name: str) -> dict:
    """极简 YAML 读取：**只取 required_sections**（不能把 optional_sections 当必需）。

    逐行状态机实现，避免正则误抓 optional 段（实测：会把「补充材料」误判为必需章节）。
    """
    p = CONTRACTS / f"{name}.yaml"
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    sections, cur, in_req = [], None, False
    for ln in text.splitlines():
        if re.match(r"^required_sections:", ln):
            in_req = True
            continue
        if in_req and re.match(r"^\S", ln):   # 遇到下一个顶层 key → 结束
            in_req = False
        if not in_req:
            continue
        m_id = re.match(r"\s*-\s*id:\s*(\S+)", ln)
        m_ti = re.match(r"\s*title:\s*(.+)", ln)
        if m_id:
            cur = {"id": m_id.group(1), "title": None}
        if m_ti and cur is not None:
            cur["title"] = m_ti.group(1).strip()
            sections.append(cur)
            cur = None
    return {"sections": [s for s in sections if s.get("title")], "raw": text}


def main() -> int:
    ap = argparse.ArgumentParser(description="确定性文档流水线")
    ap.add_argument("--md", required=True)
    ap.add_argument("--contract", default="academic_paper")
    ap.add_argument("--sourcedata", default=None, help="source data 目录（数字可追溯）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-gate", action="store_true", help="跳过门禁（仅调试用，报告标注）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    md = Path(args.md).read_text(encoding="utf-8")
    body, headings = md_to_html(md)

    # ---- 门禁 1：结构 ----
    contract = load_contract(args.contract)
    missing_sections = []
    if contract.get("sections"):
        for s in contract["sections"]:
            if not any(s["title"] in h or s["id"] in h.lower() for h in headings):
                missing_sections.append(s["title"])

    # ---- 门禁 2：数字可追溯 ----
    # 契约里写了 require_source_data: true 就必须真的强制 ——
    # 曾经不传 --sourcedata 时门禁 enabled:false 但 pass:true，等于
    # 「没检查却报告通过」，是最典型的静默失效（R2）。
    requires_sd = bool(re.search(r"require_source_data:\s*true",
                                 contract.get("raw", "")))
    if requires_sd and not args.sourcedata:
        return C.fail(
            f"契约 {args.contract} 要求 require_source_data: true，"
            "必须提供 --sourcedata <目录>；缺它时数字门禁无法判定",
            code=C.EXIT_GATE_FAIL, as_json=args.json)

    # 注意：不能用 `m not in pool` 这种**子串**判断 —— 报告写 999999 而
    # source data 里只有 1999999 时，子串匹配会放行这个幻觉数字（实测可绕过）。
    # 改为「按数字边界切词后再查」，并额外支持去掉千分位/科学计数法后匹配。
    untraceable = []
    if args.sourcedata:
        pool = ""
        for f in Path(args.sourcedata).rglob("*"):
            if f.is_file() and f.suffix in (".tsv", ".csv", ".json", ".txt"):
                pool += f.read_text(encoding="utf-8", errors="ignore")
        # 建立 source data 里的数字词集合（同样去千分位）
        pool_tokens = set(NUM_RE.findall(pool))
        pool_tokens |= {_strip_sep(t) for t in pool_tokens}
        for m in set(NUM_RE.findall(md)):
            if len(m) < 3:          # 忽略 0/1 等单字
                continue
            if m in pool_tokens or _strip_sep(m) in pool_tokens:
                continue
            untraceable.append(m)

    gates = {
        "structure": {"pass": not missing_sections, "missing": missing_sections},
        "number_traceability": {"pass": not untraceable,
                                "untraceable_sample": sorted(untraceable)[:30],
                                "enabled": bool(args.sourcedata)},
    }
    all_pass = all(g["pass"] for g in gates.values())

    # ---- 套模板 ----
    title = headings[0] if headings else "报告"
    tpl = TEMPLATE.read_text(encoding="utf-8") if TEMPLATE.exists() else "{{BODY}}"
    html = (tpl.replace("{{TITLE}}", esc(title))
               .replace("{{BODY}}", body))

    if args.out and (all_pass or args.skip_gate):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(html, encoding="utf-8")

    report = {
        "ok": all_pass,
        "summary": ("门禁全过，已产出 HTML" if all_pass
                    else f"门禁 FAIL：缺章节 {len(missing_sections)}，"
                         f"不可追溯数字 {len(untraceable)}"),
        "contract": args.contract,
        "gates": gates,
        "headings": headings,
        "html_bytes": len(html.encode("utf-8")),
        "out": args.out if (args.out and (all_pass or args.skip_gate)) else None,
        "determinism_note": "本流水线为确定性转换；含数字段落禁止写作代理改写（R10）",
        "docx_hint": "如需 DOCX：把本 HTML 交给文档流水线（tencent-docx html-to-docx）转换",
    }
    if args.out and not all_pass and not args.skip_gate:
        C.log("[GATE FAIL] 门禁未过，未写出文件（如需强制输出加 --skip-gate）")
    return C.emit(report, as_json=args.json,
                  code=C.EXIT_OK if all_pass else C.EXIT_GATE_FAIL)


if __name__ == "__main__":
    sys.exit(main())
