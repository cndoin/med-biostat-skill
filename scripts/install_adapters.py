#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""install_adapters.py —— 跨 Agent 安装器（单一来源 → 多端入口）

原则：`SKILL.md` 是唯一权威来源。本脚本**只生成入口，不复制正文**
（正文由 agent 触发后按需读取），避免「同一逻辑多处实现」造成漂移（P21）。

目标：
  claude   复制整包 → ~/.claude/skills/biostat-copilot/   （原生读 SKILL.md）
  codex    复制整包 → ~/.codex/skills/biostat-copilot/    （原生读 SKILL.md）
  agents   生成 <project>/AGENTS.md                        （Cursor/Zed/Amp/Jules/aider）
  cursor   生成 <project>/.cursor/rules/<name>.mdc
  cline    生成 <project>/.clinerules/<name>.md
  gemini   生成 <project>/GEMINI.md
  copilot  生成 <project>/.github/copilot-instructions.md
  all      以上全部

用法：
    python install_adapters.py --target all --project "<项目目录>" --json
    python install_adapters.py --target claude --json          # 只装到 Claude Code
退出码：0 成功 / 1 有失败项
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
NAME = "biostat-copilot"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.parts", "*.tmp",
                                ".git", "*.egg-info", ".pytest_cache")
# 陈旧文件比对时同样要跳过的目录名 / 后缀（与 IGNORE 保持同一套判据）
_IGNORE_DIRS = {"__pycache__", ".git", ".pytest_cache"}
_IGNORE_SUFFIXES = (".pyc", ".parts", ".tmp", ".egg-info")

# 入口文件里对外暴露的触发描述（取自 SKILL.md frontmatter.description）
TRIGGER = ("生信 / 医学研究全流程编排（GWAS、精细定位、孟德尔随机化、共定位、eQTL/pQTL/TWAS、"
           "临床统计、差异与通路、混杂审计、报告规范对表）。"
           "触发词：生信分析、GWAS、MR、共定位、eQTL、富集分析、精细定位、位点、"
           "差异表达、WGCNA、生存分析、预测模型、单细胞、蛋白组、bioinformatics、"
           "colocalization、fine-mapping、Mendelian randomization。")


def read_version() -> str:
    v = (PKG / "VERSION")
    return v.read_text(encoding="utf-8").strip() if v.exists() else "?"


def body_pointer() -> str:
    """指向 SKILL.md 与其关键片段的摘要（保持简短，正文按需读）。"""
    return (
        f"# {NAME} · 生信研究全流程副驾 (v{read_version()})\n\n"
        f"**唯一权威来源**：`{PKG / 'SKILL.md'}`（用 read 工具按需读取；本文件只是入口指针，不含正文）。\n\n"
        "## 何时启用\n"
        f"{TRIGGER}\n\n"
        "## 三条核心主张\n"
        "1. **口径先于数值** —— 缺口径即报错，不给默认值。\n"
        "2. **一致性强于「看起来对」** —— 一致性必须由哈希 / 字节 / 逐变异互证证明。\n"
        "3. **混杂是主战场** —— S4 独立一层，十二项逐条过。\n\n"
        "## 红线（摘要）\n"
        "- 拒绝「七件套」等模板套路；数字必须有出处；阴性不得包装成阳性；\n"
        "- 含数字的报告禁止写作代理改写；可信集必须按 good_cs 过滤；禁止用「最近基因」当区域基因。\n\n"
        "## 路由\n"
        "| 用户给的是 | 走哪条线 |\n|---|---|\n"
        "| 一个想法 | S0（含命题级查重）→ 方向裁定 |\n"
        "| 数据集 + 问题 | S1→S2→S3 |\n"
        "| 已有结果要核对 | S2 复算 + S4 混杂审计 |\n"
        "| 要产出一篇 | 全链路 + S6 |\n\n"
        "## 三层能力\n"
        f"- **Tier 0** `{PKG / 'scripts'}`：**纯标准库**，任何裸 Python 都能跑 —— "
        "`doctor` `fetch_parallel` `verify_integrity` `qc_gwas` `loci_count` `loci_backbone` "
        "`annotate_regions` `confounder_audit` `cross_verify` `report_pipeline` `capabilities`。\n"
        f"- **Tier 1** `{PKG / 'vendor'}`：需 numpy+pandas+scipy —— 临床统计 / 生存 / 建模（无 sklearn）/ "
        "**MR+共定位** / 差异+富集 / PTM / 纯 SVG 图 / 函数注册表 / TRIPOD+AI·STROBE 对表。\n"
        f"- **原生加速层** `{PKG / 'vendor' / 'native'}`：单文件可执行（魔改版 biostat-vcf，MIT/Apache-2.0），"
        "百万行汇总统计流式质控（λGC + MAF 分档 + 异常计数），直读 `.tsv`/`.tsv.gz`。\n"
        f"  探测：`python \"{PKG / 'scripts' / 'capabilities.py'}\" --json`；"
        f"安装：`pip install -r \"{PKG / 'vendor' / 'requirements.txt'}\"`。\n"
        "  注意：Tier1 缺依赖不影响 Tier0；`vendor/` 内文件由 `tools/vendor_sync.py` 同步，不要直接编辑。\n"
        "  ⚠️ 原生层 λGC 是 Wald 均值口径，与 `qc_gwas.py` 的 p 值中位数口径**不是同一个数**，"
        "只在 `--native` 时启用且必带 `engine`/`lambda_caliber` 字段。\n\n"
        f"自检：`python \"{PKG / 'tests' / 'run_all.py'}\"`（期望全绿）。\n"
    )


def write_entry(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def install_copy(dest_root: Path) -> dict:
    """把整包**就地覆盖同步**到 agent 的 skills 目录。

    为什么不再用 `rmtree + copytree`：
    1. `~/.claude/skills` 这类目录属于用户个人目录，整目录删除是破坏性操作，
       会被安全护栏拦下（实测 `[SAFE_DELETE_BULK_CONFIRM_REQUIRED]`，
       105 个文件超过阈值）；
    2. 覆盖同步是**幂等**的：重复安装不会留下半删状态，失败可安全重试；
    3. 不需要删目录就能完成「更新」，语义上更贴近「安装」而非「重建」。

    对源中已不存在的**陈旧文件**：只报告、不删除（删除属破坏性操作，
    交回用户显式决定），并在返回值里列出，避免「静默残留」。
    """
    dest = dest_root / NAME
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PKG, dest, ignore=IGNORE, dirs_exist_ok=True)

    def _kept(rel: Path) -> bool:
        return (not any(part in _IGNORE_DIRS for part in rel.parts)
                and not rel.as_posix().endswith(_IGNORE_SUFFIXES))

    src_files = {
        p.relative_to(PKG).as_posix() for p in PKG.rglob("*")
        if p.is_file() and _kept(p.relative_to(PKG))
    }
    stale = sorted(
        p.relative_to(dest).as_posix() for p in dest.rglob("*")
        if p.is_file() and _kept(p.relative_to(dest))
        and p.relative_to(dest).as_posix() not in src_files
    )
    return {
        "installed_to": str(dest),
        "files": sum(1 for _ in dest.rglob("*") if _.is_file()),
        "mode": "overwrite-sync（非破坏性，不删目录）",
        "stale_files_not_removed": stale[:20],
        "stale_count": len(stale),
    }


def gen_agents(project: Path) -> Path:
    return write_entry(project / "AGENTS.md", body_pointer())


def gen_cursor(project: Path) -> Path:
    mdc = (f"---\ndescription: {TRIGGER}\nglobs:\nalwaysApply: false\n---\n\n"
           + body_pointer())
    return write_entry(project / ".cursor" / "rules" / f"{NAME}.mdc", mdc)


def gen_cline(project: Path) -> Path:
    return write_entry(project / ".clinerules" / f"{NAME}.md", body_pointer())


def gen_gemini(project: Path) -> Path:
    return write_entry(project / "GEMINI.md", body_pointer())


def gen_copilot(project: Path) -> Path:
    return write_entry(project / ".github" / "copilot-instructions.md", body_pointer())


def main() -> int:
    ap = argparse.ArgumentParser(description="跨 Agent 安装器")
    ap.add_argument("--target", default="agents",
                    choices=["claude", "codex", "workbuddy", "agents", "cursor",
                             "cline", "gemini", "copilot", "all"])
    ap.add_argument("--project", default=".", help="项目目录（生成项目级入口用）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    home = Path.home()
    project = Path(args.project).resolve()
    results = {}
    errors = []

    def run(key, fn):
        try:
            results[key] = fn()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{key}: {e}")

    targets = {args.target} if args.target != "all" else {
        "claude", "codex", "workbuddy", "agents", "cursor", "cline", "gemini", "copilot"}

    if "claude" in targets:
        run("claude", lambda: install_copy(home / ".claude" / "skills"))
    if "codex" in targets:
        run("codex", lambda: install_copy(home / ".codex" / "skills"))
    if "workbuddy" in targets:
        run("workbuddy", lambda: install_copy(home / ".workbuddy" / "skills"))
    if "agents" in targets:
        run("agents", lambda: {"entry": str(gen_agents(project))})
    if "cursor" in targets:
        run("cursor", lambda: {"entry": str(gen_cursor(project))})
    if "cline" in targets:
        run("cline", lambda: {"entry": str(gen_cline(project))})
    if "gemini" in targets:
        run("gemini", lambda: {"entry": str(gen_gemini(project))})
    if "copilot" in targets:
        run("copilot", lambda: {"entry": str(gen_copilot(project))})

    report = {
        "ok": not errors,
        "summary": f"已处理 {len(results)} 个目标"
                   + (f"，{len(errors)} 个失败" if errors else ""),
        "skill_version": read_version(),
        "source": str(PKG),
        "project": str(project),
        "results": {k: v for k, v in results.items()},
        "errors": errors,
        "note": "入口文件只含指针，正文由 agent 触发后读取 SKILL.md（单一来源，防漂移）",
    }
    for k, v in results.items():
        tgt = v.get("installed_to") or v.get("entry")
        C.log(f"  [OK] {k:8s} → {tgt}")
    for e in errors:
        C.log(f"  [ERR] {e}")
    return C.emit(report, as_json=args.json,
                  code=C.EXIT_OK if not errors else C.EXIT_ERROR)


if __name__ == "__main__":
    sys.exit(main())
