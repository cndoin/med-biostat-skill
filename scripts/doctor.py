#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""doctor.py —— 环境与依赖自检（跨 agent 首跑必看）

报告：Python 版本、可选加速包、skill 目录完整性、脚本可执行性。
零重依赖：只用标准库，任何裸 Python 3.9+ 都能跑。

用法：
    python doctor.py --json
退出码：0 全就绪 / 1 有阻断项
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PKG = Path(__file__).resolve().parents[1]

REQUIRED = [
    "SKILL.md",
    "VERSION",
    "references/data_sources.md",
    "references/qc_definitions.md",
    "references/confounders.md",
    "references/stats_criteria.md",
    "references/reporting_standards.md",
    "references/tools.md",
    "references/pitfalls.md",
    "references/domain/ophthalmology.md",
    "scripts/_common.py",
]
OPTIONAL_PKGS = ["numpy", "pandas", "duckdb", "polars", "pyarrow", "scipy",
                 "pytest", "hypothesis"]


def main() -> int:
    ap = argparse.ArgumentParser(description="biostat-copilot 环境自检")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    py_ok = sys.version_info >= (3, 9)
    missing_files = [f for f in REQUIRED if not (PKG / f).exists()]

    pkgs = {}
    for name in OPTIONAL_PKGS:
        m = C.optional(name)
        pkgs[name] = C.version_of(m)

    version = (PKG / "VERSION").read_text(encoding="utf-8").strip() \
        if (PKG / "VERSION").exists() else "?"

    blockers = []
    if not py_ok:
        blockers.append(f"Python {sys.version.split()[0]} < 3.9")
    if missing_files:
        blockers.append(f"缺失 {len(missing_files)} 个必需文件")

    report = {
        "ok": not blockers,
        "summary": f"biostat-copilot v{version} · 环境就绪" if not blockers
                   else f"有 {len(blockers)} 个阻断项",
        "skill_version": version,
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "meets_min_3_9": py_ok,
        },
        "optional_packages": pkgs,
        "acceleration": {
            # 注意：缺失包的版本是字符串 "-"，不能直接 bool()（非空串恒真）。
            "vectorized": all(pkgs.get(x) != "-" for x in ("numpy", "pandas")),
            "columnar": any(pkgs.get(x) != "-" for x in ("duckdb", "polars")),
            "note": "缺失不影响功能，自动回退标准库流式实现",
        },
        "core_engine": "stdlib（无第三方运行时依赖）",
        "missing_files": missing_files,
        "blockers": blockers,
    }

    for name, ver in pkgs.items():
        C.log(f"  {'OK  ' if ver != '-' else 'MISS'} {name:12s} {ver}")
    for b in blockers:
        C.log(f"[BLOCK] {b}")

    return C.emit(report, as_json=args.json,
                  code=C.EXIT_OK if not blockers else C.EXIT_ERROR)


if __name__ == "__main__":
    sys.exit(main())
