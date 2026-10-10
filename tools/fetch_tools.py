#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fetch_tools.py —— 外部工具获取器（把「获取能力」内嵌进 skill）

背景（为什么不是把源码复制进来）：
  本 skill 核查过设计文档 §11.5 的开源清单，结论是**开源生信工具基本无法内嵌源码**：
    * GPL/LGPL 系（GWASLab / PLINK2 / ldsc / coloc / DESeq2 / FUSION …）
      一旦并入分发即触发许可传染 —— 违反 §11.4 红线；
    * 许可干净的（pysam/PyDESeq2/scanpy/lifelines/…) 二进制或依赖过重，搬运不划算。
    * GWASLab 的「许可不一致」已结案：**v3.3.x 及以前是 MIT，v3.4.39 起改为 GPL-3.0-only**
      （仓库内 LICENSE 与 LICENSE_before_v3.4.39 并存即切换点）。

因此正确做法是：把**获取与调用能力**内嵌，把**工具本体**装到隔离目录、不随 skill 分发。
  * A 档（MIT/BSD/Apache）→ 可直接 `pip install`
  * B 档（GPL 系）→ 可获取用于对拍，**明确标记不可再分发**

用法：
    python tools/fetch_tools.py --list --json
    python tools/fetch_tools.py --check --json
    python tools/fetch_tools.py --fetch gwaslab --json           # 装到隔离目录
    python tools/fetch_tools.py --fetch-all --tier A --json
退出码：0 成功 / 1 有失败
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# 隔离安装根目录（**在 skill 目录之外**，绝不随 skill 分发）
TOOLS_ROOT = Path(os.environ.get("BIOSTAT_TOOLS_ROOT", Path.home() / ".biostat-copilot" / "tools"))

TOOLS = [
    # ---- A 档：许可干净，可直接装 ----
    {"name": "numpy", "tier": "A", "license": "BSD-3", "method": "pip",
     "purpose": "Tier1 引擎的数值基础", "redistributable": True},
    {"name": "pandas", "tier": "A", "license": "BSD-3", "method": "pip",
     "purpose": "Tier1 引擎的表格基础", "redistributable": True},
    {"name": "scipy", "tier": "A", "license": "BSD-3", "method": "pip",
     "purpose": "统计检验与分布", "redistributable": True},
    {"name": "statsmodels", "tier": "A", "license": "BSD-3", "method": "pip",
     "purpose": "Cox 比例风险（PHReg）、校准斜率", "redistributable": True},
    {"name": "lifelines", "tier": "A", "license": "MIT", "method": "pip",
     "purpose": "生存分析对拍（B 档金标准）", "redistributable": True},
    {"name": "pytest", "tier": "A", "license": "MIT", "method": "pip",
     "purpose": "测试运行器（可选，内置 run_all 已覆盖）", "redistributable": True},
    {"name": "hypothesis", "tier": "A", "license": "MPL-2.0", "method": "pip",
     "purpose": "属性化边界测试（可选）", "redistributable": True},
    {"name": "duckdb", "tier": "A", "license": "MIT", "method": "pip",
     "purpose": "列式扫描加速（可选）", "redistributable": True},
    {"name": "polars", "tier": "A", "license": "MIT", "method": "pip",
     "purpose": "列式 DataFrame 加速（可选）", "redistributable": True},
    {"name": "pyarrow", "tier": "A", "license": "Apache-2.0", "method": "pip",
     "purpose": "Parquet 中间格式（可选）", "redistributable": True},

    # ---- B 档：GPL 系，只可对拍，不可分发 ----
    {"name": "gwaslab", "tier": "B", "license": "GPL-3.0-only", "method": "pip",
     "purpose": "对拍：build 推断 / 等位基因协调 / sumstats sanity check / lead 提取",
     "redistributable": False,
     "note": "v3.3.x 及以前为 MIT，v3.4.39 起改为 GPL-3.0-only。**不得并入本 skill 分发**"},
    {"name": "ldsc", "tier": "B", "license": "GPL-3.0", "method": "git",
     "repo": "https://github.com/bulik/ldsc",
     "purpose": "对拍：遗传力 / 遗传相关；LDSC-SEG 方法学来源", "redistributable": False},
    {"name": "plink2", "tier": "B", "license": "GPL-3.0", "method": "download",
     "url": "https://s3.amazonaws.com/plink2-assets/plink2_win64_latest.zip",
     "purpose": "独立位点的行业定义：--clump（LD 聚类）", "redistributable": False,
     "note": "Windows 版；Linux/macOS 请到 https://www.cog-genomics.org/plink/2.0/ 取对应包"},

    # ---- C 档：只借方法学（不获取，仅记录）----
    {"name": "LDSC-SEG", "tier": "C", "license": "GPL-3.0（依托 ldsc）", "method": "method",
     "purpose": "组织富集主方法（前10%特异表达基因 + 100kb + 52基线注释）",
     "redistributable": False, "note": "方法学必答题，自行实现，不搬运代码"},
    {"name": "sc-linker", "tier": "C", "license": "未核实", "method": "method",
     "purpose": "细胞类型 τ* 指标", "redistributable": False},
    {"name": "scDRS", "tier": "C", "license": "未核实", "method": "method",
     "purpose": "GWAS→单细胞 评分（与 SC→GWAS 互补）", "redistributable": False},
]

BY_NAME = {t["name"]: t for t in TOOLS}


def _looks_installed(dest: Path) -> bool:
    """目录存在 **且** 真的装了东西，才算已获取。

    曾经的 bug：--check 只判 dest.exists()。但 pip install 失败时
    do_fetch 已经先 mkdir 了，失败后留下**空目录** → --check 报 [OK] 已安装，
    而实际什么都没有。这是典型的「失败被上报成成功」——
    空目录与「已安装」在效果上对下游完全等价（import 一定失败）。
    """
    try:
        return dest.is_dir() and any(dest.iterdir())
    except OSError:
        return False


def _target_for(name: str) -> Path:
    return TOOLS_ROOT / name


def do_fetch(tool: dict) -> dict:
    name = tool["name"]
    method = tool["method"]
    dest = _target_for(name)

    if method == "method":
        return {"name": name, "status": "skip", "reason": "C 档只借方法学，不获取"}

    if method == "pip":
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "pip", "install", "--target", str(dest),
               "--quiet", name]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode == 0 and not _looks_installed(dest):
            # exit 0 但目标目录空 —— 视为失败（exit code 不可信）
            return {"name": name, "status": "fail", "target": str(dest),
                    "error": "pip 返回 0 但目标目录为空（假成功）",
                    "license": tool["license"],
                    "redistributable": tool["redistributable"]}
        if cp.returncode != 0:
            # 失败时清掉残留空目录，避免下次 --check 误判
            try:
                if dest.is_dir() and not any(dest.iterdir()):
                    dest.rmdir()
            except OSError:
                pass
        return {"name": name, "status": "ok" if cp.returncode == 0 else "fail",
                "target": str(dest), "license": tool["license"],
                "redistributable": tool["redistributable"],
                "stderr_tail": cp.stderr[-400:] if cp.returncode else ""}

    if method == "git":
        if dest.exists():
            return {"name": name, "status": "exists", "target": str(dest)}
        cp = subprocess.run(["git", "clone", "--depth", "1", tool["repo"], str(dest)],
                            capture_output=True, text=True)
        return {"name": name, "status": "ok" if cp.returncode == 0 else "fail",
                "target": str(dest), "license": tool["license"],
                "redistributable": tool["redistributable"],
                "stderr_tail": cp.stderr[-400:] if cp.returncode else ""}

    if method == "download":
        import urllib.request
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / Path(tool["url"]).name
        try:
            urllib.request.urlretrieve(tool["url"], out)  # noqa: S310
        except Exception as e:  # noqa: BLE001
            return {"name": name, "status": "fail", "error": repr(e)}
        return {"name": name, "status": "ok", "target": str(out),
                "license": tool["license"], "redistributable": tool["redistributable"],
                "note": "压缩包，请解压后使用"}

    return {"name": name, "status": "unsupported", "method": method}


def main() -> int:
    ap = argparse.ArgumentParser(description="外部工具获取器")
    ap.add_argument("--list", action="store_true", help="列出全部工具与许可")
    ap.add_argument("--check", action="store_true", help="检查已获取状态")
    ap.add_argument("--fetch", metavar="NAME", help="获取指定工具")
    ap.add_argument("--fetch-all", action="store_true")
    ap.add_argument("--tier", choices=["A", "B", "C"], default=None,
                    help="配合 --fetch-all 限定档位")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.list or not (args.check or args.fetch or args.fetch_all):
        report = {
            "ok": True,
            "tools_root": str(TOOLS_ROOT),
            "note": "工具装到 skill 目录之外，不随 skill 分发；GPL 系仅作对拍",
            "n_tools": len(TOOLS),
            "by_tier": {
                tier: [{"name": t["name"], "license": t["license"],
                        "method": t["method"], "purpose": t["purpose"],
                        "redistributable": t["redistributable"]}
                       for t in TOOLS if t["tier"] == tier]
                for tier in ("A", "B", "C")
            },
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            for t in TOOLS:
                print(f"  [{t['tier']}] {t['name']:12s} {t['license']:22s} "
                      f"{'可分发' if t['redistributable'] else '不可分发'}  {t['purpose'][:40]}")
        return 0

    if args.check:
        rows = []
        for t in TOOLS:
            if t["method"] == "method":
                continue
            dest = _target_for(t["name"])
            rows.append({"name": t["name"], "tier": t["tier"], "license": t["license"],
                         "installed": _looks_installed(dest),
                         "path": str(dest) if _looks_installed(dest) else None})
        rep = {"ok": True, "tools_root": str(TOOLS_ROOT),
               "summary": f"{sum(1 for r in rows if r['installed'])}/{len(rows)} 已获取",
               "tools": rows}
        if args.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
        else:
            for r in rows:
                print(f"  [{'OK  ' if r['installed'] else 'MISS'}] {r['name']:12s} {r['license']}")
        return 0

    targets = []
    if args.fetch:
        if args.fetch not in BY_NAME:
            # 协议一致性：所有分支都必须给 results 键，否则下游统一读
            # rep["results"] 时会在「未知工具」这一条路径上 KeyError。
            print(json.dumps({"ok": False, "error": f"未知工具 {args.fetch}",
                              "results": [],
                              "known": sorted(BY_NAME)},
                             ensure_ascii=False, indent=2), file=sys.stdout)
            return 1
        targets = [BY_NAME[args.fetch]]
    elif args.fetch_all:
        targets = [t for t in TOOLS if (args.tier is None or t["tier"] == args.tier)]

    results = [do_fetch(t) for t in targets]
    ok = all(r["status"] in ("ok", "exists", "skip") for r in results)
    rep = {
        "ok": ok,
        "tools_root": str(TOOLS_ROOT),
        "summary": f"处理 {len(results)} 个："
                   + ", ".join(f"{r['name']}={r['status']}" for r in results),
        "results": results,
        "license_warning": "标记 redistributable=false 的工具不得随本 skill 分发",
    }
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print(rep["summary"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
