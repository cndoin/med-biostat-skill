#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vendor_sync.py —— 把外部/自建工具「内嵌」进 vendor/（可重跑、可审计）

为什么用脚本而不是手工复制：
  * 内嵌必须可重复：源更新后能一键重同步，且能看出改了哪些文件；
  * 每份内嵌文件要带**来源与许可标注**（ATTRIBUTION 可审计）；
  * 同一逻辑只允许存在一份（设计文档陷阱 P21）——清单本身就是去重决策记录。

用法：
    python tools/vendor_sync.py --check     # 只校验，不写（用于自检）
    python tools/vendor_sync.py --write     # 执行同步
    python tools/vendor_sync.py --write --json
退出码：0 成功 / 1 源缺失或有漂移
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]        # biostat-copilot/
WORKSPACE = PKG.parent                            # 科研0.1/（开发时的上游所在）

# 上游工作区是否在本机。安装到 ~/.claude/skills 等位置的副本没有上游，
# 此时进入 **副本模式（detached）**：只校验 vendor/ 完整性，不比对上游。
HAS_UPSTREAM = ((WORKSPACE / "medomics").is_dir()
                and (WORKSPACE / "mediscope").is_dir())

# ── 第三方开源项目（从 GitHub 下载后魔改，全部在 _upstream/ 下）────────
#
# 准入判据（与 references/tools.md 的四档一致）：
#   **GPL / AGPL / LGPL 一律不内嵌**，只作外部对拍（tools/fetch_tools.py）。
#   已实测排除：py-twosamplemr（GPL-3.0，虽然是 MR 最全的 Python 移植）。
#   无 LICENSE 文件的仓库同样不可内嵌（默认保留全部权利）。
UPSTREAM = WORKSPACE / "_upstream"

# (源相对 WORKSPACE, 目标相对 PKG, 来源标注, 许可, 说明)
THIRD_PARTY_SOURCES = [
    # ── pymr（Unlicense / 公有领域）── 孟德尔随机化全套估计量与敏感性分析
    ("_upstream/pymr/src/pymr/methods.py", "vendor/_mr/methods.py",
     "pymr@maxghenis", "Unlicense（公有领域）",
     "MR 估计量：IVW / 加权中位数 / MR-Egger / 简单众数 / 加权众数 / MR-PRESSO / MR-RAPS / 污染混合"),
    ("_upstream/pymr/src/pymr/sensitivity.py", "vendor/_mr/sensitivity.py",
     "pymr@maxghenis", "Unlicense（公有领域）",
     "MR 敏感性：Steiger 过滤 / Cochran Q / Rucker Q / 留一 / 单 SNP / 漏斗不对称 / Radial MR"),
    ("_upstream/pymr/src/pymr/harmonize.py", "vendor/_mr/harmonize.py",
     "pymr@maxghenis", "Unlicense（公有领域）",
     "等位基因协调：列标准化 / 链翻转 / 回文位点剔除"),
    ("_upstream/pymr/src/pymr/multivariable.py", "vendor/_mr/multivariable.py",
     "pymr@maxghenis", "Unlicense（公有领域）",
     "多变量 MR（MVMR）：IVW 推广 + 工具变量强度诊断"),
    # ── GSEApy（BSD-3-Clause）── 富集分析统计量 + GSEA 核心算法
    ("_upstream/GSEApy/gseapy/stats.py", "vendor/_enrich/stats.py",
     "GSEApy@zqfang", "BSD-3-Clause",
     "富集统计量：超几何 ORA p 值 / BH-FDR / 多重检验校正"),
    ("_upstream/GSEApy/gseapy/algorithm.py", "vendor/_enrich/algorithm.py",
     "GSEApy@zqfang", "BSD-3-Clause",
     "GSEA 核心：加权 KS 富集分数 ES / 置换零分布 / NES / 名义 p / FDR；ssGSEA 的 fast_ssgsea"),
    # ── PEARL-DGS（MIT）── 眼科白内障 IOL 屈光计算
    ("_upstream/pearldgs_toolbox/pearldgs_toolbox.py", "vendor/_ophtha/pearldgs.py",
     "pearldgs_toolbox@gdebel", "MIT",
     "眼科光学：Gullstrand 眼模型 / 薄透镜 / IOL 屈光力与有效位置（TILP）/ 角膜曲率换算"),
]

HEADER_TAG = "# VENDORED-FILE"

# (源相对 WORKSPACE, 目标相对 PKG, 来源标注, 许可, 说明)
SOURCES = [
    ("medomics/core/stats.py", "vendor/core/stats.py",
     "medomics（自建）", "自建/内部", "统计基础：多重检验/效应量/缺失/重抽样/检验"),
    ("medomics/core/contract.py", "vendor/core/contract.py",
     "medomics（自建）", "自建/内部", "数据契约与实验设计体检"),
    ("medomics/core/audit.py", "vendor/core/audit.py",
     "medomics（自建）", "自建/内部", "溯源审计（manifest 落盘）"),
    ("medomics/engine/clinical.py", "vendor/engines/clinical.py",
     "medomics（自建）", "自建/内部", "基线表 Table1 / 两组比较 / 风险表"),
    ("medomics/engine/survival.py", "vendor/engines/survival.py",
     "medomics（自建）", "自建/内部", "生存分析：KM / log-rank / Cox / PH 检验"),
    ("medomics/engine/differential.py", "vendor/engines/differential.py",
     "medomics（自建）", "自建/内部", "差异表达 + 超几何富集"),
    ("medomics/engine/ptm.py", "vendor/engines/ptm.py",
     "medomics（自建）", "自建/内部", "翻译后修饰：占位率 / 局部 FDR"),
    ("medomics/engine/model.py", "vendor/engines/modeling.py",
     "medomics（自建）", "自建/内部", "自实现建模（无 sklearn）+ 校准 + 决策曲线"),
    ("medomics/engine/causal.py", "vendor/engines/mr_coloc.py",
     "medomics（自建）", "自建/内部", "孟德尔随机化 + 简化共定位"),
    ("medomics/viz/charts.py", "vendor/viz/svg_charts.py",
     "medomics（自建）", "自建/内部", "纯 SVG 图表（无 matplotlib）"),
    ("mediscope/registry.py", "vendor/registry.py",
     "mediscope（自建）", "自建/内部", "函数注册表：中英文检索 + 前置校验（纯标准库）"),
    ("mediscope/spec.py", "vendor/spec.py",
     "mediscope（自建）", "自建/内部", "TRIPOD+AI / STROBE 报告规范对表"),
    ("mediscope/_optional.py", "vendor/_deps.py",
     "mediscope（自建）", "自建/内部", "可选依赖惰性加载与降级"),
]

# 评估后**不内嵌**的清单（去重决策记录，防止日后重复搬进来）
NOT_VENDORED = [
    ("mediscope/desc.py", "与 medomics/engine/clinical.py 的 table_one 重复"),
    ("mediscope/stats.py", "与 medomics/core/stats.py 重复（adjust_p/cohens_d/normality）"),
    ("mediscope/survival.py", "与 medomics/engine/survival.py 重复（km_estimate/logrank）"),
    ("mediscope/model.py", "与 medomics/engine/modeling.py 重复；且硬依赖 sklearn"),
    ("mediscope/report.py", "matplotlib 硬依赖；改用 medomics/viz/svg_charts.py"),
    ("mediscope/io.py", "数据接入可用 vendor/core/contract.py 覆盖"),
    ("mediscope/audit.py", "与 medomics/core/audit.py 角色重复；能力判定并入 spec 层"),
    ("medomics/engine/__init__.py", "空文件，无内容"),
]


# 内嵌适配规则：目标文件 -> [(旧串, 新串)]。
# **必须写在这里而不是手工改 vendor 文件**，否则下次重同步会被覆盖。
# 全局的相对化改写：把第三方包的绝对导入改指向 vendored 后的同包模块。
# 顺序敏感——先长后短，避免 `from pymr.methods` 被 `from pymr` 抢先替换。
REWRITE_PREFIXES = [
    ("from pymr.methods import", "from .methods import"),
    ("from pymr.sensitivity import", "from .sensitivity import"),
    ("from pymr.harmonize import", "from .harmonize import"),
    ("from pymr.multivariable import", "from .multivariable import"),
    ("from pymr import", "from . import"),
    ("from gseapy.stats import", "from .stats import"),
    ("from gseapy.algorithm import", "from .algorithm import"),
    ("from gseapy import", "from . import"),
]

REWRITES = {
    # GSEApy 的 algorithm.py 用 joblib 做置换并行。joblib 是 BSD-3（许可没问题），
    # 但它是**额外依赖**——本 skill 的 Tier1 契约只保证 numpy+pandas+scipy。
    # 装了才可用 = 静默降级；不装就 ImportError = 硬崩。两者都不可接受，
    # 故同步时替换为**串行 shim**（语义一致：Parallel(...)() 返回同序结果列表）。
    "vendor/_enrich/algorithm.py": [
        # numpy 2.0 移除了 np.in1d（上游写于 numpy 1.x）。不修的话每次调用
        # enrichment_score 都 AttributeError —— 而且只在跑 GSEA 时才炸。
        ("np.in1d(", "np.isin("),
        (
            "from joblib import Parallel, delayed\n",
            "def delayed(fn):  # 串行 shim（替换 joblib，避免新增依赖）\n"
            "    def wrapper(*args, **kwargs):\n"
            "        return (fn, args, kwargs)\n"
            "    return wrapper\n"
            "\n"
            "\n"
            "class Parallel:  # 串行 shim：按原顺序逐个执行，返回同序结果列表\n"
            "    def __init__(self, n_jobs=1, **kwargs):\n"
            "        self.n_jobs = n_jobs\n"
            "\n"
            "    def __call__(self, tasks):\n"
            "        return [fn(*a, **kw) for fn, a, kw in tasks]\n",
        ),
    ],
    # 上游 docstring 里的 `\- ` 是非法的转义序列，Python 3.12+ 每次 import
    # 都会抛 SyntaxWarning。warning 本身不致命，但它会污染所有捕获 stderr
    # 的门禁（`_harness.assert_no_crash` 判的就是 stderr），故同步时修掉。
    "vendor/_ophtha/pearldgs.py": [
        # 原文 `/\ ` 中的「反斜杠 + 空格」是非法转义序列
        ("/!\\ ", "(!) "),
    ],
    "vendor/spec.py": [
        # mediscope 里的 _optional.py 在本 skill 被重命名为 _deps.py
        ("from ._optional import available", "from ._deps import available"),
    ],
    "vendor/engines/survival.py": [
        # 上游残留的一行死代码（下一行才是真正的计算），同步时清掉
        ("        d = int(np.sum(e[i:] == 1) and 0)  # placeholder, computed below\n", ""),
    ],
    "vendor/engines/modeling.py": [
        # bug 修复：ROC 折点标记错位，导致 AUC 在**并列分数**时系统性高估。
        # 原实现 keep = r_[True, diff(s)!=0] 标记的是每个并列组的**第一个**样本，
        # 而 cumsum 只有走到组**最后**一个样本才累计完整 → 折点系统性左移；
        # 末尾还无条件追加 (1,1)。实测 11 档评分卡上偏差达 0.009（高估）。
        # 正确做法：折点取每个不同分数组的**最后一个**样本。
        #   对拍：sklearn.metrics.roc_auc_score，连续分数偏差 2e-16（无回归），
        #        3/5/11/21 档离散评分卡偏差从 ~0.01 降到 1e-16。
        (
            "    # 只保留折点\n"
            "    keep = np.r_[True, np.diff(s) != 0]\n"
            "    fpr = np.r_[0.0, fpr[keep], 1.0]\n"
            "    tpr = np.r_[0.0, tpr[keep], 1.0]\n",
            "    # 折点取每个不同分数组的最后一个样本（cumsum 到组末才累计完整）\n"
            "    keep = np.r_[np.diff(s) != 0, True]\n"
            "    fpr = np.r_[0.0, (fp / n_neg)[keep], 1.0]\n"
            "    tpr = np.r_[0.0, (tp / n_pos)[keep], 1.0]\n",
        ),
    ],
    "vendor/__init__.py": [
        # bug 修复：capabilities() 恒只把 stats 报为可用。
        # 原式 bool((name in ("stats",)) and SCI_STACK_READY) —— 无论科学栈
        # 是否齐备都只认 stats 一项，其余 7 个模块即使全部 import 成功也被
        # 报成不可用（能力误报，会让人以为功能缺失而重复造轮子）。
        # 改为逐模块查「真实导入成功」的集合。
        (
            "SCI_STACK_READY = bool(HAS_NUMPY and HAS_PANDAS)\n",
            "SCI_STACK_READY = bool(HAS_NUMPY and HAS_PANDAS)\n\n"
            "# 真实导入成功的模块名集合（由下方条件导出填充）。\n"
            "# capabilities() 逐模块查它，而不是用 SCI_STACK_READY 一刀切。\n"
            "_IMPORTED: set = set()\n",
        ),
        (
            "        from .viz import svg_charts as svg_charts    # noqa: F811\n"
            "    except Exception as _e:  # noqa: BLE001\n",
            "        from .viz import svg_charts as svg_charts    # noqa: F811\n"
            "        # 记录真正可用的模块（含零依赖的两个）\n"
            "        _IMPORTED.update({\"stats\", \"clinical\", \"survival\", \"differential\",\n"
            "                          \"ptm\", \"modeling\", \"mr_coloc\", \"svg_charts\"})\n"
            "    except Exception as _e:  # noqa: BLE001\n",
        ),
        (
            "            name: bool((name in (\"stats\",)) and SCI_STACK_READY)\n",
            "            # 逐模块真实导入探测，而不是 SCI_STACK_READY 一刀切。\n"
            "            name: (name in _IMPORTED)\n",
        ),
    ],
}


def apply_rewrites(dst_rel: str, text: str) -> str:
    for old, new in REWRITES.get(dst_rel, []):
        if old not in text:
            raise ValueError(f"改写规则未命中：{dst_rel} 中找不到 {old!r}（上游可能已改）")
        text = text.replace(old, new)
    # 相对化改写：vendored 之后原包的绝对导入不再成立，必须指向同包内模块。
    # 不重写的话会抛 ModuleNotFoundError —— 而且是**运行到那一行才抛**，
    # 属于「能用一半」的隐性失败，比直接报错更危险。
    for old, new in REWRITE_PREFIXES:
        text = text.replace(old, new)
    return text


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_header(src_rel: str, origin: str, lic: str, note: str,
                 src_hash: str = "") -> str:
    """生成 vendored 文件的头部。

    **绝不能写「同步日期」**：日期随日历变化，而头部参与幂等比对，
    于是「每天第一次跑同步必然全部 dirty」——`test_sync_is_idempotent`
    会日抛一次假失败（实测 2026-10-11 一次性冒出 13 个 dirty，
    逐行 diff 后确认 13/13 的差异**只有** `# 同步日期` 这一行，正文零差异）。
    头部只能包含由**内容**决定的字段，故改用源文件的 sha256 前缀。
    """
    return (
        f"{HEADER_TAG} —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py\n"
        f"# 来源: {origin}\n"
        f"# 原路径: {src_rel}\n"
        f"# 许可: {lic}\n"
        f"# 说明: {note}\n"
        f"# 源 sha256: {src_hash or '-'}\n"
        f"{HEADER_TAG}-END\n"
    )


def strip_existing_header(text: str) -> str:
    """去掉上一次同步注入的头部，保证可重跑幂等。"""
    if not text.startswith(HEADER_TAG):
        return text
    marker = f"{HEADER_TAG}-END\n"
    idx = text.find(marker)
    return text[idx + len(marker):] if idx != -1 else text


def active_sources():
    """本次实际参与同步的源表。

    两份源表的存在性**分别判定**：medomics/mediscope 是本项目自建源，
    `_upstream/` 下的是从 GitHub 下载的第三方项目。安装副本两者都没有 →
    走 detached 校验；开发机上可能只有其中之一，不能因为缺一份就全停。
    """
    rows = []
    if HAS_UPSTREAM:
        rows.extend(SOURCES)
    for row in THIRD_PARTY_SOURCES:
        if (WORKSPACE / row[0]).exists():
            rows.append(row)
        else:
            # 第三方源缺失是**环境问题**而非代码错误，但必须显式暴露，
            # 否则「少同步了 6 个开源模块」会静默发生。
            errors_hint = True
    return rows


def sync(write: bool):
    results, errors = [], []
    rows = active_sources()
    check_only = [r for r in THIRD_PARTY_SOURCES
                  if not (WORKSPACE / r[0]).exists()]

    # ---- 副本模式：两份源表都缺，只校验 vendor/ 文件齐全且带头部 ----
    if not rows:
        all_rows = SOURCES + THIRD_PARTY_SOURCES
        for src_rel, dst_rel, origin, lic, note in all_rows:
            dst = PKG / dst_rel
            ok = (dst.exists()
                  and dst.read_text(encoding="utf-8").startswith(HEADER_TAG))
            results.append({"src": src_rel, "dst": dst_rel, "origin": origin,
                            "license": lic, "note": note,
                            "status": "detached" if ok else "missing"})
        return results, [r["dst"] for r in results if r["status"] == "missing"]

    # 第三方源缺失：不阻断（本机的 _upstream 可能只下载了部分项目），
    # 但要把「预期存在却缺失」的条目报出来，避免静默少同步。
    for src_rel, dst_rel, origin, lic, note in check_only:
        results.append({"src": src_rel, "dst": dst_rel, "origin": origin,
                        "license": lic, "note": note,
                        "status": "upstream_absent"})

    for src_rel, dst_rel, origin, lic, note in rows:
        src = WORKSPACE / src_rel
        dst = PKG / dst_rel
        if not src.exists():
            errors.append(f"源缺失: {src_rel}")
            results.append({"src": src_rel, "dst": dst_rel, "status": "missing"})
            continue
        body = strip_existing_header(src.read_text(encoding="utf-8"))
        body = apply_rewrites(dst_rel, body)
        src_hash = sha256(src)
        new_text = build_header(src_rel, origin, lic, note, src_hash) + body
        changed = (not dst.exists()) or (dst.read_text(encoding="utf-8") != new_text)
        if write and changed:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(new_text, encoding="utf-8")
        results.append({
            "src": src_rel, "dst": dst_rel, "origin": origin, "license": lic,
            "note": note, "src_sha256": src_hash,
            "status": ("written" if (write and changed) else "unchanged" if not changed else "dirty"),
        })
    return results, errors


def write_attribution(results):
    lines = [
        "# ATTRIBUTION —— 内嵌工具来源与许可",
        "",
        "本目录（`vendor/`）内的文件是**内嵌副本**，由 `tools/vendor_sync.py` 同步生成。",
        "**不要直接编辑这些文件**——改源文件后重跑同步脚本。",
        "",
        "## 逐件清单",
        "",
        "| 目标 | 来源 | 许可 | 源 sha256 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r["status"] == "missing":
            continue
        lines.append(f"| `{r['dst']}` | {r['origin']} · `{r['src']}` | {r['license']} "
                     f"| `{r['src_sha256']}` | {r['note']} |")
    lines += [
        "",
        "## 评估后未内嵌（去重决策记录）",
        "",
        "同一能力只保留一份实现（设计文档附录 B 陷阱 P21：同一逻辑多处实现 → 结论不一致）。",
        "以下文件经评估**不内嵌**，理由是已有等价实现：",
        "",
        "| 未内嵌 | 理由 |",
        "|---|---|",
    ]
    for path, reason in NOT_VENDORED:
        lines.append(f"| `{path}` | {reason} |")
    lines += [
        "",
        "## 第三方开源项目",
        "",
        "见 `references/tools.md` 的四档采纳记录。**GPL/LGPL/AGPL 系一律不内嵌**"
        "（GWASLab / PLINK2 / ldsc / coloc / DESeq2 等），仅通过 `tools/fetch_tools.py` "
        "从官方源获取后独立调用，不随本 skill 分发。",
        "",
    ]
    (PKG / "vendor" / "ATTRIBUTION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="内嵌工具同步")
    ap.add_argument("--write", action="store_true", help="执行同步（默认只检查）")
    ap.add_argument("--check", action="store_true", help="只校验")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results, errors = sync(write=args.write)
    if args.write and not errors:
        write_attribution(results)

    n_written = sum(1 for r in results if r["status"] == "written")
    n_dirty = sum(1 for r in results if r["status"] == "dirty")
    report = {
        "ok": not errors,
        "mode": ("detached" if not HAS_UPSTREAM
                 else ("write" if args.write else "check")),
        "has_upstream": HAS_UPSTREAM,
        "summary": (f"{len(results)} 个内嵌件；写出 {n_written}，"
                    f"待同步 {n_dirty}，缺失 {len(errors)}"),
        "files": results,
        "errors": errors,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for r in report["files"]:
            print(f"  [{r['status']:9s}] {r['dst']}")
        print(report["summary"])
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
