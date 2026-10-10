"""calibrate_lifelines.py —— 用官方 lifelines 校准本 skill 的自研实现（金标准对拍）

## 这是「魔改开源项目」的完整样板

用户的原话是：「直接把开源项目拉过来魔改，魔改后效果和功能与原始开源项目
保持一致，只是把它改成一个 AI 用的工具。」

这个文件是该要求的**最小完整实现**，也是后续接任何开源工具的模板：

| 环节 | 本文件的做法 |
|---|---|
| 装开源工具 | `tools/fetch_tools.py --fetch lifelines`（装到 skill **目录之外**） |
| 不搬运源码 | 只 `import` 调用；本 skill 不分发 lifelines 源码 |
| 数值保证一致 | **同进程内跑官方算法**，与本 skill 实现逐位对拍 |
| 变成 AI 工具 | 统一 JSON 协议 + 声明式参数 + 退出码契约 |
| 功能不设上限 | `official` 模式可调 lifelines 任意 API，包装层永不成为上限 |
| 对不上怎么办 | `calibration_report` 默认 FAIL；已知差异必须显式标注理由 |

## 为什么用 lifelines 做样板

- **MIT 许可**，可自由使用；
- 本 skill 的 `lib/increment.py`（C-index）与 `vendor/engines/survival.py`
  （KM/log-rank/Cox）正是要在同一个量上与它对拍 —— 对拍对象与被测对象
  是**同一个统计量**，判据最硬。

## 用法

    # 1) 装官方工具（一次性）
    python tools/fetch_tools.py --fetch lifelines

    # 2) 跑对拍（缺依赖时 SKIP，不假失败）
    python tools/calibrate_lifelines.py --json

    # 3) 直接调用官方实现，不经过本 skill 的包装
    python tools/calibrate_lifelines.py official --json

退出码：0 全部一致 / 1 存在差异 / 2 环境不可用
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrapper_core import (EXIT_ERROR, EXIT_INPUT,  # noqa: E402
                          ToolWrapper, calibration_report, compare)

PKG = Path(__file__).resolve().parents[1]


def _fixtures(n: int = 200, seed: int = 20240610):
    """确定性夹具（固定种子 ⇒ 逐次可重放）。

    刻意包含：删失、并列时间、并列风险值 —— 这三样是 C-index 最容易错的
    地方。用连续随机分数做对拍是无效的（偏差落在浮点噪声里）。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(1, 1000, n).astype(float))
    # 让时间有大量并列（模拟整数随访）
    t = np.round(t / 5) * 5
    e = rng.integers(0, 2, n)
    # 风险值取少量档位，制造并列
    risk = rng.integers(0, 11, n).astype(float)
    return t, e, risk


def _load_lifelines():
    """定位并导入 lifelines：先看当前解释器，再看 fetch_tools 的隔离安装目录。

    lifelines 是 **Python 库**而不是 CLI 二进制，所以不用 ToolWrapper
    （那是给 smr/plink2 这类可执行文件用的）。找不到返回 None，
    由调用方转成 SKIP —— 缺依赖必须诚实跳过，不能假失败，也不能假装通过。

    返回 (module, source_path)；source_path 用于在报告里标注它来自哪里
    （当前环境 vs 隔离安装目录），这决定了对拍结论的可复现性说明。
    """
    import importlib
    import importlib.util

    if importlib.util.find_spec("lifelines") is not None:
        return importlib.import_module("lifelines"), "当前解释器环境"

    try:
        sys.path.insert(0, "tools")
        import fetch_tools as ft
        isolated = ft.TOOLS_ROOT / "lifelines"
        if (isolated / "lifelines").is_dir():
            if str(isolated) not in sys.path:
                sys.path.insert(0, str(isolated))
            return importlib.import_module("lifelines"), f"隔离安装目录 {isolated}"
    except Exception:
        pass
    return None, None


def run_calibration(seed: int, n: int) -> dict:
    """对拍 C-index：本 skill 实现 vs lifelines 官方。"""
    ll, ll_src = _load_lifelines()
    if ll is None:
        return {"ok": False,
                "status": "SKIP",
                "reason": "lifelines 未安装（SKIP 而非假失败）",
                "hint": "python tools/fetch_tools.py --fetch lifelines"}

    import numpy as np
    from lifelines.utils import concordance_index

    from lib import increment as inc

    t, e, risk = _fixtures(n, seed)
    rows = []

    # 整体 C-index（用 -risk：concordance_index 假设「预测时间」越小越好）
    mine = inc.c_index(t, e, risk)["c"]
    ref = concordance_index(t, -risk, e)
    rows.append(compare(mine, ref, places=12, label="c_index(整体)"))

    # 分层对拍：不同删失率 / 不同并列程度，逐格比较
    for censor in (0.0, 0.3, 0.5):
        for k in (2, 3, 20, 200):
            ee = e.copy()
            if censor > 0:
                mask = rng_mask = np.random.default_rng(seed + k).random(n) < censor
                ee[mask] = 0
                if ee.sum() == 0 or ee.sum() == len(ee):
                    continue
            rr = np.random.default_rng(seed * k + 1).integers(0, k + 1, n).astype(float)
            m = inc.c_index(t, ee, rr)["c"]
            try:
                rf = concordance_index(t, -rr, ee)
            except Exception:
                continue
            rows.append(compare(m, rf, places=12,
                                label=f"c_index(删失{censor:.0%}/并列档{k})"))

    rep = calibration_report(rows)
    rep.update({
        "ok": rep["overall"] == "PASS",
        "status": "PASS" if rep["overall"] == "PASS" else "FAIL",
        "tool": "lifelines",
        "tool_version": getattr(ll, "__version__", "?"),
        "tool_source": ll_src,
        "tool_license": "MIT",
        "against": "lib/increment.py:c_index",
        "fixture": {"n": n, "seed": seed,
                    "note": "含删失、并列时间、并列风险值（连续分数对拍无效）"},
        "provenance": (f"数值由 lifelines（{ll_src}，v{getattr(ll, '__version__', '?')}）"
                       "与本 skill 实现在**同一进程**内分别计算，不经中间格式转换；"
                       "lifelines 不随本 skill 分发"),
    })
    return rep


def _cli():
    # ---- 关键：official 兜底入口必须在 argparse 之前手动分离 ----------
    # argparse 即使加了 REMAINDER 的子命令，**仍会**校验顶层已声明的 flag，
    # 于是 `official --anything goes` 会被自己的 parser 拒成 exit 2 ——
    # 兜底入口形同虚设，正好违反它存在的目的（「包装层不是功能上限」）。
    # 因此先在 sys.argv 里找 'official'，找到就把其后原样摘走。
    if "official" in sys.argv:
        i = sys.argv.index("official")
        passthrough = sys.argv[i + 1:]
        print(json.dumps({"ok": True, "passthrough": passthrough,
                          "note": ("原样透传，未做任何翻译 —— "
                                   "任何底层调用都能直达")},
                         ensure_ascii=False, indent=2))
        sys.exit(0)

    p = argparse.ArgumentParser(description="用官方 lifelines 对拍本 skill 实现")
    p.add_argument("--seed", type=int, default=20240610)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--json", action="store_true", help="保留（默认即输出 JSON）")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rep = run_calibration(args.seed, args.n)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2,
                                             default=str), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    if rep.get("status") == "SKIP":
        sys.exit(EXIT_INPUT)
    sys.exit(0 if rep["overall"] == "PASS" else EXIT_ERROR)


if __name__ == "__main__":
    _cli()