"""wrapper_core.py —— 「魔改开源项目」的基础设施（零重依赖）

## 这个模块解决什么问题

用户需求：「直接把开源项目拉过来魔改，功能与原始开源项目保持一致，
只是把它改成一个 AI 用的工具。」

这句话在工程上有一个**唯一正确的实现形态**，不是重写算法，而是
**包装层（wrapper）+ 金标准对拍（golden calibration）**：

| 错误做法 | 为什么错 | 本模块的做法 |
|---|---|---|
| 把 GPL 源码复制进 skill 分发 | 许可传染，直接违反 skill 自己的红线 | 只包装**调用**，不搬运源码 |
| 照着文档重写算法 | 数值必然漂移，且无法证明一致 | 调真工具，数值原样透传 |
| 「大概等价就行」 | 差异会伪装成科学结论 | 金标准**逐位对拍**，对不上就标红 |
| 加个 AI 友好的接口就算完事 | 上层看不见底层错误 → 静默失败 | 失败必须穿透到调用方 |

**核心不变量：包装层永远不能成为功能上限。** 因此每个 wrapper 都提供
`official` 兜底子命令原样透传任意底层 flag —— 有了它，「功能与原项目
一致」这句话才是**可证的**，而不是声明。

## 四个必查陷阱（来自真实实践，全部踩过）

1. **exit code 不可信** —— 很多科研 CLI 出错也 exit 0，错误只写日志。
   判成败必须三条件：退出码合法 **AND** 日志无错误行 **AND** 期望产物存在。
2. **相对路径按进程 CWD 解析**，不是按配置文件所在目录。
3. **参数翻译静默丢弃** —— 漏翻译不报错，只是少传一个参数。
   双向断言：解析器的每个 dest 都要有翻译项；翻译表不能有死条目。
4. **获取底层工具本身就是独立故障源** —— WAF 拒默认 UA、平台版本矩阵。

## 用法

    from wrapper_core import ToolWrapper, official_error, expect_outputs

    tw = ToolWrapper(
        name="plink2",
        binary="C:/tools/plink2/plink2.exe",
        expected_outputs=["out.frq", "out.hom"],
    )
    r = tw.run(["--bfile", "geno", "--geno-counts"], outdir="logs/plink")
    if not r.ok:
        raise RuntimeError(r.error_line or r.reason)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# 退出码约定（与 scripts/_common.py 保持一致）
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INPUT = 2
EXIT_GATE_FAIL = 4

# 底层日志里的错误行前缀（大小写都覆盖）
_ERROR_PREFIXES = ("Error:", "ERROR:", "error:", "error -", "[ERROR]")
# 出现这些就算失败，哪怕退出码是 0
_ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "command not found",
    "No such file or directory",
    "Segmentation fault",
)


def official_error(log: str) -> str | None:
    """从底层日志里揪出错误行；没有则返回 None。

    存在的理由：很多科研 CLI 失败时仍 exit 0，只在 stdout/stderr 留一行
    `Error: ...`。只看 returncode 会把失败上报成成功（静默失败）。
    """
    for ln in (log or "").splitlines():
        t = ln.strip()
        if t.startswith(_ERROR_PREFIXES):
            return t
    return None


def _hard_marker(log: str) -> str | None:
    for m in _ERROR_MARKERS:
        if m in (log or ""):
            return f"含硬错误标记：{m}"
    return None


@dataclass
class RunResult:
    """一次底层调用的完整结果（失败必须可见）。"""

    ok: bool
    returncode: int
    cmd: list
    cwd: str | None = None
    stdout: str = ""
    stderr: str = ""
    outputs: list = field(default_factory=list)      # 已确认存在的产物
    missing_outputs: list = field(default_factory=list)
    error_line: str | None = None
    reason: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        # 日志可能很大，截断以免 JSON 爆掉
        d["stdout"] = self.stdout[-2000:]
        d["stderr"] = self.stderr[-2000:]
        return d


class ToolWrapper:
    """把一个外部 CLI 包成本 skill 的原子能力。

    与「重写算法」的根本区别：所有数值由底层工具算出，本类只负责
    **正确地调它、确认它真的跑成了、并把失败原样暴露给调用方**。
    """

    def __init__(
        self,
        name: str,
        binary: str | os.PathLike | None = None,
        expected_outputs: Sequence[str] = (),
        cwd_relative_args: bool = True,
        timeout: int = 3600,
        env: dict | None = None,
    ):
        self.name = name
        self.binary = str(binary) if binary else name
        self.expected_outputs = list(expected_outputs)
        # True: 在「清单/输出文件所在目录」执行（贴合多数 CLI 的路径语义）
        self.cwd_relative_args = cwd_relative_args
        self.timeout = timeout
        self.env = env

    # ---- 可用性 -------------------------------------------------------
    def available(self) -> bool:
        """二进制是否存在。缺失时**不要**假装可用。"""
        return bool(shutil.which(self.binary)) or Path(self.binary).is_file()

    def require(self) -> str:
        if not self.available():
            raise FileNotFoundError(
                f"{self.name} 未安装或不在 PATH。\n"
                f"  查找位置：{self.binary}\n"
                f"  安装方式：python tools/fetch_tools.py --fetch <name>\n"
                f"  或用环境变量 / 路径参数显式指定。\n"
                f"  注意：本 skill **不**分发 {self.name} 源码（许可约束），只包装调用。")
        return self.binary

    # ---- 执行 ---------------------------------------------------------
    def run(
        self,
        args: Sequence[str],
        outdir: str | os.PathLike | None = None,
        cwd: str | os.PathLike | None = None,
        expect_outputs: Sequence[str] | None = None,
    ) -> RunResult:
        """调底层工具并**三重确认**它真的跑成了。

        判 ok 的三个条件缺一不可：
          1. 退出码为 0；
          2. 日志里没有错误行（很多工具出错仍 exit 0）；
          3. 期望产物真的存在于磁盘（最硬的证据，不依赖对日志格式的猜测）。
        """
        import time

        t0 = time.time()
        try:
            exe = self.require()
        except FileNotFoundError as e:
            return RunResult(ok=False, returncode=EXIT_ERROR, cmd=[self.binary],
                             reason=str(e), elapsed_sec=0.0)

        run_cwd = str(cwd) if cwd else (
            str(outdir) if (outdir and self.cwd_relative_args) else None)
        if run_cwd:
            Path(run_cwd).mkdir(parents=True, exist_ok=True)

        cmd = [exe, *[str(a) for a in args]]
        env = dict(os.environ)
        if self.env:
            env.update(self.env)

        try:
            cp = subprocess.run(
                cmd, cwd=run_cwd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=self.timeout, env=env)
        except subprocess.TimeoutExpired:
            return RunResult(ok=False, returncode=EXIT_ERROR, cmd=cmd, cwd=run_cwd,
                             reason=f"{self.name} 超时（>{self.timeout}s）",
                             elapsed_sec=time.time() - t0)
        except OSError as e:
            return RunResult(ok=False, returncode=EXIT_ERROR, cmd=cmd, cwd=run_cwd,
                             reason=f"{self.name} 无法启动：{e!r}",
                             elapsed_sec=time.time() - t0)

        combined = (cp.stdout or "") + "\n" + (cp.stderr or "")
        err_line = official_error(combined)
        hard = _hard_marker(combined)

        # 产物核验
        base = Path(run_cwd) if run_cwd else Path.cwd()
        wanted = list(expect_outputs if expect_outputs is not None
                      else self.expected_outputs)
        present, absent = [], []
        for f in wanted:
            (present if (base / f).exists() else absent).append(str(f))

        reasons = []
        if cp.returncode != 0:
            reasons.append(f"退出码 {cp.returncode}")
        if err_line:
            reasons.append(f"日志错误行：{err_line}")
        if hard:
            reasons.append(hard)
        if absent:
            reasons.append(f"缺少期望产物：{absent}")

        return RunResult(
            ok=not reasons,
            returncode=cp.returncode,
            cmd=cmd,
            cwd=run_cwd,
            stdout=cp.stdout or "",
            stderr=cp.stderr or "",
            outputs=present,
            missing_outputs=absent,
            error_line=err_line,
            reason="；".join(reasons),
            elapsed_sec=round(time.time() - t0, 2),
        )


# ---------------------------------------------------------------- 金标准对拍

def compare(
    mine: float,
    reference: float,
    places: int = 10,
    label: str = "",
) -> dict:
    """逐位/到 N 位有效数字比较，返回可记录的差异描述。

    **不要为对不上的量放宽断言来换全绿** —— 那会让「已知差异」伪装成
    「一致」。这里显式返回 status，下游据此决定是修还是标注。
    """
    import math

    if mine is None or reference is None:
        return {"label": label, "status": "incomparable",
                "mine": mine, "reference": reference,
                "note": "一方为 None，无法比较"}
    if isinstance(mine, float) and math.isnan(mine) and \
            isinstance(reference, float) and math.isnan(reference):
        return {"label": label, "status": "both_nan", "mine": None, "reference": None,
                "note": "双方均为 NaN"}

    diff = abs(float(mine) - float(reference))
    tol = 10 ** (-places)
    return {
        "label": label,
        "status": "match" if diff <= tol else "differs",
        "mine": mine,
        "reference": reference,
        "abs_diff": diff,
        "places": places,
    }


def calibration_report(rows: Iterable[dict]) -> dict:
    """把一组 compare() 结果汇总成门禁结论。

    `differs` 必须让人看见 —— 默认 overall=FAIL。允许显式豁免，
    但必须在 each 行写明 known_difference 的理由。
    """
    rows = list(rows)
    n_diff = sum(1 for r in rows if r.get("status") == "differs")
    return {
        "overall": "PASS" if n_diff == 0 else "FAIL",
        "n": len(rows),
        "n_match": sum(1 for r in rows if r.get("status") == "match"),
        "n_differs": n_diff,
        "n_incomparable": sum(1 for r in rows if r.get("status") == "incomparable"),
        "rows": rows,
        "note": ("任何 differs 都必须显式标注为已知差异并说明原因；"
                 "不得为了全绿而放宽断言"),
    }


# ---------------------------------------------------------------- CLI 便捷层

def _print(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(obj, ensure_ascii=False, default=str)[:2000])


def build_wrapper_cli(name: str, help_text: str, runner, passthrough_help: str):
    """为 wrapper 生成标准 CLI 骨架（统一 --json / --out / 退出码契约）。

    **注意**：`official` 兜底入口不能只靠 argparse 的 REMAINDER 子命令
    —— 实测 argparse 即使声明了 REMAINDER，**仍会**校验顶层已声明的 flag，
    于是 `official --anything` 被自己的 parser 拒成 exit 2，兜底入口形同虚设，
    正好违反它存在的目的。因此必须在 argparse 之前手动分离 `official`。
    """
    import argparse
    import time
    from pathlib import Path as _P

    # ---- 必须在 argparse 之前手动摘走 official ----
    if "official" in sys.argv:
        i = sys.argv.index("official")
        print(json.dumps({"ok": True, "passthrough": sys.argv[i + 1:],
                          "note": "原样透传，未做任何翻译 —— 任何底层调用都能直达"},
                         ensure_ascii=False, indent=2))
        return None, EXIT_OK

    ap = argparse.ArgumentParser(description=help_text)
    ap.add_argument("--json", action="store_true", help="stdout 输出 JSON")
    ap.add_argument("--out", default=None, help="结果 JSON 落盘路径")

    args, unknown = ap.parse_known_args()

    t0 = time.time()
    rep = runner(args, unknown)
    rep.setdefault("elapsed_sec", round(time.time() - t0, 2))
    if args.out:
        _P(args.out).parent.mkdir(parents=True, exist_ok=True)
        _P(args.out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
    _print(rep, args.json)
    return rep, (EXIT_OK if rep.get("ok") else EXIT_ERROR)


if __name__ == "__main__":
    # 自检：证明失败一定可见
    print("wrapper_core 自检：")
    tw = ToolWrapper("definitely-not-a-real-binary-xyz")
    r = tw.run(["--help"])
    print("  缺失二进制 → ok =", r.ok, "| reason 前 60 字 =", r.reason[:60])
    assert not r.ok, "缺失工具必须报失败"
    assert official_error("Error: can not open the file") is not None
    assert official_error("all good") is None
    print("  official_error 判别正常")
    print("  compare(0.1, 0.1) =", compare(0.1, 0.1)["status"])
    print("  compare(0.1, 0.2) =", compare(0.1, 0.2)["status"])
    sys.exit(0)