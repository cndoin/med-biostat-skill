#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""_harness.py —— 测试共享工具：调用脚本并解析其 stdout JSON。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SCRIPTS = PKG / "scripts"
PY = sys.executable


def run(script: str, *args, timeout: int = 180):
    """运行 scripts/<script>，返回 (exit_code, stdout_json_or_None, stderr)。"""
    cmd = [PY, str(SCRIPTS / script), *[str(a) for a in args]]
    cp = subprocess.run(cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=timeout)
    data = None
    try:
        data = json.loads(cp.stdout)
    except Exception:
        data = None
    return cp.returncode, data, cp.stderr


class CrashedAssertion(AssertionError):
    """脚本以 traceback 崩溃（而非干净的失败）。"""


def assert_no_crash(stderr: str, code: int, script: str = "") -> None:
    """断言脚本**没有**以 Python traceback 结束。

    为什么需要它：Python 未捕获异常的退出码恰好是 **1**。而各脚本的
    失败退出码约定是 0/2/3/4，只要断言写成 `exit != 0` 或
    `exit in (0,2,3,4)`，**崩溃就会被当成合法失败放过** —— 实测 14 个
    bug（IndexError / FileNotFoundError / AttributeError / OverflowError）
    全是这样溜过去的。

    判据：stderr 里出现 Python traceback 标记即视为崩溃。
    注意 argparse 的用法错误（exit 2）不算崩溃，它走的是 argparse 自己的
    消息，没有 traceback 标记。
    """
    markers = ("Traceback (most recent call last)", "SyntaxError",
               "IndentationError", "AttributeError:", "IndexError:",
               "KeyError:", "TypeError:", "ValueError:", "OverflowError",
               "FileNotFoundError", "ZeroDivisionError", "NameError:")
    hit = [m for m in markers if m in (stderr or "")]
    assert not hit, (
        f"{script} 崩溃了（exit={code}）而不是干净失败；"
        f"stderr 含 {hit}。\n--- stderr ---\n{(stderr or '')[:600]}")


def run_clean(script: str, *args, timeout: int = 180):
    """run() + 崩溃断言。测试里优先用这个。"""
    code, data, err = run(script, *args, timeout=timeout)
    assert_no_crash(err, code, script)
    return code, data, err


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def has(mod: str) -> bool:
    """某个第三方包是否可用（用于 @skipUnless 对拍金标准）。"""
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# ── 带科学栈的 interpreter 探测（唯一裁决点，P21）────────────────────
#
# 为什么单独抽出来：早先 test_lib.py 与 test_vendor.py 各写了一份
# 只搜 `PKG.parent/_venv_ref*` 的候选表，两处逻辑必然漂移。
# 更糟的是跳过理由写成「本机无带 numpy 的 interpreter」—— 实测本机
# 裸解释器确实没有 numpy，但真实原因是「包旁边没找到 venv」，
# 把「没找到」说成「不存在」就是典型的失败被伪装成别的形状。
#
# 探测顺序（先显式后隐式，命中即止）：
#   1. 环境变量 BIOSTAT_SCI_PYTHON（CI / 用户显式指定，最高优先级）
#   2. PKG.parent 下任意 */Scripts/python.exe（仓库旁建的参考 venv）
#   3. 用户级 WorkBuddy venv（安装副本在 ~/.claude/skills 等处，
#      旁边没有 venv；不加这一条，仓库外验证会让本可运行的用例集体跳过）
#   4. 当前解释器 sys.executable（若它自己就装了科学栈）
#
# `mods` 是必需的依赖集合：lib/ 只要 numpy，vendor/ 要 numpy+pandas+scipy。
# 必须保留这个差异 —— 若统一按「能 import numpy」判定，只装了 numpy 的
# 环境会把 vendor 测试从「该跳过」变成「真失败」，那是门禁噪音不是信号。
_SCI_CACHE: dict = {}


def _sci_ok(executable: Path, mods: tuple[str, ...]) -> bool:
    """某个解释器是否真的能 import 全部 mods。结果按 (路径, mods) 缓存。"""
    key = (str(executable), mods)
    if key in _SCI_CACHE:
        return _SCI_CACHE[key]
    ok = False
    try:
        cp = subprocess.run([str(executable), "-c",
                             "import " + ", ".join(mods)],
                            capture_output=True, timeout=60)
        ok = cp.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    _SCI_CACHE[key] = ok
    return ok


def find_sci_python(mods: tuple[str, ...] = ("numpy",)):
    """返回第一个能 import 全部 mods 的解释器路径；找不到返回 None。

    跳过理由必须由调用方按本函数的实际语义书写：
    「未在 <候选范围> 内发现带 X 的解释器」，而**不是**「本机没有」。
    """
    candidates: list[Path] = []

    env = os.environ.get("BIOSTAT_SCI_PYTHON")
    if env:
        candidates.append(Path(env))

    parent = PKG.parent
    try:
        for scripts in sorted(parent.glob("*/Scripts/python.exe")):
            candidates.append(scripts)
        for scripts in sorted(parent.glob("*/bin/python")):
            candidates.append(scripts)
    except OSError:
        pass

    # 用户级 WorkBuddy venv：安装副本（~/.claude/skills/…）旁没有 venv，
    # 只有靠它才能让「仓库外验证」真正跑到 Tier1 而不是集体跳过。
    home = Path.home()
    for sub in ("Scripts/python.exe", "bin/python"):
        candidates.append(home / ".workbuddy" / "binaries" / "python"
                          / "envs" / "default" / sub)

    candidates.append(Path(sys.executable))

    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and _sci_ok(c, mods):
            return c
    return None


def skip_reason_sci(mods: tuple[str, ...] = ("numpy",)) -> str:
    """统一的跳过措辞——必须说明「找的是哪里」，不能断言「本机没有」。"""
    return (f"未在候选范围内发现带 {'+'.join(mods)} 的解释器"
            f"（候选：BIOSTAT_SCI_PYTHON / 包旁 venv / 用户级 venv /"
            f" sys.executable）")
