#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""_common.py —— biostat-copilot 全部脚本的共享底座（纯标准库）

设计约定（与 SKILL.md 一致）：
  * stdout 只放数据（--json 时为 JSON），人类提示一律走 stderr
  * 退出码：0 成功 / 2 输入错误 / 3 口径缺失 / 4 判定门 FAIL / 1 其他错误
  * 零进展检测：长时间无字节推进即报错，杜绝「进程在跑但零产出」的静默失败（P01）
  * 路径安全：所有写盘路径必须落在项目根内，防路径穿越
  * 可选加速：numpy/pandas/duckdb 有则用，无则回退标准库（跨 agent 裸环境可跑）
"""

from __future__ import annotations

import gzip
import io
import json
import math
import os
import sys
import time
from pathlib import Path

# ---- 退出码契约 ----------------------------------------------------------
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INPUT = 2
EXIT_CALIBER = 3
EXIT_GATE_FAIL = 4

_UNSPECIFIED = "UNSPECIFIED"


# ---- 输出协议 ------------------------------------------------------------
def _force_utf8() -> None:
    """Windows 控制台默认 cp936，强制 UTF-8 输出，避免中文 JSON 报错。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def log(msg: str) -> None:
    """人类可读提示 → stderr（不进 stdout，保证 stdout 可被机器解析）。"""
    _force_utf8()
    print(msg, file=sys.stderr, flush=True)


def emit(obj: dict, as_json: bool = True, code: int = EXIT_OK) -> int:
    """把结构化结果写到 stdout 并返回退出码。

    **allow_nan=False**：Python 的 json.dumps 默认会写出 `NaN` / `Infinity`
    字面量，而 RFC 8259 明确不允许 —— 严格 JSON 解析器（Go/Rust/Java/JS 生态
    多数）会直接报「非法常量」拒绝整个文档。更糟的是 Python 侧 json.loads
    会**静默接受**它，导致下游拿到 NaN 却毫无察觉，最终变成一个静默传播
    的坏数字。这里宁可在这里炸掉，也不能让非法 JSON 流到下游。
    业务代码若真需要「无值」，应显式写 null。
    """
    _force_utf8()
    if as_json:
        try:
            text = json.dumps(obj, ensure_ascii=False, indent=2,
                              default=str, allow_nan=False)
        except ValueError as e:
            # 兜底：把非法浮点替换成 null 后仍然产出**合法** JSON，
            # 并在 stderr 里明确指出是哪个键 —— 失败必须可见（R2）。
            text = json.dumps(_sanitize_nan(obj), ensure_ascii=False, indent=2,
                              default=str, allow_nan=False)
            log(f"[FAIL] 输出含非有限浮点（NaN/Inf），已置为 null: {e}")
            code = code or EXIT_ERROR
        print(text)
    else:
        # 非 json 模式：仅打印ok与摘要，细节靠 stderr
        print(f"ok={obj.get('ok')} {obj.get('summary', '')}".strip())
    return code


def _sanitize_nan(obj):
    """递归把 NaN/±Inf 换成 None（仅在 emit 兜底路径使用）。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    return obj


def fail(msg: str, code: int = EXIT_ERROR, as_json: bool = True, **extra) -> int:
    """统一失败出口：JSON 里 ok=false，人类信息走 stderr。"""
    obj = {"ok": False, "error": msg}
    obj.update(extra)
    log(f"[FAIL] {msg}")
    return emit(obj, as_json=as_json, code=code)


# ---- 口径守卫（口径先于数值） --------------------------------------------
class CaliberError(Exception):
    """指标缺口径。默认必须报错，不得给默认值（P09 / R3）。"""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"缺少口径声明: {', '.join(missing)}")


def require_caliber(config: dict, keys: list[str],
                    allow_missing: bool = False) -> dict:
    """校验口径字段齐全；缺口径时抛 CaliberError（除非显式 allow_missing）。

    返回补齐后的口径字典：缺失项在 allow_missing 模式下填 UNSPECIFIED。
    """
    missing = [k for k in keys if not config.get(k)]
    if missing and not allow_missing:
        raise CaliberError(missing)
    return {k: (config.get(k) or _UNSPECIFIED) for k in keys}


# ---- 路径安全 ------------------------------------------------------------
def safe_join(root: str | os.PathLike, *parts: str) -> Path:
    """把 parts 拼到 root 下，并确保结果仍在 root 内（防路径穿越，P-安全）。"""
    root_p = Path(root).resolve()
    target = root_p.joinpath(*parts).resolve()
    try:
        target.relative_to(root_p)
    except ValueError:
        raise PermissionError(f"路径越界（疑似穿越）: {target} 不在 {root_p} 内")
    return target


# ---- 数据读取（gzip 透明 + 流式分块） ------------------------------------
def open_text(path: str | os.PathLike):
    """按后缀自动解压打开文本；.gz/.bgz 走 gzip，其余明文。

    换行统一交给 Python（newline 默认 None → 通用换行）：CRLF / CR / LF 全部归一为 '\\n'。
    **不要用 newline=""**——那会保留 '\\r'，而各脚本统一以 rstrip("\\n") 去行尾，
    结果是 CRLF 文件的**最后一个字段带着 '\\r'**，与列名匹配时静默失配（实测踩过：
    replication_check 的 n_control 因此恒为 None）。
    """
    p = Path(path)
    # encoding="utf-8-sig"：它对无 BOM 的文件行为与 utf-8 完全相同，
    # 但会剥掉 UTF-8 BOM。Excel 导出的 TSV/CSV 几乎必带 BOM，
    # 用 "utf-8" 时首列列名会变成 "﻿chr"，导致 resolve() 返回 {}，
    # 而下游只检查某一个列是否存在 → **全部质控静默失效**（实测踩过：
    # 带 BOM 时 dup_keys 恒为 0，无任何告警）。vendor/core/contract.py 早就
    # 用了 utf-8-sig，这里是同项目内两套 BOM 策略的统一。
    if p.suffix in (".gz", ".bgz"):
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8-sig")
    return open(p, "r", encoding="utf-8-sig")


def is_gzip_valid(path: str | os.PathLike) -> tuple[bool, str]:
    """完整读完 gzip 流；截断/损坏返回 (False, 原因)。用于完整性核验第 2 道。"""
    p = Path(path)
    if p.suffix not in (".gz", ".bgz"):
        return True, "not_gzip"
    try:
        with gzip.open(p, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
        return True, "ok"
    except (EOFError, OSError) as e:
        return False, f"gzip 损坏或截断: {e}"


class ProgressWatch:
    """零进展检测：在 process 调用间喂入字节数，超时无增量即抛错（P01）。

    用法：
        w = ProgressWatch(timeout=120)
        ...
        w.tick(bytes_done)
    """

    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout
        self._last_bytes = 0
        self._last_t = time.time()

    def tick(self, bytes_done: int) -> None:
        now = time.time()
        if bytes_done > self._last_bytes:
            self._last_bytes = bytes_done
            self._last_t = now
            return
        if now - self._last_t > self.timeout:
            raise TimeoutError(
                f"零进展：{self.timeout:.0f}s 内字节数未增加（卡在 {bytes_done} 字节）。"
                f"排查：分块过大 / 并发过高 / 连接被限速（见 pitfalls P01）。")


# ---- 可选加速探测 --------------------------------------------------------
def optional(name: str):
    """尝试导入可选包；缺失返回 None（调用方据此回退标准库实现）。"""
    try:
        return __import__(name)
    except Exception:
        return None


def has(mod) -> bool:
    return mod is not None


def version_of(mod) -> str:
    return getattr(mod, "__version__", "?") if mod is not None else "-"


# ---- 通用小工具 ----------------------------------------------------------
def read_json(path: str | os.PathLike, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json(path: str | os.PathLike, obj: dict) -> None:
    """原子写 JSON：先写临时文件再替换，避免中断留下半个文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")
    os.replace(tmp, p)


def normalize_chrom(value: str) -> str:
    """统一染色体前缀：'chr1'/'1' → 'chr1'（防 chrchr1，P04）。"""
    s = str(value).strip()
    return s if s.startswith("chr") else "chr" + s


def as_maf(freq: float) -> float:
    """把任意等位频率折算成次等位频率 min(f, 1-f)（防 MAF 口径混淆，P05）。"""
    f = float(freq)
    return min(f, 1.0 - f)


def sha256_of(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def md5_of(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def file_size(path: str | os.PathLike) -> int:
    return Path(path).stat().st_size
