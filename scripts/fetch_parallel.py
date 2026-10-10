#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fetch_parallel.py —— S1② 分块并行取回（小分块 + 断点续传 + 零进展检测）

硬要求（来自实测反例：16 路 + 48 MB 分块跑 30 分钟仍是 0 字节）：
  * **小分块**（默认 8 MB），进度必须可见
  * 每块独立文件 → 已完整块直接复用（断点续传零成本）
  * **零进展检测**：超时无字节推进即报错，杜绝静默失败（P01）

用法：
    python fetch_parallel.py --url <URL> --out raw/x.gz \
        --chunk-mb 8 --concurrency 8 --json
    # 冒烟测试（截断，产物标记 truncated=true，P02）
    python fetch_parallel.py --url <URL> --out raw/smoke.gz --max-bytes 25000000 --json
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

UA = "biostat-copilot/1.0 (+parallel-chunked-fetch)"
_lock = threading.Lock()


def head(url: str, timeout: int = 30):
    """探测 Content-Length 与是否支持 Range。"""
    req = Request(url, method="HEAD", headers={"User-Agent": UA})
    with urlopen(req, timeout=timeout) as r:
        h = r.headers
        total = int(h.get("Content-Length", 0))
        accept = h.get("Accept-Ranges", "").lower() == "bytes"
        return total, accept, dict(h)


def fetch_chunk(url: str, start: int, end: int, path: Path, timeout: int = 120):
    """下载 [start, end] 闭区间到 path。已存在且大小正确则复用（断点续传）。"""
    need = end - start + 1
    if path.exists() and path.stat().st_size == need:
        return "reused", need
    req = Request(url, headers={"User-Agent": UA,
                                "Range": f"bytes={start}-{end}"})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
        got = 0
        while True:
            blk = r.read(1 << 20)
            if not blk:
                break
            fh.write(blk)
            got += len(blk)
    if got != need:
        tmp.unlink(missing_ok=True)
        raise IOError(f"分块 {start}-{end} 期望 {need} 字节，实得 {got}")
    os.replace(tmp, path)
    return "fetched", got


def main() -> int:
    ap = argparse.ArgumentParser(description="分块并行取回")
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-mb", type=float, default=8.0, help="分块大小（MB），小分块是硬要求")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-bytes", type=int, default=None,
                    help="冒烟上限；触发时产物写 truncated=true（P02）")
    ap.add_argument("--no-progress-timeout", type=float, default=120,
                    help="零进展判定超时（秒）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = out.parent / (out.name + ".parts")
    parts_dir.mkdir(exist_ok=True)

    try:
        total, accept, _ = head(args.url)
    except (HTTPError, URLError, OSError) as e:
        return C.fail(f"HEAD 探测失败：{e}", code=C.EXIT_INPUT, as_json=args.json)

    if total <= 0:
        return C.fail("Content-Length 未知，无法分块（口径缺失，拒绝盲下）",
                      code=C.EXIT_INPUT, as_json=args.json)

    limit = min(total, args.max_bytes) if args.max_bytes else total
    truncated = bool(args.max_bytes and args.max_bytes < total)
    chunk = int(args.chunk_mb * 1024 * 1024)
    ranges = [(s, min(s + chunk - 1, limit - 1)) for s in range(0, limit, chunk)]

    C.log(f"总大小 {total:,} 字节；本次下载 {limit:,} 字节；{len(ranges)} 块 "
          f"× {args.chunk_mb:g} MB；并发 {args.concurrency}"
          + ("  [截断模式]" if truncated else ""))

    t0 = time.time()
    done_bytes = 0
    reused = fetched = 0
    watch = C.ProgressWatch(timeout=args.no_progress_timeout)

    def run(idx_range):
        idx, (s, e) = idx_range
        part = parts_dir / f"{out.name}.part{idx:05d}"
        kind, n = fetch_chunk(args.url, s, e, part)
        return kind, n

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run, (i, r)): i for i, r in enumerate(ranges)}
            for fut in as_completed(futs):
                kind, n = fut.result()
                with _lock:
                    done_bytes += n
                    if kind == "reused":
                        reused += 1
                    else:
                        fetched += 1
                watch.tick(done_bytes)
                if (reused + fetched) % max(1, len(ranges) // 10) == 0:
                    C.log(f"  ... {done_bytes:,}/{limit:,} 字节")
    except (TimeoutError, IOError, HTTPError, URLError) as e:
        return C.fail(f"取回中断：{e}", code=C.EXIT_ERROR, as_json=args.json,
                      downloaded_bytes=done_bytes, truncated=truncated)

    # 合并（仅非截断模式合并为最终文件）
    merged = None
    if not truncated:
        with open(out, "wb") as w:
            for i in range(len(ranges)):
                w.write((parts_dir / f"{out.name}.part{i:05d}").read_bytes())
        merged = str(out)
        C.log(f"已合并 → {out}（{out.stat().st_size:,} 字节）")

    report = {
        "ok": True,
        "summary": f"下载 {done_bytes:,} 字节（复用 {reused} 块 / 新取 {fetched} 块）"
                   + ("，截断模式" if truncated else ""),
        "url": args.url,
        "total_bytes": total,
        "downloaded_bytes": done_bytes,
        "n_chunks": len(ranges),
        "chunk_mb": args.chunk_mb,
        "concurrency": args.concurrency,
        "accept_ranges": accept,
        "reused_chunks": reused,
        "fetched_chunks": fetched,
        "truncated": truncated,          # ★ 截断标记（P02）
        "merged_file": merged,
        "parts_dir": str(parts_dir),
        "elapsed_sec": round(time.time() - t0, 1),
        "throughput_mb_s": round(done_bytes / 1e6 / max(time.time() - t0, 1e-6), 3),
    }
    if truncated:
        C.log("[WARN] 截断模式：产物不得作为最终结论（P02）")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
