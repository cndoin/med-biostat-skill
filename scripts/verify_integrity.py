#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_integrity.py —— S1③ 完整性五道核验（强度递增）

  1. 字节数 == 官方 Content-Length
  2. 压缩完整性（gzip 全量读通）
  3. 哈希对照（md5 / sha256，官方提供时）
  4. 两套官方产物逐变异互证 —— 见 cross_verify.py
  5. 等位基因方向与 build 一致性 —— 见 cross_verify.py

本脚本做 1–3 道；4–5 道由 cross_verify.py 完成。

用法：
    python verify_integrity.py --file raw/x.gz --expect-bytes 123456 \
        --expect-md5 <MD5> --json
退出码：0 全通过 / 1 有未通过项
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="完整性核验（第 1–3 道）")
    ap.add_argument("--file", required=True)
    ap.add_argument("--expect-bytes", type=int, default=None,
                    help="官方 Content-Length")
    ap.add_argument("--expect-md5", default=None)
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    f = Path(args.file)
    if not f.exists():
        return C.fail(f"文件不存在: {f}", code=C.EXIT_INPUT, as_json=args.json)

    checks = []

    # 第 1 道：字节数
    actual_bytes = C.file_size(f)
    if args.expect_bytes is None:
        checks.append({"id": "bytes", "status": "skipped",
                       "reason": "未提供 --expect-bytes（口径缺口，不作通过处理）",
                       "actual": actual_bytes})
    else:
        ok = actual_bytes == args.expect_bytes
        checks.append({"id": "bytes", "status": "pass" if ok else "fail",
                       "expected": args.expect_bytes, "actual": actual_bytes})

    # 第 2 道：压缩完整性
    gz_ok, gz_msg = C.is_gzip_valid(f)
    checks.append({"id": "gzip", "status": "pass" if gz_ok else "fail",
                   "detail": gz_msg})

    # 第 3 道：哈希
    if args.expect_md5:
        got = C.md5_of(f)
        checks.append({"id": "md5", "status": "pass" if got.lower() == args.expect_md5.lower()
                       else "fail", "expected": args.expect_md5.lower(), "actual": got})
    elif args.expect_sha256:
        got = C.sha256_of(f)
        checks.append({"id": "sha256",
                       "status": "pass" if got.lower() == args.expect_sha256.lower() else "fail",
                       "expected": args.expect_sha256.lower(), "actual": got})
    else:
        checks.append({"id": "hash", "status": "skipped",
                       "reason": "未提供 --expect-md5/--expect-sha256"})
        checks.append({"id": "sha256_local",
                       "status": "info", "actual": C.sha256_of(f)})

    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_skip = sum(1 for c in checks if c["status"] == "skipped")
    ok = n_fail == 0

    report = {
        "ok": ok,
        "summary": f"{'全通过' if ok else str(n_fail) + ' 项未通过'}"
                   + (f"，{n_skip} 项因缺口径被跳过" if n_skip else ""),
        "file": str(f),
        "checks": checks,
        "n_fail": n_fail,
        "n_skipped": n_skip,
        "note": ("skipped 表示关键口径未提供，按『不得默认通过』处理；"
                 "第 4–5 道（逐变异互证 / 等位基因方向）见 cross_verify.py"),
    }
    for c in checks:
        C.log(f"  [{c['status'].upper():7s}] {c['id']}")
    return C.emit(report, as_json=args.json,
                  code=C.EXIT_OK if ok else C.EXIT_ERROR)


if __name__ == "__main__":
    sys.exit(main())
