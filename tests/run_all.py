#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_all.py —— 全量自检驱动（纯标准库 unittest，零依赖）

用法：
    python tests/run_all.py            # 人类可读
    python tests/run_all.py --json     # 结构化摘要
退出码：0 全绿 / 1 有失败
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PKG = TESTS.parent
sys.path.insert(0, str(TESTS))

SUITES = ["test_core", "test_stability", "test_security",
          "test_guards_negative", "test_vendor", "test_mr_chain", "test_lib",
          "test_regression_audit", "test_wrapper", "test_native",
          "test_opensource",
          # v1.7：主流分析法（13 个新模块，40 个用例）
          "test_methods"]


class _Collector(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.failures_detail = []

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.failures_detail.append({"test": str(test), "kind": "FAIL",
                                     "detail": self._exc_info_to_string(err, test)[:800]})

    def addError(self, test, err):
        super().addError(test, err)
        self.failures_detail.append({"test": str(test), "kind": "ERROR",
                                     "detail": self._exc_info_to_string(err, test)[:800]})


def main() -> int:
    ap = argparse.ArgumentParser(description="biostat-copilot 全量自检")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in SUITES:
        try:
            suite.addTests(loader.loadTestsFromName(name))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] 无法加载 {name}: {e}", file=sys.stderr)

    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=_Collector)
    result = runner.run(suite)

    total = result.testsRun
    n_fail = len(result.failures)
    n_err = len(result.errors)
    n_skip = len(result.skipped)
    passed = total - n_fail - n_err - n_skip
    ok = (n_fail == 0 and n_err == 0)

    version = (PKG / "VERSION").read_text(encoding="utf-8").strip()
    report = {
        "ok": ok,
        "summary": f"biostat-copilot v{version} 自检：{passed}/{total} 通过"
                   + (f"，{n_fail} 失败 {n_err} 错误" if not ok else ""),
        "total": total, "passed": passed, "failures": n_fail,
        "errors": n_err, "skipped": n_skip,
        "details": result.failures_detail,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(stream.getvalue())
        print("=" * 70)
        print(report["summary"])
        for f in result.failures_detail:
            print(f"  [{f['kind']}] {f['test']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
