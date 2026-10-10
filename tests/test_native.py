#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_native.py —— 原生加速层（魔改版 biostat-vcf）的自检

守护四件事：

1. **参考实现本身要对** —— 先证明 `reference_sumstats.py` 能在夹具上
   复现出注入异常的**确定**期望值；否则「对拍通过」只说明两个都错。
2. **原生层与参考实现同口径同结果** —— 有二进制时逐字段比对 λGC 与
   各项 issues 计数。
3. **失败语义不能被伪装** —— 二进制不存在 → SKIP；校验阻断 → BLOCKED；
   两者都不得被读成 CRASH，也不得被读成 OK。
4. **对拍门禁必须能失败** —— 人为扰动后门禁要报差异，否则它只是摆设。

无二进制时本模块**不 fail**，只 skip 依赖二进制的用例
（裸 Python / 无 Rust 产物的环境照常可跑）。
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PKG = TESTS.parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(PKG / "tools"))

from fixtures.make_fixtures import build                      # noqa: E402
import reference_sumstats as REF                              # noqa: E402
import native_accel as NA                                     # noqa: E402

FIX: dict = {}

ISSUE_KEYS = ["nonfinite_p", "p_out_of_range", "nonfinite_beta",
              "nonpositive_se", "missing_af", "duplicate_keys"]


def setUpModule():
    global FIX
    FIX = build(tempfile.mkdtemp(prefix="biostat_native_fix_"))


def _parity_diff(a: dict, b: dict, tol: float = 1e-9) -> list:
    """比对两个 sumstats 结果；返回差异清单（空 = 一致）。

    比对**逐字段**而不是只比 λGC 一个数：只比 λGC 时，
    issues 计数全错也能通过。
    """
    diffs = []
    la, lb = a.get("lambda_gc"), b.get("lambda_gc")
    if la is None or lb is None:
        if la != lb:
            diffs.append(f"lambda_gc: {la} vs {lb}")
    elif not math.isclose(la, lb, rel_tol=tol, abs_tol=tol):
        diffs.append(f"lambda_gc: {la!r} vs {lb!r}")
    for k in ("rows_read", "rows_used"):
        if a.get(k) != b.get(k):
            diffs.append(f"{k}: {a.get(k)} vs {b.get(k)}")
    ia, ib = a.get("issues", {}), b.get("issues", {})
    for k in ISSUE_KEYS:
        if ia.get(k) != ib.get(k):
            diffs.append(f"issues.{k}: {ia.get(k)} vs {ib.get(k)}")
    return diffs


def _native_report(path: str) -> dict:
    payload, _code = NA.sumstats(path)
    return payload


class TestReferenceAgainstKnownFixture(unittest.TestCase):
    """第 1 层：参考实现必须复现出**注入时确定的**期望值。"""

    def test_issue_counts_match_injected(self):
        rep = REF.reference(FIX["sumstats"])
        exp = FIX["sumstats_expected"]
        for k in ISSUE_KEYS:
            self.assertEqual(rep["issues"][k], exp[k],
                             f"{k} 期望 {exp[k]}，参考实现给出 {rep['issues'][k]}")
        self.assertEqual(rep["rows_read"], exp["rows_read"])
        self.assertIsNotNone(rep["lambda_gc"])
        self.assertTrue(math.isfinite(rep["lambda_gc"]))

    def test_gzip_and_plain_agree(self):
        a = REF.reference(FIX["sumstats"])
        b = REF.reference(FIX["sumstats_gz"])
        self.assertEqual(_parity_diff(a, b), [], "gz 与未压缩必须给出同一结果")

    def test_af_alt_above_half_is_folded_not_dropped(self):
        """`af_alt>0.5` 必须折算成 MAF，而不是被当越界丢掉（P05）。

        夹具含 2000 正常行 + 1 行 af_alt=0.92（折叠后 maf=0.08，应可用）；
        若实现不做折叠，rows_used 会少 1。
        """
        exp = FIX["sumstats_expected"]["rows_used"]
        rep = REF.reference(FIX["sumstats"])
        self.assertEqual(rep["rows_used"], exp,
                         f"af_alt 折叠失效：rows_used={rep['rows_used']}，期望 {exp}")

    def test_hash_chrom_header_is_not_treated_as_comment(self):
        """`#chrom` 是表头（管线惯例），不是注释。"""
        rep = REF.reference(FIX["sumstats_hash"])
        self.assertEqual(rep["rows_read"], FIX["sumstats_expected"]["rows_read"],
                         "#chrom 表头被当注释跳过了")
        self.assertGreater(rep["rows_used"], 0, "未解析出可用行")

    def test_hash_and_plain_header_agree(self):
        a = REF.reference(FIX["sumstats"])
        b = REF.reference(FIX["sumstats_hash"])
        self.assertEqual(_parity_diff(a, b), [], "#chrom 与 chr 表头必须同结果")


class TestParityGateCanFail(unittest.TestCase):
    """第 4 层：门禁本身必须能失败（否则无法区分「有效」与「摆设」）。"""

    def test_gate_detects_perturbed_lambda(self):
        good = REF.reference(FIX["sumstats"])
        bad = json.loads(json.dumps(good))
        bad["lambda_gc"] = good["lambda_gc"] * 1.05          # 人为扰动 5%
        self.assertTrue(_parity_diff(good, bad), "λGC 变了却报一致 → 门禁失效")

    def test_gate_detects_perturbed_issue_count(self):
        good = REF.reference(FIX["sumstats"])
        bad = json.loads(json.dumps(good))
        bad["issues"]["nonpositive_se"] += 1                 # 只动一个计数
        diffs = _parity_diff(good, bad)
        self.assertTrue(any("nonpositive_se" in d for d in diffs),
                        "issues 计数变了却报一致 → 门禁只看 λGC")

    def test_gate_detects_row_count_drift(self):
        good = REF.reference(FIX["sumstats"])
        bad = json.loads(json.dumps(good))
        bad["rows_used"] += 1
        self.assertTrue(_parity_diff(good, bad), "行数漂移未被发现")


class TestFailureSemantics(unittest.TestCase):
    """第 3 层：失败必须可见且分类正确（不需要二进制）。"""

    def test_missing_binary_is_skip_not_crash(self):
        import os
        old = os.environ.get("BIOSTAT_NATIVE_BIN")
        os.environ["BIOSTAT_NATIVE_BIN"] = str(PKG / "vendor" / "native" / "__nope__")
        try:
            payload, code = NA.sumstats(FIX["sumstats"])
        finally:
            if old is None:
                os.environ.pop("BIOSTAT_NATIVE_BIN", None)
            else:
                os.environ["BIOSTAT_NATIVE_BIN"] = old
        self.assertEqual(payload["status"], "SKIP")
        self.assertFalse(payload["ok"])
        self.assertNotEqual(code, 0, "缺二进制必须是非零退出，不得伪装成功")
        self.assertIn("qc_gwas.py", payload["fallback"],
                      "必须给出可用的降级路径，而不是只说不行")


@unittest.skipUnless(NA.available(), "需要 vendor/native/biostat-vcf 二进制")
class TestNativeParity(unittest.TestCase):
    """第 2 层：有二进制时与参考实现逐字段对拍。"""

    def test_official_passthrough_needs_no_input(self):
        """兜底透传入口必须可用（「包装层不是功能上限」的实现方式）。

        回归保护：早期 `--input` 前置校验放在透传分支之前，
        导致 `official -- --help` 被自己的 parser 拒成 exit 2，兜底入口形同虚设。
        """
        cp = subprocess.run(
            [sys.executable, str(PKG / "tools" / "native_accel.py"),
             "official", "--", "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(cp.returncode, 0,
                         f"透传入口被拒：{cp.stdout[:200]}{cp.stderr[:200]}")
        self.assertIn("Usage", cp.stdout, "透传未真正调到二进制")

    def test_no_capability_drift(self):
        rep = NA.probe()
        self.assertTrue(rep["available"])
        self.assertEqual(rep["declared_but_missing"], [],
                         f"二进制缺少已声明能力：{rep['declared_but_missing']}")
        self.assertIn("sumstats", rep["reported_commands"])

    def test_parity_with_reference(self):
        native = (_native_report(FIX["sumstats"]) or {}).get("report") or {}
        reference = REF.reference(FIX["sumstats"])
        diffs = _parity_diff(native, reference)
        self.assertEqual(diffs, [], "原生层与参考实现不一致：\n  " + "\n  ".join(diffs))

    def test_issue_counts_match_injected(self):
        native = (_native_report(FIX["sumstats"]) or {}).get("report") or {}
        exp = FIX["sumstats_expected"]
        for k in ISSUE_KEYS:
            self.assertEqual(native["issues"][k], exp[k],
                             f"{k} 期望 {exp[k]}，原生层给出 {native['issues'][k]}")

    def test_gzip_transparently_decoded(self):
        plain = (_native_report(FIX["sumstats"]) or {}).get("report") or {}
        gz = (_native_report(FIX["sumstats_gz"]) or {}).get("report") or {}
        self.assertEqual(_parity_diff(plain, gz), [], "gz 未被透明解码")

    def test_af_alt_above_half_is_folded_not_dropped(self):
        exp = FIX["sumstats_expected"]["rows_used"]
        native = (_native_report(FIX["sumstats"]) or {}).get("report") or {}
        self.assertEqual(native.get("rows_used"), exp,
                         f"原生层 af_alt 折叠失效：rows_used={native.get('rows_used')}，期望 {exp}")

    def test_hash_chrom_header_parity(self):
        """原生层必须把 `#chrom` 当表头（回归保护：曾把整张表读成缺列）。"""
        native = (_native_report(FIX["sumstats_hash"]) or {}).get("report") or {}
        self.assertGreater(native.get("rows_used", 0), 0,
                           "原生层把 #chrom 表头当注释 → 一行都没用上")
        reference = REF.reference(FIX["sumstats_hash"])
        diffs = _parity_diff(native, reference)
        self.assertEqual(diffs, [], "#chrom 表头上原生层与参考实现不一致：\n  "
                         + "\n  ".join(diffs))

    def test_caliber_label_present(self):
        payload = _native_report(FIX["sumstats"])
        self.assertIn("lambda_caliber", payload,
                      "必须带口径标签，否则下游会与 qc_gwas 的口径混用")
        self.assertIn("wald", payload["lambda_caliber"])
        self.assertEqual(payload["implementation"], "rust:biostat-vcf/sumstats")

    def test_deterministic_across_threads(self):
        r1 = (_native_report(FIX["sumstats"]) or {}).get("report") or {}
        payload, _ = NA.sumstats(FIX["sumstats"], threads=1)
        r2 = payload.get("report") or {}
        self.assertEqual(_parity_diff(r1, r2), [],
                         "线程数不同时结果不可有差异")

    def test_blocking_failure_is_classified_not_crash(self):
        """缺必需列 → 上游走错误路径（信封在 stderr）→ 必须判 BLOCKED。"""
        bad = Path(FIX["sumstats"]).with_name("bad_cols.tsv")
        bad.write_text("chr\tpos\taf\tbeta\tp\n1\t100\t0.3\t0.1\t0.01\n",
                       encoding="utf-8")
        payload, code = NA.sumstats(str(bad))
        self.assertEqual(payload["status"], "BLOCKED",
                         f"应判为 BLOCKED，实际 {payload['status']}")
        self.assertNotEqual(payload["status"], "CRASH",
                            "stderr 里的合法错误信封被误判成崩溃")
        self.assertNotEqual(code, 0)
        self.assertIsNotNone(payload.get("report"),
                             "阻断时仍要带回 report（诊断信息不能丢）")

    def test_missing_file_is_io_not_crash(self):
        payload, _ = NA.sumstats(str(Path(FIX["sumstats"]).with_name("nope.tsv")))
        self.assertEqual(payload["status"], "IO",
                         f"文件不存在应判 IO，实际 {payload['status']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
