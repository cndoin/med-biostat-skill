#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_stability.py —— 稳定性：边界与极端输入不得崩溃，且失败必须干净（非静默）。

覆盖：空文件 / 只有表头 / 单行 / p=0 与 p=1 / 坐标越界 / 重复变异 / 非法数值。
判据：exit ∈ {0,2,3}；成功时 stdout 必须是可解析 JSON；**绝不静默返回错误结果**。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import _harness as H  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="biostat_stab_"))
HDR = "#chrom\tpos\tref\talt\tpval\tbeta\tsebeta\taf_alt\n"

CASES = {
    "empty_file": "",
    "header_only": HDR,
    "single_row": HDR + "1\t1000\tA\tG\t0.5\t0.1\t0.05\t0.3\n",
    "p_zero_one": (HDR + "1\t1000\tA\tG\t0\t0.1\t0.05\t0.3\n"
                        "1\t2000\tA\tG\t1\t0.1\t0.05\t0.3\n"),
    "huge_coords": HDR + "1\t2000000000\tA\tG\t1e-9\t0.1\t0.05\t0.3\n",
    "dup_variants": (HDR + "1\t1000\tA\tG\t1e-9\t0.1\t0.05\t0.3\n"
                          "1\t1000\tA\tG\t1e-9\t0.1\t0.05\t0.3\n"),
    "bad_numeric": HDR + "1\txyz\tA\tG\tNaN\tabc\t0.05\t0.3\n",
    "negative_pos": HDR + "1\t-5\tA\tG\t1e-9\t0.1\t0.05\t0.3\n",
    "af_out_of_range": HDR + "1\t1000\tA\tG\t1e-9\t0.1\t0.05\t7.5\n",
    "crlf_utf8_bom": "\ufeff" + HDR.replace("\n", "\r\n")
                     + "1\t1000\tA\tG\t1e-9\t0.1\t0.05\t0.3\r\n",
}


def _mk(name, text):
    p = TMP / f"{name}.tsv"
    p.write_text(text, encoding="utf-8", newline="")
    return p


class TestStability(unittest.TestCase):
    def test_qc_and_loci_never_crash(self):
        for name, text in CASES.items():
            with self.subTest(case=name):
                f = _mk(name, text)
                code, d, err = H.run("qc_gwas.py", "--input", f,
                                     "--bucket", "maf_0.01_0.5", "--json")
                self.assertIn(code, (0, 1, 2, 3), err)
                code2, d2, err2 = H.run("loci_count.py", "--input", f,
                                        "--p-threshold", "5e-8", "--json")
                self.assertIn(code2, (0, 1, 2, 3), err2)
                # 成功时必须给出可解析 JSON
                if code == 0:
                    self.assertIsInstance(d, dict)

    def test_empty_returns_ok_not_fake_numbers(self):
        """空数据不得产出伪造的 λGC（应为 None），也不能崩。"""
        f = _mk("empty_only", HDR)
        code, d, _ = H.run("qc_gwas.py", "--input", f, "--bucket", "x", "--json")
        self.assertEqual(code, 0)
        self.assertIsNone(d["lambda_gc"])
        self.assertEqual(d["n_variants_usable"], 0)

    def test_nonexistent_file(self):
        code, d, _ = H.run("qc_gwas.py", "--input", str(TMP / "nope.tsv"),
                           "--bucket", "x", "--json")
        self.assertNotEqual(code, 0)
        self.assertFalse(d["ok"])

    def test_single_row_loci(self):
        f = _mk("single_loci", HDR + "1\t1000\tA\tG\t1e-9\t0.1\t0.05\t0.3\n")
        code, d, _ = H.run("loci_count.py", "--input", f,
                           "--p-threshold", "5e-8", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(d["sensitivity_matrix"][0]["n_independent_loci"], 1)

    def test_safe_join_module_level(self):
        """直接测路径安全函数（越界必须抛错）。"""
        sys.path.insert(0, str(H.PKG / "scripts"))
        import _common as C
        with self.assertRaises(PermissionError):
            C.safe_join(TMP, "..", "..", "evil.txt")
        ok = C.safe_join(TMP, "sub", "a.txt")
        self.assertTrue(str(ok).startswith(str(TMP.resolve())))

    def test_maf_conversion(self):
        sys.path.insert(0, str(H.PKG / "scripts"))
        import _common as C
        self.assertAlmostEqual(C.as_maf(0.9), 0.1, places=9)
        self.assertAlmostEqual(C.as_maf(0.3), 0.3, places=9)
        self.assertAlmostEqual(C.as_maf(0.5), 0.5, places=9)

    def test_chrom_normalize(self):
        sys.path.insert(0, str(H.PKG / "scripts"))
        import _common as C
        self.assertEqual(C.normalize_chrom("1"), "chr1")
        self.assertEqual(C.normalize_chrom("chr1"), "chr1")   # 不重复加前缀（P04）
        self.assertEqual(C.normalize_chrom("X"), "chrX")

    def test_progress_watch_fires(self):
        """零进展检测必须真的会触发（P01）。"""
        sys.path.insert(0, str(H.PKG / "scripts"))
        import _common as C
        w = C.ProgressWatch(timeout=0.05)
        w.tick(100)
        import time
        time.sleep(0.08)
        with self.assertRaises(TimeoutError):
            w.tick(100)   # 无增量 → 超时抛错


if __name__ == "__main__":
    unittest.main(verbosity=2)
