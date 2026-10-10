#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_security.py —— 安全：注入 / 路径穿越 / 越界输入。

对应质量门「安全」维度。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent / "scripts"))
import _harness as H  # noqa: E402
import _common as C  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="biostat_sec_"))


class TestPathTraversal(unittest.TestCase):
    def test_safe_join_blocks_escape(self):
        for evil in ["../evil.txt", "sub/../../evil.txt", ".."]:
            with self.assertRaises(PermissionError):
                C.safe_join(TMP, evil)

    def test_safe_join_allows_inside(self):
        p = C.safe_join(TMP, "a", "b", "c.txt")
        self.assertTrue(str(p.resolve()).startswith(str(TMP.resolve())))


class TestHtmlInjection(unittest.TestCase):
    def test_markdown_script_is_escaped(self):
        """报告里的 <script> 必须被转义，不得原样进入 HTML。"""
        md = H.write(TMP / "inj.md",
                     "# 标题<script>alert(1)</script>\n\n"
                     "## 摘要\n<img src=x onerror=alert(2)>\n\n"
                     "## 引言\nx\n\n## 方法\nx\n\n## 结果\nx\n\n"
                     "## 讨论\nx\n\n## 结论\nx\n\n## 数据与代码可用性\nx\n")
        out = TMP / "inj.html"
        code, d, _ = H.run("report_pipeline.py", "--md", md,
                           "--contract", "academic_paper",
                           "--out", str(out), "--json")
        # 无论门禁是否通过，只要产出了 HTML，就必须已转义
        if out.exists():
            html = out.read_text(encoding="utf-8")
            # 危险的是「未转义的标签」，而不是转义后文本里的字面量
            self.assertNotIn("<script>", html)
            self.assertNotIn("<img", html)
            self.assertIn("&lt;script&gt;", html)

    def test_inline_escape_direct(self):
        import importlib
        rp = importlib.import_module("report_pipeline")
        body, _ = rp.md_to_html("# T\n\n<script>bad()</script>\n")
        self.assertNotIn("<script>", body)
        self.assertIn("&lt;script&gt;", body)


class TestOutOfRangeInput(unittest.TestCase):
    def test_extreme_p_no_crash(self):
        f = H.write(TMP / "extreme.tsv",
                    "#chrom\tpos\tref\talt\tpval\tbeta\tsebeta\taf_alt\n"
                    "1\t1000\tA\tG\t-1\t0.1\t0.05\t0.3\n"      # p<0
                    "1\t2000\tA\tG\t2\t0.1\t0.05\t0.3\n"       # p>1
                    "1\t3000\tA\tG\t1e-300\t0.1\t0.05\t0.3\n")  # 极小 p
        code, d, _ = H.run("qc_gwas.py", "--input", f, "--bucket", "x", "--json")
        self.assertIn(code, (0, 1, 2, 3))
        if code == 0:
            self.assertGreaterEqual(d["sanity_issues"]["p_out_of_range"], 2)

    def test_chi2_no_overflow(self):
        """极小 p 不得导致 χ² 溢出崩溃。"""
        v = C.optional  # 只确认函数存在
        import importlib
        qc = importlib.import_module("qc_gwas")
        self.assertGreater(qc.chi2_df1(1e-300), 0)
        self.assertGreater(qc.chi2_df1(0.5), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
