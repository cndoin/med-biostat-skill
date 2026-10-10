#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_core.py —— 功能测试：每个脚本在正常输入下给出正确、且口径完整的输出。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import _harness as H  # noqa: E402
from fixtures.make_fixtures import build  # noqa: E402

FIX = {}


def setUpModule():
    global FIX
    FIX = build(tempfile.mkdtemp(prefix="biostat_fix_"))


class TestDoctor(unittest.TestCase):
    def test_doctor_ok(self):
        code, d, _ = H.run("doctor.py", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(d["ok"], d.get("blockers"))
        self.assertEqual(d["core_engine"], "stdlib（无第三方运行时依赖）")
        self.assertEqual(d["missing_files"], [])

    def test_version_matches_file(self):
        """验证脚本不硬编码期望值（P22）：doctor 读 VERSION 文件。"""
        code, d, _ = H.run("doctor.py", "--json")
        v = (H.PKG / "VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(d["skill_version"], v)


class TestQcGwas(unittest.TestCase):
    def test_missing_caliber_errors(self):
        """口径缺失必须报错（exit 3），不得给默认值（R3）。"""
        code, d, _ = H.run("qc_gwas.py", "--input", FIX["gwas_full"], "--json")
        self.assertEqual(code, 3)
        self.assertFalse(d["ok"])
        self.assertIn("bucket", d["missing_caliber"])

    def test_lambda_gc_with_caliber(self):
        code, d, _ = H.run("qc_gwas.py", "--input", FIX["gwas_full"],
                           "--bucket", "maf_0.01_0.5", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(d["ok"])
        # 均匀 p 的合成数据，λGC 应接近 1（显著位点会让它略高）
        self.assertIsNotNone(d["lambda_gc"])
        self.assertLess(d["lambda_gc"], 3.0)
        self.assertEqual(d["caliber"]["lambda_gc_maf_bucket"], "maf_0.01_0.5")
        self.assertIn("不等于独立位点数", d["unit_warning"])

    def test_gw_variant_count(self):
        """变异数口径：SIG 有 5 个 P<5e-8。"""
        code, d, _ = H.run("qc_gwas.py", "--input", FIX["gwas_full"],
                           "--bucket", "maf_0.01_0.5", "--json")
        self.assertGreaterEqual(d["n_gw_significant_variants"], 5)


class TestLociCount(unittest.TestCase):
    def test_variants_vs_loci(self):
        """核心：变异数 ≠ 独立位点数（P08）。"""
        code, d, _ = H.run("loci_count.py", "--input", FIX["gwas_full"],
                           "--p-threshold", "5e-8", "--window-kb", "1000,3000", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(d["ok"])
        self.assertGreater(d["n_raw_variants_below_threshold"],
                           d["sensitivity_matrix"][0]["n_independent_loci"])
        self.assertEqual(d["caliber"]["cluster_method"], "distance")
        self.assertEqual(len(d["sensitivity_matrix"]), 2)

    def test_truncated_flag(self):
        """截断必须标记 truncated=true（P02）。"""
        code, d, _ = H.run("loci_count.py", "--input", FIX["gwas_full"],
                           "--p-threshold", "5e-8", "--max-variants", "50", "--json")
        self.assertTrue(d["truncated"])


class TestLociBackbone(unittest.TestCase):
    def test_good_cs_filter(self):
        """低纯度大可信集必须被剔除并记录行数（R6）。"""
        code, d, _ = H.run("loci_backbone.py", "--finemap", FIX["snp_filter"],
                           "--good-cs", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(d["n_dropped"], 1)
        self.assertEqual(d["dropped_sample"][0]["cs_size"], "21398")
        self.assertTrue(d["caliber"]["filter_enabled"])


class TestAnnotateRegions(unittest.TestCase):
    def test_interval_annotation(self):
        """区间注释必须命中 SOX2 / LIM2（R7 反例场景）。"""
        code, d, _ = H.run("annotate_regions.py", "--regions", FIX["regions"],
                           "--annotation", FIX["genes"], "--json")
        self.assertEqual(code, 0)
        genes = d["regions"][0]["genes"]
        self.assertIn("SOX2", genes)
        self.assertIn("LIM2", genes)
        self.assertIn("interval", d["caliber"]["method"])


class TestCrossVerify(unittest.TestCase):
    def test_allele_order_and_maf(self):
        """同序命中；MAF 折算后偏差≈0（P03/P05）。"""
        code, d, _ = H.run("cross_verify.py", "--full", FIX["gwas_full"],
                           "--subset", FIX["snp_filter"], "--min-hit-rate", "0.8", "--json")
        self.assertEqual(code, 0, d)
        self.assertEqual(d["caliber"]["allele_order"], "same")
        self.assertGreaterEqual(d["hit_rate_same_order"], d["hit_rate_swapped_order"])
        self.assertLess(d["max_abs_dmaf"], 1e-6)   # 折算后应≈0
        self.assertEqual(d["max_abs_dp"], 0)


class TestConfounderAudit(unittest.TestCase):
    def test_twelve_items(self):
        code, d, _ = H.run("confounder_audit.py", "--project", FIX["dir"],
                           "--endpoints", FIX["endpoints"], "--json")
        self.assertEqual(code, 0)
        self.assertEqual(len(d["items"]), 12)
        total = sum(d["counts"].values())
        self.assertEqual(total, 12)
        # 样本重叠应被自动检出（两行对照数相同）
        item3 = [x for x in d["items"] if x["id"] == 3][0]
        self.assertEqual(item3["status"], "checked")
        # 无法查的项必须带 reason（不得写「未见异常」）
        for it in d["items"]:
            if it["status"] == "cannot_check":
                self.assertTrue(it["reason"])


class TestReportPipeline(unittest.TestCase):
    def _md(self, with_untraceable=False):
        extra = "\n结果中位 123456 个月。\n" if with_untraceable else "\n结果显著。\n"
        return ("# 白内障遗传结构\n\n"
                "## 摘要\n本研究报告白内障的遗传结构。\n\n"
                "## 引言\n背景略。\n\n"
                "## 方法\n数据来源与版本见附。统计方法用 logistic 回归。\n\n"
                "## 结果\n共 5 个独立位点，效应量 0.31。\n\n"
                "## 讨论\n局限：组织错配无法排除。\n\n"
                "## 结论\n结论略。\n\n"
                "## 数据与代码可用性\n代码见仓库。\n" + extra)

    def test_structure_gate_pass(self):
        md = H.write(Path(FIX["dir"]) / "draft.md", self._md())
        sd = Path(FIX["dir"]) / "sourcedata"; H.write(sd / "num.txt", "5 0.31")
        code, d, _ = H.run("report_pipeline.py", "--md", md,
                           "--contract", "academic_paper", "--sourcedata", sd,
                           "--out", str(Path(FIX["dir"]) / "out.html"), "--json")
        self.assertEqual(code, 0, d)
        self.assertTrue(d["gates"]["structure"]["pass"])
        self.assertTrue(d["gates"]["number_traceability"]["pass"])

    def test_number_gate_fail(self):
        """含不可追溯数字 → 数字门禁 FAIL，且不写出文件（R2/R10）。"""
        md = H.write(Path(FIX["dir"]) / "draft2.md", self._md(True))
        sd = Path(FIX["dir"]) / "sourcedata2"; H.write(sd / "num.txt", "5 0.31")
        code, d, _ = H.run("report_pipeline.py", "--md", md,
                           "--contract", "academic_paper", "--sourcedata", sd,
                           "--out", str(Path(FIX["dir"]) / "out2.html"), "--json")
        self.assertEqual(code, 4)
        self.assertFalse(d["ok"])
        self.assertFalse(d["gates"]["number_traceability"]["pass"])
        self.assertIsNone(d["out"])


class TestTriggerCheck(unittest.TestCase):
    """自动触发自检：description 必须覆盖真实问法，且有「不触发」边界。"""

    def test_trigger_coverage(self):
        code, d, _ = H.run("trigger_check.py", "--json")
        self.assertEqual(code, 0, d)
        self.assertTrue(d["ok"])
        self.assertEqual(d["positive"]["covered"], d["positive"]["n"])
        self.assertEqual(d["negative"]["false_positives"], 0)
        self.assertTrue(d["boundary_declared"])

    def test_trigger_check_can_fail(self):
        """反向验证：把 description 抽空的 skill 必须被判 FAIL（证明检查不是摆设）。"""
        import tempfile
        from pathlib import Path as _P
        bad = _P(tempfile.mkdtemp()) / "SKILL.md"
        bad.write_text("---\nname: x\ndescription: 一个无关的技能\n---\n\n# x\n",
                       encoding="utf-8")
        code, d, _ = H.run("trigger_check.py", "--skill", str(bad), "--json")
        self.assertNotEqual(code, 0)
        self.assertFalse(d["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
