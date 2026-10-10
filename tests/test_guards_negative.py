#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_guards_negative.py —— 门禁负向验证（证明门禁有效，不是摆设）

元规则（来自用户明确要求）：
  * 门禁必须先被证明**会失败**，否则无法区分「门禁有效」与「门禁是摆设」；
  * 测试不能只测「该报警的场景报警了」，必须造**反向样本**测假阳性；
  * 用退化检测器（恒定输出）做对照，证明真门禁有分辨力。

本文件全部是「应当失败」与「不应误报」两类断言。
"""

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
    FIX = build(tempfile.mkdtemp(prefix="biostat_neg_"))


# ---------- 1. 口径门禁：缺失必须 FAIL，不得默认放行 ----------
class TestCaliberGate(unittest.TestCase):
    def test_missing_caliber_fails(self):
        code, d, _ = H.run("qc_gwas.py", "--input", FIX["gwas_full"], "--json")
        self.assertEqual(code, 3, "口径缺失必须 exit 3")
        self.assertIn("bucket", d["missing_caliber"])

    def test_allow_missing_downgrades_to_unspecified(self):
        """显式放行时必须留下 UNSPECIFIED 痕迹，不是静默给个默认值。"""
        code, d, _ = H.run("qc_gwas.py", "--input", FIX["gwas_full"],
                           "--allow-missing-caliber", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(d["caliber"]["lambda_gc_maf_bucket"], "UNSPECIFIED")


# ---------- 2. 数字门禁（抗幻觉）----------
class TestNumberGate(unittest.TestCase):
    def _contract_sections(self):
        return ("## 摘要\nx\n\n## 引言\nx\n\n## 方法\nx\n\n## 结果\nx\n\n"
                "## 讨论\nx\n\n## 结论\nx\n\n## 数据与代码可用性\nx\n")

    def test_plausible_fake_number_is_blocked(self):
        """故意制造「看起来合理」的数字（123456），必须被拦住。"""
        md = H.write(Path(FIX["dir"]) / "fake.md",
                     "# 报告\n\n" + self._contract_sections()
                     + "\n共有 123456 个独立位点。\n")
        sd = Path(FIX["dir"]) / "src_fake"; H.write(sd / "known.txt", "5 0.31")
        code, d, _ = H.run("report_pipeline.py", "--md", md,
                           "--contract", "academic_paper", "--sourcedata", sd,
                           "--out", str(sd / "fake.html"), "--json")
        self.assertEqual(code, 4)
        self.assertIn("123456", d["gates"]["number_traceability"]["untraceable_sample"])

    def test_traceable_number_passes(self):
        """反向样本：可追溯的数字不得被误报。"""
        md = H.write(Path(FIX["dir"]) / "real.md",
                     "# 报告\n\n" + self._contract_sections()
                     + "\n共有 5 个独立位点，效应量 0.31。\n")
        sd = Path(FIX["dir"]) / "src_real"; H.write(sd / "known.txt", "5 0.31")
        code, d, _ = H.run("report_pipeline.py", "--md", md,
                           "--contract", "academic_paper", "--sourcedata", sd,
                           "--out", str(sd / "real.html"), "--json")
        self.assertEqual(code, 0, d)
        self.assertTrue(d["gates"]["number_traceability"]["pass"])


# ---------- 3. 结构门禁 ----------
class TestStructureGate(unittest.TestCase):
    def test_missing_section_fails(self):
        md = H.write(Path(FIX["dir"]) / "short.md",
                     "# 报告\n\n## 摘要\nx\n\n## 引言\nx\n")   # 缺方法/结果/...
        # 契约 academic_paper 要求 require_source_data: true —— 必须提供，
        # 否则门禁在数字那一关就返回 FAIL，根本走不到结构判定（实测踩过：
        # 这条用例曾因缺 --sourcedata 而拿不到 gates 键）。
        sd = Path(FIX["dir"]) / "src_struct"; H.write(sd / "n.txt", "1 2 3")
        code, d, _ = H.run("report_pipeline.py", "--md", md,
                           "--contract", "academic_paper", "--sourcedata", sd,
                           "--json")
        self.assertEqual(code, 4)
        self.assertFalse(d["gates"]["structure"]["pass"])
        self.assertIn("方法", d["gates"]["structure"]["missing"])

    def test_missing_sourcedata_is_rejected(self):
        """契约要求 source data 时，不给就必须 FAIL（不得「没检查却报通过」）。"""
        md = H.write(Path(FIX["dir"]) / "ok.md", "# 报告\n\n## 摘要\n1234\n")
        code, d, _ = H.run("report_pipeline.py", "--md", md,
                           "--contract", "academic_paper", "--json")
        self.assertEqual(code, 4, d)
        self.assertFalse(d["ok"])
        self.assertIn("require_source_data", d["error"])


# ---------- 4. 一致性门禁：系统性偏差必须 FAIL ----------
class TestConsistencyGate(unittest.TestCase):
    def _bad_subset(self):
        """构造一个「p 值被整体改过」的子集 → 应判为系统性偏差。"""
        src = Path(FIX["snp_filter"]).read_text(encoding="utf-8").splitlines()
        out = [src[0]]
        for ln in src[1:]:
            f = ln.split("\t")
            try:
                f[6] = repr(float(f[6]) * 1e6)   # 人为篡改 p
            except (ValueError, IndexError):
                pass
            out.append("\t".join(f))
        p = Path(FIX["dir"]) / "tampered.tsv"
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        return p

    def test_systematic_bias_fails(self):
        bad = self._bad_subset()
        code, d, _ = H.run("cross_verify.py", "--full", FIX["gwas_full"],
                           "--subset", bad, "--json")
        self.assertEqual(code, 4, "篡改后必须 exit 4")
        self.assertFalse(d["ok"])
        self.assertIn("系统性偏差", d["verdict"])

    def test_normal_input_not_flagged(self):
        """反向样本：正常输入不得被判为偏差（测假阳性）。"""
        code, d, _ = H.run("cross_verify.py", "--full", FIX["gwas_full"],
                           "--subset", FIX["snp_filter"], "--min-hit-rate", "0.8", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(d["verdict"], "一致")

    def test_hit_rate_threshold_is_live(self):
        """把命中率下限拉到 100%，有缺口的输入必须 FAIL —— 证明该参数真的在起作用。"""
        code, d, _ = H.run("cross_verify.py", "--full", FIX["gwas_full"],
                           "--subset", FIX["snp_filter"], "--min-hit-rate", "1.0", "--json")
        self.assertEqual(code, 4)
        self.assertLess(d["hit_rate"], 1.0)


# ---------- 5. 可信集纯度门禁：不过滤 vs 过滤 必须不同 ----------
class TestPurityGate(unittest.TestCase):
    def test_filter_changes_result(self):
        _, no_filter, _ = H.run("loci_backbone.py", "--finemap", FIX["snp_filter"], "--json")
        _, filtered, _ = H.run("loci_backbone.py", "--finemap", FIX["snp_filter"],
                               "--good-cs", "--json")
        self.assertEqual(no_filter["n_dropped"], 0)
        self.assertEqual(filtered["n_dropped"], 1)
        self.assertNotEqual(no_filter["n_kept"], filtered["n_kept"])

    def test_missing_cs_size_column_rejected(self):
        """缺 cs_size 列时拒绝静默放行（口径不完整）。"""
        p = Path(FIX["dir"]) / "nocsize.tsv"
        p.write_text("chromosome\tposition\tp\nchr1\t1000\t1e-9\n", encoding="utf-8")
        code, d, _ = H.run("loci_backbone.py", "--finemap", p, "--good-cs", "--json")
        self.assertEqual(code, 3)


# ---------- 6. 元测试：证明门禁有分辨力（对照退化检测器）----------
class TestGateHasDiscrimination(unittest.TestCase):
    def test_degenerate_gate_would_pass_bad_input(self):
        """退化门禁（恒返回 True）对坏输入仍放行 —— 我们的门禁必须不同。"""
        def degenerate_gate(_result):        # 退化检测器：恒定输出
            return True

        # 真实门禁对同一坏输入给出 False（FAIL）
        bad = Path(FIX["dir"]) / "degen_probe.tsv"
        bad.write_text("#chrom\tpos\tref\talt\tpval\n1\t1000\tA\tG\t0.5\n",
                       encoding="utf-8")
        code, d, _ = H.run("cross_verify.py", "--full", FIX["gwas_full"],
                           "--subset", bad, "--min-hit-rate", "1.0", "--json")
        real_gate_pass = (code == 0)

        # 退化门禁说「通过」，真门禁说「不通过」→ 两者可区分
        self.assertTrue(degenerate_gate(d))
        self.assertFalse(real_gate_pass, "真门禁必须能对坏输入说 FAIL")
        self.assertNotEqual(degenerate_gate(d), real_gate_pass)


if __name__ == "__main__":
    unittest.main(verbosity=2)
