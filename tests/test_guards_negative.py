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

import json
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


class TestGatesPreRegistration(unittest.TestCase):
    """判定门前置落盘（scripts/gates.py）的负向验证。

    这一组专门防一个**很隐蔽**的失效模式：门禁自己有 bug 时，
    "门被改过"这类事件会被静默放行（`ok=true` + 退出码 0）。
    2026-10-12 实测踩到过：指纹不匹配时 `ok` 仍是 true，
    等于"检查指纹"这件事本身完全不设防。所以每条断言都同时查
    **输出内容**与**退出码** —— 只查内容会漏掉"报错了但仍放行"。
    """

    def setUp(self):
        self.proj = Path(tempfile.mkdtemp(prefix="biostat_gates_"))
        (self.proj / "logs").mkdir(parents=True, exist_ok=True)
        self.gfile = self.proj / "logs" / "gates.json"

    def _init(self, *extra):
        return H.run("gates.py", "--init", "--project", self.proj,
                     "--min-detectable-rate", "0.6",
                     "--min-lambda-gc-headroom", "1.1",
                     "--min-positive-control-hits", "1",
                     "--max-agreement-with-existing", "0.95",
                     "--max-p-value", "0.05", "--json", *extra)

    def _edit(self, mutate):
        doc = json.loads(self.gfile.read_text(encoding="utf-8"))
        mutate(doc)
        self.gfile.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    def test_missing_gate_file_fails(self):
        """判定门没落盘 = 门禁最该拦的场景。"""
        code, data, err = H.run("gates.py", "--check", "--project", self.proj, "--json")
        self.assertEqual(code, 1, "缺 gates.json 必须非零退出（否则 CI 会放行）")
        self.assertFalse(data and data.get("ok"))

    def test_fresh_gates_pass(self):
        code, data, _ = self._init()
        self.assertEqual(code, 0, data)
        self.assertTrue(data["ok"])
        code, data, _ = H.run("gates.py", "--check", "--project", self.proj, "--json")
        self.assertEqual(code, 0, "刚落盘、没动过，必须通过")
        self.assertTrue(data["fingerprint_matches"])

    def test_hand_edit_threshold_is_caught(self):
        """**核心用例**：有人绕过 --amend 直接把阈值改松，必须被抓。"""
        self._init()
        self._edit(lambda d: d["gates"][1].update(threshold=0.01))
        code, data, _ = H.run("gates.py", "--check", "--project", self.proj, "--json")
        self.assertEqual(code, 1, "手改阈值必须非零退出")
        self.assertFalse(data["ok"])
        self.assertFalse(data["fingerprint_matches"])

    def test_relax_amend_is_refused(self):
        """事后放宽阈值 = 最常见的事后合理化，必须拒绝。"""
        self._init()
        code, data, _ = H.run("gates.py", "--amend", "--project", self.proj,
                              "--gate", "G2", "--min-detectable-rate", "0.3",
                              "--reason", "想放宽", "--json")
        self.assertEqual(code, 1)
        self.assertFalse(data["ok"])
        doc = json.loads(self.gfile.read_text(encoding="utf-8"))
        g2 = [g for g in doc["gates"] if g["id"] == "G2"][0]
        self.assertAlmostEqual(float(g2["threshold"]), 0.6, places=6,
                               msg="放宽被拒后阈值必须原样不动")

    def test_tighten_amend_allowed_and_traced(self):
        """收紧允许，且必须留痕（旧阈值 + 理由）。"""
        self._init()
        code, data, _ = H.run("gates.py", "--amend", "--project", self.proj,
                              "--gate", "G2", "--min-detectable-rate", "0.85",
                              "--reason", "预检发现检出率偏低", "--json")
        self.assertEqual(code, 0, data)
        doc = json.loads(self.gfile.read_text(encoding="utf-8"))
        g2 = [g for g in doc["gates"] if g["id"] == "G2"][0]
        self.assertAlmostEqual(float(g2["threshold"]), 0.85, places=6)
        self.assertIn("reason", g2["amended"])
        self.assertAlmostEqual(float(g2["amended"]["previous_threshold"]), 0.6,
                               places=6)

    def test_amend_requires_reason(self):
        self._init()
        code, data, _ = H.run("gates.py", "--amend", "--project", self.proj,
                              "--gate", "G2", "--min-detectable-rate", "0.9",
                              "--json")
        self.assertEqual(code, 1, "改门不留理由必须被拦")

    def test_partial_criteria_rejected(self):
        """只给一半判定条件（缺 threshold）必须报错，不许静默当待定。"""
        self._init()
        self._edit(lambda d: [g.pop("threshold", None) for g in d["gates"]
                              if g["id"] == "G3"])
        code, data, _ = H.run("gates.py", "--check", "--project", self.proj, "--json")
        self.assertEqual(code, 1)
        self.assertTrue(any("G3" in e for e in (data or {}).get("errors", [])))

    def test_unknown_gate_rejected(self):
        self._init()
        code, data, _ = H.run("gates.py", "--amend", "--project", self.proj,
                              "--gate", "G99", "--min-positive-control-hits", "2",
                              "--reason", "x", "--json")
        self.assertEqual(code, 1)

    def test_confounder_audit_extended_items(self):
        """混杂审计的 --extended：加项必须改变口径，且缺输入报 cannot_check。"""
        proj = self.proj
        code, data, _ = H.run("confounder_audit.py", "--project", proj, "--json")
        self.assertEqual(code, 0, data)
        self.assertEqual(data["n_items"], 12, "默认仍是十二项（不得改变既有口径）")
        self.assertNotIn("extended", data.get("amended", []))

        pb = proj / "logs" / "pb.json"
        pb.write_text(json.dumps({"n_donors": 12, "n_cells": 8400}), encoding="utf-8")
        code, data, _ = H.run("confounder_audit.py", "--project", proj,
                              "--extended", "--pseudobulk", pb, "--json")
        self.assertEqual(code, 0, data)
        self.assertEqual(data["n_items"], 15)
        names = [i["name"] for i in data["items"]]
        self.assertTrue(any("伪重复" in n for n in names))
        # 缺 --fidelity / --composition 必须是 cannot_check，不能写"未见异常"
        by_name = {i["name"]: i for i in data["items"]}
        fid = [i for i in data["items"] if "整合伪影" in i["name"]][0]
        self.assertEqual(fid["status"], "cannot_check")
        self.assertIsNone(fid["conclusion"])
        comp = [i for i in data["items"] if "细胞组成" in i["name"]][0]
        self.assertEqual(comp["status"], "cannot_check")
        self.assertIsNone(comp["conclusion"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
