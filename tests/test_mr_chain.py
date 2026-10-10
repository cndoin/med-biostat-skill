#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_mr_chain.py —— 药物靶点 MR 链自检（Tier0 纯标准库）

覆盖新增的 P0 模块：cis 工具变量、Wald ratio、ABF 共定位、检验家族、复现检查、确定性管道。

关键点：这些用例**断言等于指定数值**（不是「不崩溃」），并对最容易写错的地方
设反向样本 —— 特别是共定位的 H3（不同因果变异）与 H4（共享因果变异）必须被区分开。
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
sys.path.insert(0, str(TESTS))
import _harness as H  # noqa: E402
from fixtures.make_mr_fixtures import build  # noqa: E402

FIX = {}


def setUpModule():
    global FIX
    FIX = build(tempfile.mkdtemp(prefix="biostat_mr_"))


def jrun(script, *args):
    """跑脚本并按人类输出解析（脚本 stdout 是 JSON，stderr 是日志）。"""
    return H.run(script, *args)


class TestCisInstrument(unittest.TestCase):
    def test_palindromic_and_window(self):
        code, d, _ = H.run("cis_instrument.py", "--eqtl", FIX["dir"] + "/eqtl.tsv",
                           "--genes", FIX["dir"] + "/genes.tsv", "--json")
        self.assertEqual(code, 0, d)
        ins = {(r["gene"], r["tissue"]): r for r in d["instruments"]}
        keys = set(ins)
        # 回文 SNP（A/T，P=1e-40 最强）必须被排除，因此选出的是次强的 A/G
        self.assertIn(("PTGDS", "Whole_Blood"), keys)
        self.assertEqual(ins[("PTGDS", "Whole_Blood")]["lead"]["ref"], "A")
        self.assertEqual(ins[("PTGDS", "Whole_Blood")]["lead"]["alt"], "G")
        self.assertGreater(d["n_palindromic_excluded"], 0)
        # F = (0.5/0.04)² = 156.25
        self.assertAlmostEqual(ins[("PTGDS", "Whole_Blood")]["lead"]["F"], 156.25, places=2)
        # 窗口外变异（chr9:139000000）不得入选；只对 PTGDS 检查（CACNA1C 是另一基因）
        for r in d["instruments"]:
            if r["gene"] == "PTGDS":
                self.assertLess(abs(r["lead"]["pos"] - 136900000), 20000)

    def test_weak_instrument_reported_unavailable(self):
        code, d, _ = H.run("cis_instrument.py", "--eqtl", FIX["dir"] + "/eqtl.tsv",
                           "--genes", FIX["dir"] + "/genes.tsv", "--json")
        un = {(r["gene"], r["tissue"]) for r in d["unavailable"]}
        self.assertIn(("PTGDS", "Artery_Tibial"), un)   # F=4 < 10


class TestWaldMR(unittest.TestCase):
    def test_wald_and_delta_se(self):
        """Wald = β_out/β_exp；se 用 delta 法；手算对拍。"""
        code, d, _ = H.run("wald_mr.py", "--instruments", FIX["dir"] + "/mr_input.tsv", "--json")
        self.assertEqual(code, 0, d)
        r = [x for x in d["results"] if x["tissue"] == "Whole_Blood"
             and x["beta_exposure"] == 0.5][0]
        self.assertAlmostEqual(r["beta"], 0.5, places=10)          # 0.25/0.5
        # se = sqrt(0.08²/0.5² + 0.25²·0.04²/0.5⁴) = sqrt(0.0256+0.0016)
        self.assertAlmostEqual(r["se"], math.sqrt(0.0272), places=10)
        self.assertAlmostEqual(r["F"], 156.25, places=2)

    def test_weak_marked_and_zero_exposure_unestimable(self):
        code, d, _ = H.run("wald_mr.py", "--instruments", FIX["dir"] + "/mr_input.tsv", "--json")
        weak = [x for x in d["results"] if x["weak_instrument"]]
        self.assertEqual(len(weak), 1)
        self.assertEqual(weak[0]["F"], 4.0)
        unest = [x for x in d["results"] if x["beta"] is None]
        self.assertEqual(len(unest), 1)

    def test_rejects_multi_instrument_methods(self):
        """口径里必须写明单工具禁用 Egger/异质性/留一法。"""
        code, d, _ = H.run("wald_mr.py", "--instruments", FIX["dir"] + "/mr_input.tsv", "--json")
        self.assertEqual(d["caliber"]["multi_instrument_methods"], "REJECTED")
        self.assertIn("不得", d["caliber"]["note"])


class TestAbfColoc(unittest.TestCase):
    def test_shared_vs_distinct_causal_variant(self):
        """核心：共享因果变异 → 高 H4；不同因果变异 → 高 H3、低 H4。"""
        _, shared, _ = H.run("abf_coloc.py",
                             "--trait1", FIX["dir"] + "/trait1_region.tsv",
                             "--trait2", FIX["dir"] + "/trait2_region.tsv", "--json")
        _, distinct, _ = H.run("abf_coloc.py",
                               "--trait1", FIX["dir"] + "/trait1_region.tsv",
                               "--trait2", FIX["dir"] + "/trait2_null.tsv", "--json")
        self.assertGreater(shared["main"]["PP.H4"], 0.9)
        self.assertLess(distinct["main"]["PP.H4"], 0.1)
        self.assertGreater(distinct["main"]["PP.H3"], 0.9)

    def test_posteriors_sum_to_one(self):
        _, d, _ = H.run("abf_coloc.py", "--trait1", FIX["dir"] + "/trait1_region.tsv",
                        "--trait2", FIX["dir"] + "/trait2_region.tsv", "--json")
        m = d["main"]
        total = sum(m[k] for k in ("PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4"))
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_sensitivity_matrix_shape(self):
        _, d, _ = H.run("abf_coloc.py", "--trait1", FIX["dir"] + "/trait1_region.tsv",
                        "--trait2", FIX["dir"] + "/trait2_region.tsv",
                        "--p12-list", "1e-6,5e-6,1e-5", "--sdy-list", "0.5,1,1.5", "--json")
        self.assertEqual(len(d["sensitivity_matrix"]), 9)   # 3 p12 × 3 sdY

    def test_insufficient_region_refuses_to_claim(self):
        """区域 SNP 太少时必须拒绝，不得声称共定位（PROTOCOL 硬约束）。"""
        p1 = Path(FIX["dir"]) / "tiny1.tsv"
        p2 = Path(FIX["dir"]) / "tiny2.tsv"
        p1.write_text("snp\tbeta\tse\nrs1\t0.1\t0.05\n", encoding="utf-8")
        p2.write_text("snp\tbeta\tse\nrs1\t0.1\t0.05\n", encoding="utf-8")
        code, d, _ = H.run("abf_coloc.py", "--trait1", p1, "--trait2", p2, "--json")
        self.assertNotEqual(code, 0)
        self.assertFalse(d["ok"])
        self.assertIn("不得", d["error"])


class TestTestingFamily(unittest.TestCase):
    def test_bonferroni_and_bh_values(self):
        code, d, _ = H.run("testing_family.py", "--input", FIX["dir"] + "/tests.tsv", "--json")
        self.assertEqual(code, 0, d)
        t = {x["test_id"]: x for x in d["tests"]}
        # main 用 Bonferroni：0.0029*2=0.0058 显著；0.031*2=0.062 不显著
        self.assertEqual(t["PTGDS_blood"]["correction"], "Bonferroni")
        self.assertAlmostEqual(t["PTGDS_blood"]["p_adjusted"], 0.0058, places=6)
        self.assertTrue(t["PTGDS_blood"]["significant"])
        self.assertFalse(t["CACNA1C_artery"]["significant"])
        # secondary 用 BH
        self.assertEqual(t["PTGDS_aorta"]["correction"], "BH-FDR")
        self.assertEqual(d["families"]["main"]["n_tests"], 2)
        self.assertEqual(d["families"]["secondary"]["n_tests"], 3)

    def test_bh_monotone(self):
        """BH 校正后 q 必须随 p 单调不减（手算对拍）。"""
        code, d, _ = H.run("testing_family.py", "--input", FIX["dir"] + "/tests.tsv", "--json")
        sec = sorted([x for x in d["tests"] if x["family"] == "secondary"],
                     key=lambda x: x["p"])
        qs = [x["p_adjusted"] for x in sec]
        self.assertEqual(qs, sorted(qs))


class TestReplicationCheck(unittest.TestCase):
    def test_direction_overlap_ancestry(self):
        code, d, _ = H.run("replication_check.py",
                           "--discovery", FIX["dir"] + "/disc.tsv",
                           "--replication", FIX["dir"] + "/rep.tsv", "--json")
        self.assertEqual(code, 0, d)
        self.assertEqual(d["counts"]["n_flipped"], 1)              # CACNA1C 方向翻转
        self.assertEqual(len(d["sample_overlap_flags"]), 1)        # PTGDS 对照数相同
        self.assertEqual(len(d["ancestry_mismatch"]), 1)           # CACNA1C EUR vs EAS
        self.assertTrue(d["caliber"]["no_pooling"])


class TestPipeline(unittest.TestCase):
    def _cfg(self):
        cfg = {
            "study": "test_chain",
            "steps": {
                "cis_instrument": {"eqtl": FIX["dir"] + "/eqtl.tsv",
                                   "genes": FIX["dir"] + "/genes.tsv", "window_kb": 100},
                "wald_mr": {"instruments": FIX["dir"] + "/mr_input.tsv"},
                "abf_coloc": [{"trait1": FIX["dir"] + "/trait1_region.tsv",
                               "trait2": FIX["dir"] + "/trait2_region.tsv", "p12": 1e-5}],
                "testing_family": {"input": FIX["dir"] + "/tests.tsv"},
                "replication_check": {"discovery": FIX["dir"] + "/disc.tsv",
                                      "replication": FIX["dir"] + "/rep.tsv"},
            },
        }
        p = Path(FIX["dir"]) / "cfg.json"
        p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        return p

    def test_full_chain_and_manifest(self):
        outdir = Path(FIX["dir"]) / "run1"
        cp = subprocess.run([sys.executable, str(H.PKG / "pipeline" / "run_drug_target_mr.py"),
                             "--config", str(self._cfg()), "--outdir", str(outdir), "--json"],
                            capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        rep = json.loads(cp.stdout)
        self.assertTrue(rep["ok"])
        self.assertEqual(len(rep["steps"]), 5)
        m = json.loads((outdir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(m["all_ok"])
        self.assertTrue(m["config_sha256"])
        self.assertTrue(all(s["out_sha256"] for s in m["steps"]))

    def test_manifest_deterministic(self):
        """同一 config + 同一输入 → 产物哈希必须一致（可复现性）。"""
        out1 = Path(FIX["dir"]) / "det1"
        out2 = Path(FIX["dir"]) / "det2"
        for o in (out1, out2):
            subprocess.run([sys.executable, str(H.PKG / "pipeline" / "run_drug_target_mr.py"),
                            "--config", str(self._cfg()), "--outdir", str(o), "--json"],
                           capture_output=True, text=True, encoding="utf-8")
        m1 = json.loads((out1 / "run_manifest.json").read_text(encoding="utf-8"))
        m2 = json.loads((out2 / "run_manifest.json").read_text(encoding="utf-8"))
        h1 = {s["step"] + s["tag"]: s["out_sha256"] for s in m1["steps"]}
        h2 = {s["step"] + s["tag"]: s["out_sha256"] for s in m2["steps"]}
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
