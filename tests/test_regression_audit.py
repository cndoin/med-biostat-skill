"""test_regression_audit.py —— 2026-10-10 深度审计 14 条 bug 的常驻回归用例

**为什么单独建一个文件**：这 14 条 bug 是在自检报 70/70 全绿的情况下
被深度审计挖出来的 —— 它们全在自检盲区里。逐条固化成用例，才不会在
下一次重构里悄悄回来。

每条用例都遵循同一个纪律：
  1. 用**修复前真实触发**的输入构造场景（不是新编的等价场景）；
  2. 断言的是**修复后的正确行为**，而不是「跑通就行」；
  3. 涉及数值的尽量与权威实现（lifelines / sklearn）对拍。

**反向验证**：TestMetaGates.test_crash_detector_catches_real_crash 用一个
真会崩的脚本证明崩溃检测本身有效 —— 否则这套断言可能形同虚设。
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import _harness as H
from _harness import PKG, PY, SCRIPTS, write as _write

# 直接 import 本 skill 的模块做数值对拍，需要把包根与 scripts/ 加进 sys.path
for _p in (PKG, PKG / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

AUDIT = PKG / "tests" / ".audit_tmp"


def _w(rel: str, text: str) -> Path:
    return H.write(AUDIT / rel, text)


# ============================================================ P0-1 c_index
@unittest.skipUnless(H.has("numpy"),
                     "lib.increment 需要 numpy（当前测试进程没有；"
                     "请用带科学栈的解释器运行 run_all.py，或设 BIOSTAT_SCI_PYTHON）")
class TestCIndexRegression(unittest.TestCase):
    """lib/increment.py c_index：误纳「删失者在前」的 pair。"""

    def test_censored_first_pair_is_not_counted(self):
        """2 人数据：A 在 t=1 删失，B 在 t=2 发生事件。

        Harrell / lifelines 定义下该 pair **不可比**（先后不可知）。
        修复前会给出 c=1.0 —— 换一个看似无关的风险值就翻成 0.0。
        """
        from lib import increment as inc
        r = inc.c_index([1, 2], [0, 1], [5.0, 1.0])
        self.assertEqual(r["n_pairs"], 0,
                         "先删失后事件的 pair 不可比，不该计入")
        self.assertTrue(r["c"] != r["c"], "无可比对时应为 NaN，不得给数字")

    def test_tied_time_exactly_one_event_is_comparable(self):
        """同刻时间、恰好一个事件 → **可比**（这是最易漏的一条规则）。"""
        from lib import increment as inc
        r = inc.c_index([5, 5], [1, 0], [0.9, 0.1])
        self.assertEqual(r["n_pairs"], 1, "同刻且恰好一个事件应可比")
        self.assertEqual(r["c"], 1.0, "高风险者恰为事件者 → 完全一致")

    def test_tied_time_both_events_is_not_comparable(self):
        from lib import increment as inc
        r = inc.c_index([5, 5], [1, 1], [0.9, 0.1])
        self.assertEqual(r["n_pairs"], 0, "同刻且都事件 → 无法定先后")

    @unittest.skipUnless(H.has("lifelines"), "需要 lifelines 作为金标准")
    def test_matches_lifelines_with_censoring(self):
        """与 lifelines 对拍（修复前最大偏差 0.25）。"""
        import numpy as np
        from lifelines.utils import concordance_index
        from lib import increment as inc

        rng = np.random.default_rng(2024)
        worst = 0.0
        n = 0
        for _ in range(300):
            k = 30
            t = np.sort(rng.integers(1, 200, k).astype(float))
            e = rng.integers(0, 2, k)
            if e.sum() in (0, k):
                continue
            r_ = rng.normal(size=k)
            mine = inc.c_index(t, e, r_)["c"]
            if mine != mine:      # NaN
                continue
            worst = max(worst, abs(mine - concordance_index(t, -r_, e)))
            n += 1
        self.assertGreater(n, 100, "有效样本太少，用例本身失去意义")
        self.assertLess(worst, 1e-12, f"与 lifelines 偏差 {worst:.3e} 过大")


# ============================================================ P0-2 roc_auc
@unittest.skipUnless(H.has("numpy"),
                     "vendor.engines 需要 numpy（当前测试进程没有；"
                     "请用带科学栈的解释器运行 run_all.py，或设 BIOSTAT_SCI_PYTHON）")
class TestRocAucRegression(unittest.TestCase):
    """vendor/engines/modeling.py roc_curve：并列分数时折点左移，AUC 高估。"""

    def test_minimal_tie_case(self):
        from vendor.engines import modeling as md
        y = [0, 0, 1, 1, 1, 0, 1, 0]
        s = [0.1, 0.2, 0.3, 0.3, 0.5, 0.5, 0.7, 0.8]
        self.assertAlmostEqual(md.roc_curve(y, s)[2], 0.5938, places=4)

    @unittest.skipUnless(H.has("sklearn"), "需要 sklearn 作为金标准")
    def test_matches_sklearn_on_tied_scores(self):
        """离散评分卡场景 —— 自检原来只测连续分数，偏差 1e-16 落在噪声内。"""
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from vendor.engines import modeling as md

        rng = np.random.default_rng(99)
        for levels in (3, 11, 21):
            for _ in range(60):
                y = rng.integers(0, 2, 150)
                if y.sum() in (0, 150):
                    continue
                s = rng.integers(0, levels, 150).astype(float)
                self.assertAlmostEqual(md.roc_curve(y, s)[2],
                                       roc_auc_score(y, s), places=10)


# ============================================================ P0-3 NaN
class TestNonFinitePval(unittest.TestCase):
    """scripts/qc_gwas.py：p=NaN 静默混入 λGC 并产出非法 JSON。"""

    GWAS = "chr\tpos\tp\taf\n1\t100\t1e-20\t0.2\n1\t200\tNaN\t0.3\n1\t300\tnan\t0.25\n1\t400\t0.5\t0.1\n"

    def _run(self, tmp_name: str = "g.tsv"):
        p = _w(f"p0_3/{tmp_name}", self.GWAS)
        code, d, err = H.run_clean(
            "qc_gwas.py", "--input", p, "--maf-min", "0.05", "--maf-max", "0.5",
            "--n-case", "1000", "--n-control", "1000", "--json")
        return code, d, err

    def test_nan_pval_is_counted_not_silently_used(self):
        code, d, _ = self._run()
        self.assertEqual(code, 0, d)
        self.assertEqual(d["sanity_issues"]["p_not_finite"], 2,
                         "两个 NaN 必须被显式计数")

    def test_output_is_strict_json(self):
        """RFC 8259 不允许 NaN 字面量；Python 的 json 会静默接受它。"""
        _, d, _ = self._run()
        text = json.dumps(d)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        json.loads(text, parse_constant=lambda c: self.fail(
            f"输出含非法 JSON 常量 {c}"))

    def test_emit_rejects_nonfinite(self):
        """_common.emit 必须拦住非有限浮点（收口层）。

        NaN 是 RFC 8259 非法常量，而 Python 的 json 会静默接受它 →
        输出能过自己的解析，却会被下游绝大多数严格解析器拒绝。
        """
        import contextlib
        import io

        import _common as C
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            C.emit({"ok": True, "lam": float("nan")}, as_json=True)
        out = buf.getvalue()
        json.loads(out, parse_constant=lambda c: self.fail(f"非法常量 {c}"))
        self.assertIn("null", out, "非有限值应被置为 null 而非 NaN")


# ============================================================ P0-4 se 守卫
class TestWaldSeGuard(unittest.TestCase):
    """scripts/wald_mr.py：se_exposure<=0 → F=inf 被判「非弱工具」。"""

    def test_zero_se_rejected(self):
        import wald_mr as wm
        r = wm.wald(0.5, 0.0, 0.1, 0.05)
        self.assertIsNone(r["beta"], "se=0 时不可估")
        self.assertIn("se_exposure", r["reason"])

    def test_negative_se_rejected(self):
        import wald_mr as wm
        r = wm.wald(0.5, -0.04, 0.1, 0.05)
        self.assertIsNone(r["beta"], "负 se 是非法输入，不得平方后当有效")

    def test_normal_value_unchanged(self):
        import wald_mr as wm
        r = wm.wald(0.5, 0.04, 0.1, 0.05)
        self.assertAlmostEqual(r["F"], 156.25, places=6)

    def test_or_overflow_does_not_crash(self):
        """beta=699 时指数 699.196 溢出；旧守卫 abs(beta)<700 放行后崩掉。"""
        import wald_mr as wm
        r = wm.wald(0.001, 1.0, 0.699, 0.1)   # beta = 699
        self.assertNotIn("OverflowError", str(r))
        self.assertIsNotNone(r["beta"])


# ============================================================ P1 BOM
class TestBomHandling(unittest.TestCase):
    """scripts/_common.open_text：BOM 使列名匹配静默失效。"""

    def test_bom_file_is_parsed_like_plain(self):
        p = AUDIT / "p1_bom" / "g.tsv"
        p.parent.mkdir(parents=True, exist_ok=True)
        # 用 utf-8-sig 写出 BOM 文件，且制造一个重复变异供检出
        p.write_bytes("chr\tpos\tp\taf\n1\t100\t1e-20\t0.2\n1\t100\t1e-21\t0.2\n"
                      "1\t200\t0.5\t0.1\n".encode("utf-8-sig"))
        self.assertTrue(p.read_bytes().startswith(b"\xef\xbb\xbf"), "确认为 BOM 文件")
        code, d, err = H.run_clean(
            "qc_gwas.py", "--input", p, "--maf-min", "0.05", "--maf-max", "0.5",
            "--n-case", "1000", "--n-control", "1000", "--json")
        self.assertEqual(code, 0, err)
        self.assertEqual(d["sanity_issues"]["dup_keys"], 1,
                         "带 BOM 时也必须检出重复变异（修复前恒为 0）")


# ============================================================ P1 无法判定
class TestCannotCheck(unittest.TestCase):
    """scripts/confounder_audit.py：缺列时不得写「未见异常」。"""

    def test_missing_column_is_cannot_check(self):
        proj = AUDIT / "p1_conf"; (proj / "logs").mkdir(parents=True, exist_ok=True)
        ep = _w("p1_conf/ep.tsv", "endpoint\tn_case\nA\t100\nB\t200\n")
        code, d, err = H.run_clean("confounder_audit.py", "--project", proj,
                                   "--endpoints", ep, "--json")
        self.assertEqual(code, 0, err)
        rec = [i for i in d["items"] if i["name"] == "样本重叠"][0]
        self.assertEqual(rec["status"], "cannot_check")
        self.assertIn("不得读作无重叠", rec["conclusion"])

    def test_shared_control_still_detected(self):
        """反向保护：能查时必须仍然检出（防止把修复写成「永远 cannot_check」）。"""
        proj = AUDIT / "p1_conf2"; (proj / "logs").mkdir(parents=True, exist_ok=True)
        ep = _w("p1_conf2/ep.tsv",
                "endpoint\tn_control\nA\t409535\nB\t409535\nC\t300000\n")
        code, d, err = H.run_clean("confounder_audit.py", "--project", proj,
                                   "--endpoints", ep, "--json")
        self.assertEqual(code, 0, err)
        rec = [i for i in d["items"] if i["name"] == "样本重叠"][0]
        self.assertEqual(rec["status"], "checked")
        self.assertIn("存在共享对照", rec["conclusion"])


# ============================================================ P1 keep-out
class TestKeepOutColumns(unittest.TestCase):
    """scripts/loci_backbone.py：--keep-out 写出空列。"""

    def test_original_columns_preserved(self):
        src = _w("p1_keep/fm.tsv",
                 "chromosome\tposition\tp\tcs_size\tcs_min_r2\n"
                 "9\t136900000\t1e-9\t25\t0.85\n"
                 "9\t136950000\t1e-8\t900\t0.60\n")
        out = AUDIT / "p1_keep" / "keep.tsv"
        code, d, err = H.run_clean("loci_backbone.py", "--finemap", src,
                                   "--good-cs", "--keep-out", out, "--json")
        self.assertEqual(code, 0, err)
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(lines[0], "chromosome\tposition\tp\tcs_size\tcs_min_r2")
        cells = lines[1].split("\t")
        self.assertEqual(cells[0], "9", "chromosome 列不得为空")
        self.assertEqual(cells[1], "136900000", "position 列不得为空")


# ============================================================ P1 数字门禁
class TestNumberGateBypass(unittest.TestCase):
    """scripts/report_pipeline.py：子串匹配可放行幻觉数字。"""

    def _run(self, md_text: str, pool_text: str = "1999999\n"):
        base = AUDIT / "p1_num"
        sd = base / "sd"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "n.txt").write_text(pool_text, encoding="utf-8")
        md = _w("p1_num/d.md", md_text)
        code, d, err = H.run_clean("report_pipeline.py", "--md", md,
                                   "--contract", "academic_paper",
                                   "--sourcedata", sd, "--json")
        return code, d

    def test_substring_hallucination_is_blocked(self):
        """报告写 999999，source data 只有 1999999 → 旧实现按子串放行。"""
        code, d = self._run("# T\n\n结果为 999999。\n")
        g = d["gates"]["number_traceability"]
        self.assertFalse(g["pass"], "999999 是 1999999 的子串，必须拦下")
        self.assertIn("999999", g["untraceable_sample"])

    def test_legitimate_number_passes(self):
        code, d = self._run("# T\n\n结果为 1999999。\n")
        self.assertTrue(d["gates"]["number_traceability"]["pass"])

    def test_thousands_separator_is_same_number(self):
        code, d = self._run("# T\n\n结果为 1,999,999。\n")
        self.assertTrue(d["gates"]["number_traceability"]["pass"],
                        "千分位写法应视为同一数字，否则误报")


# ============================================================ P2 崩溃
class TestNoCrashOnBadInput(unittest.TestCase):
    """P2 四条：曾经的 IndexError / FileNotFoundError / OverflowError。"""

    def test_empty_window_kb_is_clean_error(self):
        src = _w("p2_crash/g.tsv", "chr\tpos\tp\taf\n1\t100\t1e-20\t0.2\n")
        for bad in ("", "  ", ",,", "abc"):
            code, d, err = H.run_clean("loci_count.py", "--input", src,
                                       "--window-kb", bad, "--json")
            self.assertNotEqual(code, 0, f"{bad!r} 应当失败")
            self.assertEqual(code, 2, f"{bad!r} 应当是干净的输入错误 exit 2")
            self.assertIsNotNone(d, f"{bad!r} 必须给出可解析 JSON")
            self.assertFalse(d["ok"])

    def test_missing_genes_file_is_clean_error(self):
        src = _w("p2_crash/g2.tsv", "chr\tpos\tp\taf\n1\t100\t1e-20\t0.2\n")
        code, d, err = H.run_clean("cis_instrument.py", "--eqtl", src,
                                   "--genes", AUDIT / "p2_crash" / "nope.tsv",
                                   "--json")
        self.assertEqual(code, 2, err)
        self.assertIn("不存在", d["error"])


# ============================================================ P2 pipeline
class TestPipelineConfigValidation(unittest.TestCase):
    """pipeline/run_drug_target_mr.py：steps 类型错误曾崩成 AttributeError。"""

    def _run(self, cfg: dict, name: str):
        base = AUDIT / "p2_pipe"
        base.mkdir(parents=True, exist_ok=True)
        cf = base / f"{name}.json"
        cf.write_text(json.dumps(cfg), encoding="utf-8")
        cp = subprocess.run(
            [PY, str(PKG / "pipeline" / "run_drug_target_mr.py"),
             "--config", str(cf), "--outdir", str(base / f"out_{name}"), "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        H.assert_no_crash(cp.stderr, cp.returncode, "run_drug_target_mr.py")
        return cp.returncode, json.loads(cp.stdout)

    def test_list_for_non_list_step(self):
        code, d = self._run({"study": "t", "steps": {"qc_gwas": [{"input": "a"}]}}, "a")
        self.assertEqual(code, 2)
        self.assertIn("配置非法", d["error"])

    def test_steps_must_be_object(self):
        code, d = self._run({"study": "t", "steps": ["qc_gwas"]}, "b")
        self.assertEqual(code, 2)
        self.assertIn("必须是对象", d["error"])


# ============================================================ P3 协议
class TestFetchToolsProtocol(unittest.TestCase):
    """tools/fetch_tools.py：不同分支返回不同形状 + 空目录假成功。"""

    def test_unknown_tool_has_results_key(self):
        cp = subprocess.run(
            [PY, str(PKG / "tools" / "fetch_tools.py"), "--fetch", "no-such-tool",
             "--json"], capture_output=True, text=True, encoding="utf-8")
        d = json.loads(cp.stdout)
        self.assertEqual(cp.returncode, 1)
        self.assertIn("results", d, "所有分支都必须给 results 键")
        self.assertIn("known", d)

    def test_empty_dir_is_not_installed(self):
        """pip 失败会留下空目录；--check 曾把它报成已安装。"""
        root = AUDIT / "p3_root"
        root.mkdir(parents=True, exist_ok=True)
        (root / "numpy").mkdir(exist_ok=True)      # 空目录
        import os
        env = dict(os.environ, BIOSTAT_TOOLS_ROOT=str(root))
        cp = subprocess.run(
            [PY, str(PKG / "tools" / "fetch_tools.py"), "--check", "--json"],
            capture_output=True, text=True, encoding="utf-8", env=env)
        d = json.loads(cp.stdout)
        np_row = [t for t in d["tools"] if t["name"] == "numpy"][0]
        self.assertFalse(np_row["installed"], "空目录不等于已安装")


class TestVendorCapabilities(unittest.TestCase):
    """vendor/__init__.py：capabilities 恒只报 stats。"""

    def test_reports_each_module(self):
        """capabilities() 必须逐模块反映事实，不能恒只报 stats。

        判据直接查 vendor 包自身的导出（这才是对外契约），
        而不是猜模块在 core/engines/viz 哪个子包下。
        """
        import vendor
        mods = vendor.capabilities()["tier1_modules"]
        self.assertEqual(len(mods), 8)
        if not vendor.capabilities()["tier1_sci_stack_ready"]:
            self.skipTest("科学栈未齐备")
        n_reported = sum(1 for v in mods.values() if v)
        n_real = sum(1 for name in mods if getattr(vendor, name, None) is not None)
        self.assertEqual(n_reported, n_real,
                         "报告的可用模块数必须等于 vendor 实际导出的模块数")

    @unittest.skipUnless(H.has("numpy"), "需要科学栈")
    def test_all_modules_available_when_stack_ready(self):
        """科学栈齐备时，8 个引擎必须**全部**报为可用。

        这条是旧 bug 的直接判据：当时无论栈是否齐备都只报 stats。
        """
        import vendor
        c = vendor.capabilities()
        if not c["tier1_sci_stack_ready"]:
            self.skipTest("科学栈未齐备")
        mods = c["tier1_modules"]
        self.assertTrue(all(mods.values()),
                        f"栈齐备却报不可用：{[k for k, v in mods.items() if not v]}")


# ============================================================ 元门禁
class TestMetaGates(unittest.TestCase):
    """证明上面这套断言本身有效 —— 否则它可能形同虚设。"""

    def test_crash_detector_catches_real_crash(self):
        """反向验证：让一个真会崩的脚本跑一遍，断言必须失败。"""
        with self.assertRaises(AssertionError):
            H.assert_no_crash(
                "Traceback (most recent call last):\n"
                '  File "x.py", line 1\nIndexError: list index out of range\n',
                1, "probe.py")

    def test_crash_detector_passes_clean_failure(self):
        """反向验证：干净的失败（argparse 用法错误）不该被误判为崩溃。"""
        H.assert_no_crash("usage: x.py\nerror: unrecognized arguments: --nope\n",
                           2, "probe.py")

    def test_exit_one_alone_is_ambiguous(self):
        """记录这条元事实：exit=1 既可能是失败也可能是崩溃。

        这解释了为什么必须靠 stderr 里的 traceback 标记来判别，
        而不能只看退出码。
        """
        crash = subprocess.run([PY, "-c", "raise SystemExit(1)"],
                              capture_output=True, text=True)
        real_crash = subprocess.run([PY, "-c", "1/0"],
                                    capture_output=True, text=True)
        self.assertEqual(crash.returncode, real_crash.returncode)
        self.assertNotEqual(crash.stderr, real_crash.stderr,
                            "退出码相同但 stderr 不同 → 必须看 stderr")


if __name__ == "__main__":
    unittest.main(verbosity=2)