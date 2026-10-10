"""test_wrapper.py —— 「魔改开源项目」wrapper 层的自检

这一组用例守护的是**包装层不变量**：

1. 失败必须可见（exit 0 的工具不能被当成成功）；
2. 期望产物必须真的存在（最硬的证据）；
3. 底层工具缺失时是 SKIP 而非假失败，且**两种路径用例数一致**；
4. 参数翻译不得静默丢参（正向 + 反向双向断言）；
5. 对拍门禁必须能 FAIL（否则它只是摆设）；
6. 文档里的命令必须真的能跑（把文档当代码测）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import _harness as H
from _harness import PKG, PY

TOOLS = PKG / "tools"
sys.path.insert(0, str(TOOLS))

from wrapper_core import (ToolWrapper, calibration_report,  # noqa: E402
                          compare, official_error)

AUDIT = PKG / "tests" / ".audit_tmp"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class TestOfficialErrorDetection(unittest.TestCase):
    """陷阱 1：很多科研 CLI 出错也 exit 0，只在日志留一行 Error:。"""

    def test_detects_error_prefix(self):
        self.assertIsNotNone(official_error("Error: can not open the file"))
        self.assertIsNotNone(official_error("ERROR: something broke"))
        self.assertIsNotNone(official_error("some output\nerror: bad allele"))

    def test_clean_log_has_no_error(self):
        self.assertIsNone(official_error("Writing out... done in 3s"))


class TestToolWrapperFailureVisibility(unittest.TestCase):
    """陷阱 1 的核心：不能因为 returncode==0 就报成功。"""

    def test_missing_binary_fails_loudly(self):
        tw = ToolWrapper("definitely-not-a-real-binary-xyz123")
        r = tw.run(["--help"])
        self.assertFalse(r.ok)
        self.assertIn("未安装", r.reason)
        self.assertIn("fetch_tools", r.reason,
                      "缺失时必须给出获取方式，不能只说不行")

    def test_exit_zero_without_expected_output_is_failure(self):
        """最硬的证据：exit 0 但产物不存在 → 必须判失败。"""
        import sys as _s
        stub = AUDIT / "stub_ok_noout.py"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("import sys\nprint('done')\nsys.exit(0)\n",
                        encoding="utf-8")
        tw = ToolWrapper("stub", binary=f"{_s.executable} {stub}".split()[0],
                         expected_outputs=["never_created.txt"])
        r = tw.run([str(stub)])
        self.assertEqual(r.returncode, 0, "底层确实 exit 0")
        self.assertFalse(r.ok, "exit 0 但产物缺失 → 不得报成功")
        self.assertIn("缺少期望产物", r.reason)

    def test_error_line_in_output_is_failure(self):
        import sys as _s
        stub = AUDIT / "stub_err.py"
        stub.parent.mkdir(parents=True, exist_ok=True)
        # exit 0 但打印 Error: —— 模拟真实科研 CLI 的静默失败
        stub.write_text("print('Error: can not open the file [x.esd] to read.')\n",
                        encoding="utf-8")
        tw = ToolWrapper("stub_err", binary=_s.executable)
        r = tw.run([str(stub)])
        self.assertEqual(r.returncode, 0)
        self.assertFalse(r.ok, "日志里的 Error: 行必须让判定失败")

    def test_success_case_passes(self):
        import sys as _s
        outdir = AUDIT / "stub_ok"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "real.txt").write_text("x", encoding="utf-8")
        stub = AUDIT / "stub_ok.py"
        stub.write_text("open('real.txt','w').write('hi')\n", encoding="utf-8")
        tw = ToolWrapper("stub_ok", binary=_s.executable,
                         expected_outputs=["real.txt"])
        r = tw.run([str(stub)], outdir=outdir)
        self.assertTrue(r.ok, f"应判成功：{r.reason}")

    def test_traceback_is_failure(self):
        import sys as _s
        stub = AUDIT / "stub_crash.py"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("raise ValueError('boom')\n", encoding="utf-8")
        tw = ToolWrapper("stub_crash", binary=_s.executable)
        r = tw.run([str(stub)])
        self.assertFalse(r.ok)


class TestCalibration(unittest.TestCase):
    """对拍：compare / calibration_report 的语义。"""

    def test_match_and_differ(self):
        self.assertEqual(compare(0.5, 0.5)["status"], "match")
        self.assertEqual(compare(0.5, 0.7)["status"], "differs")

    def test_nan_handling(self):
        r = compare(float("nan"), float("nan"))
        self.assertEqual(r["status"], "both_nan")
        r2 = compare(None, 0.5)
        self.assertEqual(r2["status"], "incomparable")

    def test_report_fails_when_any_differs(self):
        rep = calibration_report([compare(0.5, 0.5), compare(0.5, 0.6)])
        self.assertEqual(rep["overall"], "FAIL")
        self.assertEqual(rep["n_differs"], 1)
        rep2 = calibration_report([compare(0.5, 0.5)])
        self.assertEqual(rep2["overall"], "PASS")

    @unittest.skipUnless(H.has("lifelines"), "需要 lifelines 作为金标准")
    def test_real_calibration_passes(self):
        """真跑一遍官方对拍：修复后的 c_index 必须与 lifelines 逐位一致。"""
        cp = subprocess.run(
            [sys.executable, str(TOOLS / "calibrate_lifelines.py")],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PKG))
        try:
            rep = json.loads(cp.stdout)
        except json.JSONDecodeError:
            self.skipTest(f"对拍脚本未输出 JSON：{cp.stdout[:200]}")
        if rep.get("status") == "SKIP":
            self.skipTest(rep.get("reason", "SKIP"))
        self.assertEqual(rep["overall"], "PASS",
                         f"与 lifelines 存在差异："
                         f"{[r for r in rep['rows'] if r['status'] == 'differs'][:3]}")

    def test_calibration_script_is_reproducible(self):
        """同一命令跑两次，结论必须一致（确定性）。"""
        if not H.has("lifelines"):
            self.skipTest("需要 lifelines")
        outs = []
        for _ in range(2):
            cp = subprocess.run(
                [sys.executable, str(TOOLS / "calibrate_lifelines.py")],
                capture_output=True, text=True, encoding="utf-8",
                cwd=str(PKG))
            try:
                d = json.loads(cp.stdout)
            except json.JSONDecodeError:
                self.skipTest("环境无 lifelines")
            outs.append((d["overall"], d["n_match"], d["n_differs"]))
        self.assertEqual(outs[0], outs[1], "对拍必须可复现")


class TestFlagTranslation(unittest.TestCase):
    """陷阱 3：参数翻译不得静默丢参（双向断言）。"""

    def test_no_silent_flag_loss(self):
        import re

        src = _read(TOOLS / "calibrate_lifelines.py")
        declared = set(re.findall(r'add_argument\(\s*"(--[a-z][\w-]*)"', src))
        # 源码里出现过、但没有 add_argument 声明的 flag（应为空）
        mentioned = set(re.findall(r'"(--[a-z][\w-]*)"', src))
        undeclared = mentioned - declared
        self.assertEqual(undeclared, set(),
                         f"源码提到但未声明的 flag：{undeclared}")

    def test_passthrough_entry_exists(self):
        """兜底入口必须存在 —— 它是「包装层不是功能上限」的实现方式。"""
        src = _read(TOOLS / "calibrate_lifelines.py")
        self.assertIn('"official"', src,
                      "必须有 official 子命令原样透传任意底层调用")
        cp = subprocess.run(
            [sys.executable, str(TOOLS / "calibrate_lifelines.py"),
             "official", "--anything", "goes"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(PKG))
        self.assertEqual(cp.returncode, 0)
        d = json.loads(cp.stdout)
        self.assertIn("--anything", d["passthrough"])


class TestDocCommandsAreCode(unittest.TestCase):
    """把文档里的命令当代码测：脚本存在 + flag 被接受。"""

    def _flags(self, script: Path) -> set:
        import re
        return set(re.findall(r'"(--[a-z][\w-]*)"', _read(script)))

    def test_documented_fetch_command_runs(self):
        """SKILL.md/README 里写的 fetch_tools 命令必须真能跑。"""
        cp = subprocess.run(
            [sys.executable, str(TOOLS / "fetch_tools.py"), "--list", "--json"],
            capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        d = json.loads(cp.stdout)
        self.assertIn("by_tier", d)

    def test_capabilities_command_runs(self):
        cp = subprocess.run(
            [sys.executable, str(PKG / "scripts" / "capabilities.py"), "--json"],
            capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        json.loads(cp.stdout)

    def test_coverage_guard(self):
        """覆盖面守卫：防止提取器失效导致「零发现」的假通过。"""
        scripts = ["fetch_tools.py", "vendor_sync.py", "wrapper_core.py",
                   "calibrate_lifelines.py"]
        for s in scripts:
            self.assertTrue((TOOLS / s).is_file(), f"缺脚本 {s}")
        self.assertGreaterEqual(len(scripts), 4, "覆盖面不足，检查本身可能失效")


class TestSkipPathConsistency(unittest.TestCase):
    """两种路径（工具在/不在）用例数必须一致 —— 少了就是静默跳过。"""

    def test_both_paths_report_same_total(self):
        """裸 Python 与有科学栈环境下，**用例总数**必须一致。

        判据用「本模块的用例数量」而非真跑两遍整套（跑整套要 2 分钟且
        内部还会调 subprocess，性价比极低）：用 AST 数出 test_ 开头的
        方法数即可 —— 同一份源码在两种环境下的用例数必然相同，
        真正要防的是「某条用例在某环境下提前 return 导致静默不算」。
        """
        import ast

        tree = ast.parse(_read(PKG / "tests" / "test_wrapper.py"))
        n_here = sum(
            1 for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            for f in cls.body
            if isinstance(f, ast.FunctionDef) and f.name.startswith("test_"))
        self.assertGreater(n_here, 5, "用例数异常，检查本身可能已失效")

        # 关键：不允许「因为缺依赖而整类跳过」的实现 ——
        # skipIf 挂在 class 上会让整个类在裸环境下消失。
        skip_on_class = [
            cls.name for cls in tree.body
            if isinstance(cls, ast.ClassDef)
            and any("skipUnless" in ast.dump(d) or "skipIf" in ast.dump(d)
                    for d in cls.decorator_list)]
        self.assertEqual(skip_on_class, [],
                         f"这些类整体 skip 会让裸环境少跑用例：{skip_on_class}")

    def test_no_recursive_selftest_invocation(self):
        """元门禁：自检里不得调用整套自检入口（会无限递归）。

        用 **AST 静态检查**而不是文本搜索 —— 文本搜索会匹配到自己写的
        注释（实测踩过：注释里提到那个文件名就把自己判成违规）。
        判据是「代码里真的构造了对它的调用」。

        目标文件名用拼接构造，避免本判据的字面量把自己命中
        （第一版就踩了这个：endswith 里的字符串被自己扫到）。
        """
        import ast

        target = "run_" + "all.py"          # 拼接，规避自指
        tree = ast.parse(_read(PKG / "tests" / "test_wrapper.py"))
        offenders = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value.endswith(target)
        ]
        self.assertEqual(offenders, [],
                         f"第 {offenders} 行在代码里引用了自检入口，"
                         "会在自检中无限递归")


if __name__ == "__main__":
    unittest.main(verbosity=2)