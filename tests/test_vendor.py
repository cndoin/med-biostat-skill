#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_vendor.py —— 内嵌层（vendor/）自检

验证四件事：
  1. **优雅降级**：裸 Python（无 numpy/pandas/scipy）下 import vendor 不崩，零依赖子模块仍可用；
  2. **Tier1 可用**：有科学栈的 interpreter 下引擎确实能 import 并调用；
  3. **同步幂等**：vendor 与上游一致（重跑 sync 无漂移）；
  4. **去重生效**：评估后决定不内嵌的文件确实不在 vendor 里（防重复实现回流）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import _harness as H  # noqa: E402

PKG = H.PKG
TOOLS = PKG / "tools"

# Tier1 需要 numpy+pandas+scipy 三者齐全（唯一裁决点：_harness.find_sci_python）
_SCI_MODS = ("numpy", "pandas", "scipy")


class TestVendorDegradation(unittest.TestCase):
    """裸 Python 下必须能 import 且降级，不能一 import 就崩。"""

    def test_import_does_not_crash(self):
        cp = subprocess.run([sys.executable, "-c",
                             "import vendor; print('OK')"],
                            cwd=str(PKG), capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertIn("OK", cp.stdout)

    def test_zero_dep_modules_always_available(self):
        """registry / spec 是纯标准库，任何环境都必须可用。"""
        cp = subprocess.run(
            [sys.executable, "-c",
             "import vendor;"
             "print(vendor.registry.get_registry() is not None);"
             "print(callable(getattr(vendor.spec,'tripod_ai',None)))"],
            cwd=str(PKG), capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(cp.stdout.split(), ["True", "True"])

    def test_capabilities_reports_degragation(self):
        cp = subprocess.run([sys.executable, "-c",
                             "import json, vendor;"
                             "print(json.dumps(vendor.capabilities()))"],
                            cwd=str(PKG), capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        caps = json.loads(cp.stdout)
        self.assertIn("tier1_sci_stack_ready", caps)
        self.assertIn("zero_dep_modules", caps)


class TestCapabilitiesScript(unittest.TestCase):
    def test_json_output(self):
        code, d, _ = H.run("capabilities.py", "--json")
        self.assertEqual(code, 0)
        self.assertTrue(d["tier0"]["ready"])
        self.assertIn("tier1", d)


class TestVendorSync(unittest.TestCase):
    def test_sync_is_idempotent(self):
        """开发目录：重跑检查应无漂移；安装副本：校验 vendor 文件齐全。"""
        cp = subprocess.run([sys.executable, str(TOOLS / "vendor_sync.py"), "--json"],
                            capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(cp.returncode, 0, cp.stderr)
        rep = json.loads(cp.stdout)
        self.assertEqual(rep["errors"], [])
        if rep.get("mode") == "detached":
            # 安装副本（~/.claude/skills 等）没有上游源，只要求内嵌文件齐全
            missing = [f for f in rep["files"] if f["status"] == "missing"]
            self.assertEqual(missing, [], f"副本缺少内嵌文件：{missing}")
        else:
            n_dirty = sum(1 for f in rep["files"] if f["status"] == "dirty")
            self.assertEqual(n_dirty, 0, f"有 {n_dirty} 个文件与上游不一致，需重跑 --write")

    def test_rewrites_actually_applied(self):
        """适配规则必须真的生效（spec.py 的 import 应指向 _deps）。"""
        spec = (PKG / "vendor" / "spec.py").read_text(encoding="utf-8")
        self.assertIn("from ._deps import available", spec)
        self.assertNotIn("from ._optional import", spec)

    def test_no_duplicate_implementations(self):
        """去重生效：NOT_VENDORED 里的文件不得出现在 vendor（防重复实现回流，P21）。"""
        sys.path.insert(0, str(TOOLS))
        import vendor_sync
        present = {p.name for p in (PKG / "vendor").rglob("*.py")}
        # 评估后不内嵌的文件，其同名文件不应存在（model.py→modeling.py 属改名，不在其中）
        forbidden = {"desc.py", "io.py"}
        self.assertFalse(forbidden & present,
                         f"发现本该去重的重复实现：{forbidden & present}")

    def test_header_is_content_determined_not_calendar_determined(self):
        """同步头部不得含随日历变化的字段。

        为什么专门锁这条：早先头部写 `# 同步日期: <today>`，而头部参与
        幂等比对 —— 于是**每天第一次跑同步必然全部 dirty**，
        `test_sync_is_idempotent` 日抛一次假失败（2026-10-11 实测一次冒出
        13 个 dirty，逐行 diff 确认 13/13 的差异只有日期那一行，正文零差异）。
        假失败比没门禁更糟：它会让人习惯性忽略门禁输出。
        """
        sys.path.insert(0, str(TOOLS))
        import vendor_sync
        h = vendor_sync.build_header("a/b.py", "src", "MIT", "note", "abc123")
        self.assertNotIn("同步日期", h, "头部不得写同步日期（会随日历漂移）")
        self.assertIn("源 sha256", h)
        # 同一输入必须得到完全相同的输出（幂等的必要条件）
        self.assertEqual(
            h, vendor_sync.build_header("a/b.py", "src", "MIT", "note", "abc123"))
        # 内容变 → 头部变（头部确实由内容决定，不是被写死）
        self.assertNotEqual(
            h, vendor_sync.build_header("a/b.py", "src", "MIT", "note", "def456"))
        # 根因锁死：源码里不得再引入 date.today()
        src = (TOOLS / "vendor_sync.py").read_text(encoding="utf-8")
        self.assertNotIn("date.today()", src,
                         "vendor_sync 不得再使用 date.today()（日历漂移源）")


class TestVendorTier1(unittest.TestCase):
    """有科学栈时的引擎可用性（找不到带 numpy 的 interpreter 则跳过）。"""

    @classmethod
    def setUpClass(cls):
        cls.py = H.find_sci_python(_SCI_MODS)

    def test_engines_available(self):
        if not self.py:
            self.skipTest(H.skip_reason_sci(_SCI_MODS))
        code = (
            "import vendor\n"
            "assert vendor.capabilities()['tier1_sci_stack_ready'] is True\n"
            "assert callable(vendor.stats.adjust_p)\n"
            "assert callable(vendor.mr_coloc.mr)\n"
            "assert callable(vendor.clinical.table_one)\n"
            "assert callable(vendor.survival.logrank)\n"
            "assert callable(vendor.differential.differential)\n"
            "assert callable(vendor.ptm.occupancy)\n"
            "assert callable(vendor.spec.strobe)\n"
            "print('TIER1-OK')\n"
        )
        cp = subprocess.run([str(self.py), "-c", code], cwd=str(PKG),
                            capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertIn("TIER1-OK", cp.stdout)

    def test_mr_end_to_end(self):
        """MR 引擎端到端：合成工具变量 → 三种方法 + 异质性都有输出。"""
        if not self.py:
            self.skipTest(H.skip_reason_sci(_SCI_MODS))
        code = (
            "import numpy as np, pandas as pd, vendor\n"
            "rng=np.random.default_rng(0); n=12\n"
            "bx=rng.uniform(0.05,0.3,n)\n"
            "by=0.5*bx+rng.normal(0,0.02,n)\n"
            "se=rng.uniform(0.01,0.03,n)\n"
            "d=pd.DataFrame({'beta_exposure':bx,'beta_outcome':by,'se_outcome':se})\n"
            "r=vendor.mr_coloc.mr(d)\n"
            "print('METHODS', sorted(r['method'].tolist()))\n"
            "ivw=r[r['method']=='IVW'].iloc[0]\n"
            "print('IVW_BETA_NEAR_HALF', abs(ivw['beta']-0.5)<0.2)\n"
        )
        cp = subprocess.run([str(self.py), "-c", code], cwd=str(PKG),
                            capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertIn("IVW_BETA_NEAR_HALF True", cp.stdout)
        for m in ("IVW", "MR-Egger", "加权中位数"):
            self.assertIn(m, cp.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
