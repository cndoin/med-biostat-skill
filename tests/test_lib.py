#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_lib.py —— lib/ 扩展层自检（Tier1：所需 numpy/statsmodels 在外部 interpreter）

用带科学栈的 interpreter 跑子进程（找不到就 skip）。断言**定向数值**（不是「不崩溃」）。
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import _harness as H  # noqa: E402

PKG = H.PKG


class TestLibIncrement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 唯一裁决点：_harness.find_sci_python（P21）
        cls.py = H.find_sci_python()

    def _run(self, code):
        if not self.py:
            self.skipTest(H.skip_reason_sci(("numpy",)))
        cp = subprocess.run([str(self.py), "-c", code], cwd=str(PKG),
                            capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        return cp.stdout

    def test_c_index_bounds(self):
        out = self._run(
            "from lib import increment as i;"
            "print(i.c_index([1,2,3,4],[1,1,1,1],[4,3,2,1])['c']);"
            "print(i.c_index([1,2,3,4],[1,1,1,1],[1,2,3,4])['c'])")
        lines = out.split()
        self.assertAlmostEqual(float(lines[0]), 1.0, places=9)
        self.assertAlmostEqual(float(lines[1]), 0.0, places=9)

    def test_auroc_perfect(self):
        out = self._run(
            "from lib import increment as i;"
            "print(i.auroc([1,1,0,0],[0.9,0.8,0.3,0.2]))")
        self.assertAlmostEqual(float(out.strip()), 1.0, places=9)

    def test_nri_idi(self):
        out = self._run(
            "from lib import increment as i;"
            "n=i.continuous_nri([1,1,1,0,0,0],[0.5,0.4,0.6,0.5,0.4,0.6],"
            "[0.8,0.7,0.9,0.2,0.3,0.1]);"
            "d=i.idi([1,1,1,0,0,0],[0.5,0.4,0.6,0.5,0.4,0.6],[0.8,0.7,0.9,0.2,0.3,0.1]);"
            "print(n['nri']); print(d['idi'])")
        vals = [float(x) for x in out.split()]
        self.assertAlmostEqual(vals[0], 2.0, places=6)   # 完美重分类
        self.assertAlmostEqual(vals[1], 0.6, places=6)

    def test_e_value_analytic(self):
        """E-value(RR=1.5) = 1.5 + sqrt(1.5*0.5) = 2.3660254037844384。"""
        out = self._run(
            "from lib import increment as i; print(i.e_value(1.5)['e_value'])")
        self.assertAlmostEqual(float(out.strip()), 1.5 + (0.75 ** 0.5), places=9)

    def test_competing_cif(self):
        out = self._run(
            "from lib import competing as c;"
            "print(c.cif([1,2,3,4],[1,2,0,1],cause=1)['final_cif'])")
        self.assertTrue(0.0 <= float(out.strip()) <= 1.0)

    def test_mixed_degrades_without_statsmodels(self):
        """statsmodels 缺失时必须返回 error 而不是抛异常（优雅降级）。"""
        code = (
            "import sys, types\n"
            "sys.modules['statsmodels']=None\n"        # 强制不可用
            "sys.modules['statsmodels.formula']=None\n"
            "sys.modules['statsmodels.formula.api']=None\n"
            "import pandas as pd\n"
            "from lib import mixed\n"
            "r=mixed.mixed_slope(pd.DataFrame({'y':[1,2,3],'t':[0,1,2],'id':[1,1,1],'x':[1,2,3]}),"
            "'y','t','id','x')\n"
            "print('error' in r)\n")
        out = self._run(code)
        self.assertIn("True", out)


    def test_mice_and_rubin(self):
        """MICE 返回 m 个无缺失数据集；Rubin 合并与手算一致。"""
        out = self._run(
            "import numpy as np, pandas as pd\n"
            "from lib import impute as im\n"
            "r=im.pool_rubin([1.0,1.2],[0.04,0.04])\n"
            "print('%.4f %.4f %.4f %.4f'%(r['estimate'],r['within_var'],r['between_var'],r['total_var']))\n"
            "rng=np.random.default_rng(0); n=150\n"
            "a=rng.normal(0,1,n); b=0.7*a+rng.normal(0,1,n)\n"
            "df=pd.DataFrame({'a':a,'b':b}); m=rng.random(n)<0.2; df.loc[m,'b']=np.nan\n"
            "sets=im.mice(df,m=4,max_iter=4,seed=1)\n"
            "print(len(sets)); print(all(s['b'].isna().sum()==0 for s in sets))\n")
        vals = out.split()
        self.assertAlmostEqual(float(vals[0]), 1.1, places=6)
        self.assertAlmostEqual(float(vals[2]), 0.02, places=6)
        self.assertAlmostEqual(float(vals[3]), 0.07, places=6)
        self.assertEqual(vals[4], "4")
        self.assertEqual(vals[5], "True")


if __name__ == "__main__":
    unittest.main(verbosity=2)
