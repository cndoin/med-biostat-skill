#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_opensource.py —— 从 GitHub 开源项目魔改而来的能力自检

覆盖三层：
  1. **vendored 开源核心**（vendor/_mr、_enrich、_ophtha）
     —— 必须能 import、能算、且与**独立参考值**一致
  2. **包装层**（lib/mr_full、lib/ophtha、lib/prep）
  3. **溯源**：ATTRIBUTION 必须如实记录来源与许可，不得把 GPL 写进去

稳定性测试用**不变性 / 边界 / 大样本**三类；功能性测试断言**定向数值**
（不是「不崩溃」）。另有负向用例证明门禁会失败。
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


#: MR 相关模块（vendor/_mr/*）需要 numpy + pandas + scipy 三件套；
#: 纯 numpy 的模块（ophtha 的光学、prep 的清洗）单独用 LIGHT。
#: **必须分级**：统一按 numpy 判定的话，缺 scipy 的环境会把「该跳过」变成
#: 「子进程 import 失败」—— 那是门禁噪音，掩盖了「本机其实能跑一半」的事实。
HEAVY = ("numpy", "pandas", "scipy")
LIGHT = ("numpy",)


def _run(code: str, mods: tuple = HEAVY):
    """用带科学栈的解释器跑一段代码，返回 stdout。"""
    py = H.find_sci_python(mods)
    if py is None:
        raise unittest.SkipTest(H.skip_reason_sci(mods))
    cp = subprocess.run([str(py), "-c", code], cwd=str(PKG),
                        capture_output=True, text=True, timeout=300)
    if cp.returncode != 0:
        # 「缺依赖」与「算错了」必须区分：只有前者该跳过，后者是真失败。
        if "ModuleNotFoundError" in (cp.stderr or ""):
            raise unittest.SkipTest(
                f"子进程缺依赖（不是代码问题）："
                f"{[l for l in cp.stderr.splitlines() if 'ModuleNotFoundError' in l]}")
        raise AssertionError(f"子进程失败(exit={cp.returncode}):\n{cp.stderr[:1200]}")
    return cp.stdout


def _runl(code: str):
    """只依赖 numpy 的轻量入口（ophtha 光学 / prep 清洗）。"""
    return _run(code, LIGHT)


class TestVendoredOpenSource(unittest.TestCase):
    """vendored 的开源模块必须真的可用（不是文件放进去了就算数）。"""

    def test_modules_are_importable(self):
        out = _run(
            "import numpy as np\n"
            "from vendor._mr import methods, sensitivity, harmonize, multivariable\n"
            "from vendor._enrich import stats as enrich_stats\n"
            "from vendor._ophtha import pearldgs\n"
            "print('IMPORT-OK')\n")
        self.assertIn("IMPORT-OK", out)

    def test_vendored_files_carry_header(self):
        """每个 vendored 文件都必须带溯源头部（否则无法追责来源）。"""
        for rel in ("vendor/_mr/methods.py", "vendor/_mr/sensitivity.py",
                    "vendor/_mr/harmonize.py", "vendor/_mr/multivariable.py",
                    "vendor/_enrich/stats.py", "vendor/_ophtha/pearldgs.py"):
            p = PKG / rel
            self.assertTrue(p.exists(), f"缺少 {rel}")
            head = p.read_text(encoding="utf-8")[:200]
            self.assertIn("VENDORED-FILE", head, f"{rel} 缺少溯源头部")

    def test_no_absolute_import_leak(self):
        """vendored 后不得残留原包的绝对导入（会 ModuleNotFoundError）。"""
        out = _run(
            "from pathlib import Path\n"
            "bad=[]\n"
            "for d in ('vendor/_mr','vendor/_enrich','vendor/_ophtha'):\n"
            "    for f in Path(d).glob('*.py'):\n"
            "        t=f.read_text(encoding='utf-8')\n"
            "        if 'from pymr' in t or 'from gseapy' in t:\n"
            "            bad.append(str(f))\n"
            "print('LEAK:', bad)\n")
        self.assertIn("LEAK: []", out)


class TestMendelianRandomization(unittest.TestCase):
    """MR 全套：定向数值 + 稳定性。"""

    def _sim(self, n=40, true_beta=0.5, seed=0):
        return (
            "import numpy as np\n"
            "from lib import mr_full\n"
            f"rng=np.random.default_rng({seed}); n={n}\n"
            f"bx=rng.normal(0.3,0.05,n); sx=np.full(n,0.02)\n"
            f"by=bx*{true_beta}+rng.normal(0,0.01,n); sy=np.full(n,0.03)\n"
        )

    def test_ivw_recovers_true_effect(self):
        """定向：真值 0.5，IVW 必须落在 ±0.05。"""
        out = _run(self._sim() +
                   "r=mr_full.run_all(bx,sx,by,sy)\n"
                   "print('%.6f'%r['estimates']['ivw']['beta'])\n")
        beta = float(out.strip().splitlines()[-1])
        self.assertAlmostEqual(beta, 0.5, delta=0.05,
                               msg=f"IVW 未恢复真值 0.5，得到 {beta}")

    def test_all_estimators_agree_within_tolerance(self):
        """稳定性：7 个估计量应彼此接近（真值 0.5），极差 < 0.15。"""
        out = _run(self._sim(n=60) +
                   "r=mr_full.run_all(bx,sx,by,sy)\n"
                   "for k,v in r['estimates'].items(): print('%.6f'%v['beta'])\n")
        vals = [float(x) for x in out.strip().splitlines()]
        self.assertGreaterEqual(len(vals), 6, "可计算的估计量太少")
        self.assertLess(max(vals) - min(vals), 0.15,
                        f"估计量分歧过大：{vals}")

    def test_null_effect_gives_zero(self):
        """定向：暴露与结局无因果时，IVW 应≈0（不是「不崩」就行）。"""
        out = _run(
            "import numpy as np\n"
            "from lib import mr_full\n"
            "rng=np.random.default_rng(1); n=50\n"
            "bx=rng.normal(0.3,0.05,n); sx=np.full(n,0.02)\n"
            "by=rng.normal(0,0.01,n); sy=np.full(n,0.03)\n"
            "r=mr_full.run_all(bx,sx,by,sy)\n"
            "print('%.6f'%r['estimates']['ivw']['beta'])\n")
        beta = float(out.strip().splitlines()[-1])
        self.assertAlmostEqual(beta, 0.0, delta=0.05,
                               msg=f"零效应下 IVW 应≈0，得到 {beta}")

    def test_sensitivity_runs_and_q_is_sane(self):
        """Cochran Q 的 df 必须 = n-1（df 错了整个检验就错）。"""
        out = _run(self._sim(n=40) +
                   "d=mr_full.diagnose(bx,sx,by,sy)\n"
                   "q=d['checks']['cochrans_q']\n"
                   "print(int(q['Q_df']),'%.4f'%float(q['Q']))\n")
        df, q = out.strip().split()
        self.assertEqual(int(df), 39, f"Cochran Q 自由度应为 39，得到 {df}")
        self.assertGreaterEqual(float(q), 0.0)

    def test_too_few_instruments_is_reported_not_silently_skipped(self):
        """负向：工具变量不足时，必须**出现在 skipped 里并带原因**，不能默默少给结果。"""
        out = _run(
            "import numpy as np\n"
            "from lib import mr_full\n"
            "bx=np.array([0.3,0.28]); sx=np.array([0.02,0.02])\n"
            "by=np.array([0.15,0.14]); sy=np.array([0.03,0.03])\n"
            "r=mr_full.run_all(bx,sx,by,sy)\n"
            "print('SKIPPED_KEYS', sorted(r['skipped']))\n")
        self.assertIn("weighted_median", out, "2 个 SNP 时加权中位数必须被记录为跳过")

    def test_unknown_method_raises(self):
        """负向：拼错的方法名必须报错，不能静默返回 IVW。"""
        code = ("import numpy as np\nfrom lib import mr_full\n"
                "bx=np.array([0.3]*10); sx=np.array([0.02]*10)\n"
                "by=np.array([0.15]*10); sy=np.array([0.03]*10)\n"
                "try:\n"
                "    mr_full.estimate('ivvv', bx,sx,by,sy)\n"
                "    print('NO-RAISE')\n"
                "except KeyError:\n"
                "    print('RAISED')\n")
        self.assertIn("RAISED", _run(code))


class TestOphthalmologyCataract(unittest.TestCase):
    """只需 numpy（pearldgs 是纯 numpy 实现）。"""

    def _runl(self, code):
        return _run(code, LIGHT)

    """眼科 / 白内障：光学往返自洽 + 标准换算定向值。"""

    def test_optics_roundtrip_is_exact(self):
        """核心不变性：SE → TILP → SE 必须回到原值（浮点极限内）。"""
        out = self._runl(
            "from lib import ophtha as O\n"
            "worst=0.0\n"
            "for se in (-3,-2,-1,-0.5,0,0.5,1,2):\n"
            "    t=O.tilp_from_refraction(se, al_mm=23.5, rco1_mm=7.8, rco2_mm=6.5)\n"
            "    b=O.refraction_from_tilp(t, al_mm=23.5, rco1_mm=7.8, rco2_mm=6.5)\n"
            "    worst=max(worst, abs(b-se))\n"
            "print('%.3e'%worst)\n")
        worst = float(out.strip().splitlines()[-1])
        self.assertLess(worst, 1e-9, f"光学往返不闭合，最大误差 {worst:.3e}")

    def test_logmar_standard_values(self):
        """定向：6/6=0、6/12=0.3010、6/60=1.0（国际标准定义）。"""
        out = self._runl(
            "from lib import ophtha as O\n"
            "print('%.4f'%O.logmar_from_snellen(6,6))\n"
            "print('%.4f'%O.logmar_from_snellen(6,12))\n"
            "print('%.4f'%O.logmar_from_snellen(6,60))\n")
        v = [float(x) for x in out.strip().splitlines()]
        self.assertAlmostEqual(v[0], 0.0, places=4)
        self.assertAlmostEqual(v[1], 0.3010, places=3)
        self.assertAlmostEqual(v[2], 1.0, places=3)

    def test_logmar_decimal_inverse(self):
        out = self._runl(
            "from lib import ophtha as O\n"
            "for d in (0.1,0.25,0.5,0.8,1.0):\n"
            "    print('%.3e'%abs(O.decimal_from_logmar(O.logmar_from_decimal(d))-d))\n")
        for line in out.strip().splitlines():
            self.assertLess(float(line), 1e-12)

    def test_refractive_outcome_zero_error_case(self):
        """定向：预测完全准确时 ME=MAE=0 且 ±0.25D 内为 100%。"""
        out = self._runl(
            "import numpy as np\nfrom lib import ophtha as O\n"
            "p=np.array([-0.5,0.0,0.5,1.0])\n"
            "r=O.refractive_outcome(p,p)\n"
            "print('%.6f %.6f %.6f'%(r['ME'],r['MAE'],r['within_0.25D']))\n")
        me, mae, w = [float(x) for x in out.strip().split()]
        self.assertAlmostEqual(me, 0.0, places=9)
        self.assertAlmostEqual(mae, 0.0, places=9)
        self.assertAlmostEqual(w, 1.0, places=9)

    def test_refractive_outcome_sign_convention(self):
        """定向：实际比预测更近视（更负）→ PE 为负。符号约定错了结论就反了。"""
        out = self._runl(
            "import numpy as np\nfrom lib import ophtha as O\n"
            "p=np.array([0.0,0.0]); a=np.array([-1.0,-2.0])\n"
            "r=O.refractive_outcome(p,a)\n"
            "print('%.6f'%r['ME'])\n")
        self.assertAlmostEqual(float(out.strip()), -1.5, places=6)

    def test_cataract_gene_sets_are_nonempty_and_disjoint_enough(self):
        """基因集必须非空；晶状体蛋白家族之间不能有全集重叠（否则是配错了）。"""
        out = self._runl(
            "from lib import ophtha as O\n"
            "names=sorted(O.CATARACT_GENE_SETS)\n"
            "print(len(names), min(len(v) for v in O.CATARACT_GENE_SETS.values()))\n"
            "print(O.lens_gene_overlap(['CRYAA','CRYAB','EGFR'],'crystallin_alpha')['n_overlap'])\n")
        lines = out.strip().splitlines()
        n_sets, min_size = [int(x) for x in lines[0].split()]
        self.assertGreaterEqual(n_sets, 5)
        self.assertGreaterEqual(min_size, 2)
        self.assertEqual(int(lines[1]), 2, "CRYAA/CRYAB 应命中 crystallin_alpha 的 2 个")

    def test_gene_overlap_requires_enrichment_test(self):
        """负向安全网：交集计数**不等于**富集；未知基因集必须报错。"""
        code = ("from lib import ophtha as O\n"
                "try:\n"
                "    O.lens_gene_overlap(['CRYAA'],'no_such_set')\n"
                "    print('NO-RAISE')\n"
                "except KeyError:\n"
                "    print('RAISED')\n")
        self.assertIn("RAISED", self._runl(code))


class TestDataPrep(unittest.TestCase):
    """只需 numpy。"""

    def _runl(self, code):
        return _run(code, LIGHT)

    """数据清洗：结构识别 + 不变性。"""

    def _build(self):
        return (
            "import numpy as np\n"
            "from lib import prep\n"
            "rng=np.random.default_rng(3); n=120\n"
            "X=rng.normal(0,1,(n,5))\n"
            "X[:,3]=X[:,0]*0.98+rng.normal(0,0.05,n)\n"   # 与 col0 共线
            "X[:,4]=3.0\n"                                 # 常数列
            "X[5,1]=14.0\n"                                # 离群
            "X[7,2]=np.nan\n"                              # 缺失
        )

    def test_detects_collinearity_before_pruning(self):
        """关键：VIF 必须在剪枝**前**算，否则共线性会被自己掩盖。

        实测教训：最初实现先剪枝再算 VIF，r=0.998 的两列给出 max_vif=1.019，
        报告看起来完全正常，而真实共线性严重。VIF 是暴露问题的，不能先掩盖。
        """
        out = self._runl(self._build() +
                   "rep=prep.prep_pipeline(X, names=['a','b','c','d','e'])\n"
                   "print('%.2f'%rep['vif']['max_vif'])\n")
        self.assertGreater(float(out.strip()), 10.0,
                           "VIF 未暴露共线性（可能又变成剪枝后再算了）")

    def test_detects_constant_and_outlier_and_missing(self):
        out = self._runl(self._build() +
                   "rep=prep.prep_pipeline(X)\n"
                   "print(rep['low_variance_dropped'])\n"
                   "print(sum(o['n_outliers'] for o in rep['outliers']))\n"
                   "print(rep['missing']['cols_with_any_missing'])\n")
        lines = out.strip().splitlines()
        self.assertIn("4", lines[0], "常数列 4 未被识别")
        self.assertGreater(int(lines[1]), 0, "离群点未被识别")
        self.assertIn("2", lines[2], "含缺失的列 2 未被识别")

    def test_prep_does_not_mutate_input(self):
        """不变性：prep_pipeline 只报告不改数据（自动改数据最难追责）。"""
        out = self._runl(self._build() +
                   "before=X.copy()\n"
                   "rep=prep.prep_pipeline(X)\n"
                   "same_nan = (np.isnan(before)==np.isnan(X)).all()\n"
                   "same_val = np.array_equal(np.nan_to_num(before,nan=-999),"
                   "                          np.nan_to_num(X,nan=-999))\n"
                   "print(bool(same_nan and same_val))\n")
        self.assertIn("True", out)

    def test_outlier_methods_boundary(self):
        """边界：无离群的均匀数据，三种口径都不该乱报。"""
        out = self._runl(
            "import numpy as np\nfrom lib import prep\n"
            "x=np.linspace(0,1,50)\n"
            "print(sum(prep.outlier_report(x,m)['n_outliers'] for m in ('iqr','zscore','mad')))\n")
        self.assertEqual(int(out.strip()), 0)

    def test_unknown_outlier_method_raises(self):
        code = ("import numpy as np\nfrom lib import prep\n"
                "try:\n"
                "    prep.outlier_report(np.arange(10.0), method='nope')\n"
                "    print('NO-RAISE')\n"
                "except ValueError:\n"
                "    print('RAISED')\n")
        self.assertIn("RAISED", self._runl(code))

    def test_large_input_runs(self):
        """稳定性：2 万行 × 20 列不得退化（O(n²) 会在这里暴露）。"""
        out = self._runl(
            "import numpy as np, time\nfrom lib import prep\n"
            "rng=np.random.default_rng(0)\n"
            "X=rng.normal(0,1,(20000,20))\n"
            "t=time.perf_counter(); rep=prep.prep_pipeline(X); dt=time.perf_counter()-t\n"
            "print('%.2f'%dt)\n")
        dt = float(out.strip())
        self.assertLess(dt, 30.0, f"2万×20 清洗耗时 {dt:.1f}s，疑似复杂度退化")


class TestAttributionHonesty(unittest.TestCase):
    """溯源必须如实——把 GPL 项目写进 ATTRIBUTION 是许可事故。"""

    def test_attribution_lists_new_projects(self):
        text = (PKG / "vendor" / "ATTRIBUTION.md").read_text(encoding="utf-8")
        for token in ("pymr", "GSEApy", "pearldgs", "Unlicense", "BSD-3-Clause", "MIT"):
            self.assertIn(token, text, f"ATTRIBUTION 未记录 {token}")

    def test_no_gpl_project_is_vendored(self):
        """负向：py-twosamplemr 是 GPL-3.0，绝不能出现在 vendor 源码里。"""
        hits = []
        for p in (PKG / "vendor").rglob("*.py"):
            t = p.read_text(encoding="utf-8", errors="replace")
            if "py-twosamplemr" in t or "pytwosamplemr" in t:
                hits.append(str(p.relative_to(PKG)))
        self.assertEqual(hits, [], f"GPL 项目被内嵌：{hits}")

    def test_vendor_sync_declares_third_party_sources(self):
        text = (PKG / "tools" / "vendor_sync.py").read_text(encoding="utf-8")
        self.assertIn("THIRD_PARTY_SOURCES", text)
        # 准入判据必须写在代码里，否则后来者会随手加 GPL 项目
        self.assertIn("GPL", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)