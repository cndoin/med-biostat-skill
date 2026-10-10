#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_methods.py —— 主流分析法（v1.7 新增 13 个模块）的功能性与稳定性测试。

测试分三层，缺一层就抓不到对应类型的 bug（这个分级是踩出来的）：
  1. **定向数值**（断言等于已知真值）—— 抓「方向错 / 口径错」
     例：Cox-LASSO 系数符号反了、padj 解包错位、SuSiE 残差减错
  2. **不变性与稳定性**（换尺度 / 换顺序 / 大样本 / 边界）—— 抓「数值发散」
  3. **负向用例**（构造已知会错的输入，断言**报错**而不是给出假结果）
     —— 抓「静默失败」。这类 bug 最危险：它把「算不出来」伪装成「算出来是 0」

「能跑通」不算通过。每个分析法至少要有一条**定向数值**断言。
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

HEAVY = ("numpy", "pandas", "scipy")
LIGHT = ("numpy",)


def _run(code: str, mods: tuple = HEAVY):
    """用带科学栈的解释器跑一段代码；缺依赖 -> 跳过，算错 -> 失败。"""
    py = H.find_sci_python(mods)
    if py is None:
        raise unittest.SkipTest(H.skip_reason_sci(mods))
    cp = subprocess.run([str(py), "-c", code], cwd=str(PKG),
                        capture_output=True, text=True, timeout=600)
    if cp.returncode != 0:
        if "ModuleNotFoundError" in (cp.stderr or ""):
            raise unittest.SkipTest(
                f"子进程缺依赖（不是代码问题）："
                f"{[l for l in cp.stderr.splitlines() if 'ModuleNotFoundError' in l]}")
        raise AssertionError(f"子进程失败(exit={cp.returncode}):\n{cp.stderr[:1500]}")
    return cp.stdout


class TestAllMethodsImportable(unittest.TestCase):
    """13 个新分析法模块必须真的能 import（不是文件放进去了就算数）。"""

    MODULES = [
        "network", "singlecell", "pathway", "immune", "regularized", "meta",
        "causal", "prs", "finemap", "mutational", "spatial", "trajectory", "power",
    ]

    def test_import_all(self):
        code = "from lib import " + ", ".join(self.MODULES) + "\nprint('IMPORT-OK')\n"
        self.assertIn("IMPORT-OK", _run(code))

    def test_capabilities_reports_each_module(self):
        """能力探测必须逐模块报告，不能用 tier1_ready 一刀切。

        一刀切会把「某个模块缺依赖」误报成「整层不可用」，
        进而让人以为功能缺失而去重复造轮子。
        """
        code = (
            "import sys, json\n"
            "sys.path.insert(0, 'scripts')\n"
            "import capabilities\n"
            "lm = capabilities._probe_lib_modules()\n"
            "print(json.dumps(lm))\n")
        out = _run(code)
        import json
        d = json.loads(out.strip().splitlines()[-1])
        self.assertIn("lib.network", d)
        self.assertTrue(d["lib.network"], "WGCNA 模块应可用")
        self.assertGreaterEqual(sum(1 for v in d.values() if v), 13,
                                f"可用模块数不足：{d}")


class TestWGCNA(unittest.TestCase):
    """分析法③：WGCNA 共表达网络。"""

    def test_recovers_two_planted_modules(self):
        """种下两个互不相关的共表达块，必须被分为两个模块。"""
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import network as N\n"
            "rng=np.random.default_rng(7); n=80\n"
            "b1=rng.normal(size=n); b2=rng.normal(size=n)\n"
            "E=np.column_stack([b1+rng.normal(0,.25,n) for _ in range(20)]"
            "+[b2+rng.normal(0,.25,n) for _ in range(20)]"
            "+[rng.normal(size=n) for _ in range(10)])\n"
            "df=pd.DataFrame(E,columns=[f'g{i}' for i in range(E.shape[1])])\n"
            "r=N.wgcna(df,traits=pd.DataFrame({'t1':b1,'t2':b2}),power=6,min_module_size=8)\n"
            "print('MODULES',r['n_modules'])\n"
            "mt=r['module_trait']['correlation']\n"
            "print('MAXCORR_T1',round(float(mt['t1'].abs().max()),3))\n"
            "print('MAXCORR_T2',round(float(mt['t2'].abs().max()),3))\n")
        out = _run(code)
        self.assertIn("MODULES 2", out, f"应识别出 2 个模块：{out}")
        for k in ("MAXCORR_T1", "MAXCORR_T2"):
            v = float([l for l in out.splitlines() if k in l][0].split()[-1])
            self.assertGreater(v, 0.9, f"{k} 应 >0.9（ME 与驱动因子强相关），实际 {v}")

    def test_soft_threshold_table_is_wellformed(self):
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import network as N\n"
            "rng=np.random.default_rng(1)\n"
            "E=rng.normal(size=(50,30))\n"
            "r=N.soft_threshold(E,powers=[4,6,8])\n"
            "print('ROWS',len(r['table']),'COLS',list(r['table'].columns))\n")
        out = _run(code)
        self.assertIn("ROWS 3", out)
        self.assertIn("SFT.R.sq", out)

    def test_negative_too_few_genes_raises(self):
        """基因数 <3 时相关网络退化，必须报错而不是给出无意义的模块。"""
        code = (
            "import numpy as np\n"
            "from lib import network as N\n"
            "try:\n"
            "    N.wgcna(np.random.normal(size=(10,2)),power=6)\n"
            "    print('NO-RAISE')\n"
            "except ValueError as e:\n"
            "    print('RAISED', str(e)[:40])\n")
        out = _run(code)
        self.assertIn("RAISED", out, "退化输入必须报错，不能静默出结果")


class TestSingleCell(unittest.TestCase):
    """分析法④：单细胞全流程（算法移植自 scanpy，BSD-3）。"""

    def test_pipeline_recovers_planted_clusters(self):
        """种下 3 个细胞群，Louvain 应把它们分开（ARI 高）。"""
        code = (
            "import numpy as np\n"
            "from lib import singlecell as SC\n"
            "rng=np.random.default_rng(2)\n"
            "cent=np.array([[5.,0.,0.],[0.,5.,0.],[0.,0.,5.]])\n"
            "X=np.vstack([rng.poisson(cent[i],(40,3)).astype(float) for i in range(3)])\n"
            "genes=['G1','G2','MT-x']\n"
            "r=SC.standard_pipeline(X,gene_names=genes,min_genes=1,min_cells=1,"
            "n_top_genes=3,n_comps=2,n_neighbors=5,resolution=1.0,seed=3)\n"
            "lab=r['labels']; kept=r['cell_index_kept']\n"
            "# QC 过滤会丢细胞，真值必须按**保留下来的索引**对齐，\n"
            "# 否则长度不一致（实测 crosstab 直接抛 ValueError）\n"
            "truth=np.repeat([0,1,2],40)[kept]\n"
            "from scipy.optimize import linear_sum_assignment\n"
            "import pandas as pd\n"
            "ct=pd.crosstab(truth,lab).values\n"
            "ri,ci=linear_sum_assignment(-ct)\n"
            "acc=ct[ri,ci].sum()/len(truth)\n"
            "print('ACC',round(float(acc),3))\n"
            "print('ALGO',r['algorithm'])\n")
        out = _run(code)
        acc = float([l for l in out.splitlines() if "ACC" in l][0].split()[-1])
        self.assertGreater(acc, 0.9, f"聚类准确率应 >0.9，实际 {acc}")
        self.assertIn("louvain", out, "必须标注是 Louvain 而非 Leiden")

    def test_qc_metrics_mito_prefix(self):
        code = (
            "import numpy as np\n"
            "from lib import singlecell as SC\n"
            "X=np.array([[10.,1.,0.],[0.,0.,5.]])\n"
            "q=SC.qc(X,gene_names=['MT-CO1','ACTB','GAPDH'],mito_prefix='MT-')\n"
            "print('MITO_PCT_0',round(float(q['pct_mito'][0]),2))\n"
            "print('NGENES_1',int(q['n_genes'][1]))\n")
        out = _run(code)
        # 第一行：MT 计数 10 / 总数 11 → 90.91%
        self.assertIn("MITO_PCT_0 90.91", out)
        self.assertIn("NGENES_1 1", out)

    def test_markers_ranks_true_marker_first(self):
        code = (
            "import numpy as np\n"
            "from lib import singlecell as SC\n"
            "rng=np.random.default_rng(9)\n"
            "X=rng.normal(2,0.5,(60,20))\n"
            "X[:30,0]+=6.0   # 只有 g0 是第 0 簇的 marker\n"
            "lab=np.repeat([0,1],30)\n"
            "genes=[f'g{i}' for i in range(20)]\n"
            "m=SC.markers(X,lab,gene_names=genes,method='wilcoxon',n_top=5,\n"
            "             min_logfc=None,max_padj=None)\n"
            "top=m['significant']['0'].iloc[0]['gene']\n"
            "print('TOP',top)\n")
        out = _run(code)
        self.assertIn("TOP g0", out, "真正上调的基因必须排第一")

    def test_negative_signature_without_overlap_raises(self):
        """签名基因一个都没匹配上时必须报错（不能静默给全 0）。"""
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import immune as I\n"
            "df=pd.DataFrame(np.zeros((3,3)),columns=['a','b','c'])\n"
            "try:\n"
            "    I.estimate_score(df)\n"
            "    print('NO-RAISE')\n"
            "except ValueError:\n"
            "    print('RAISED')\n")
        _run(code)  # 不崩即可
        out = _run(code)
        self.assertIn("RAISED", out)


class TestPathway(unittest.TestCase):
    """分析法⑤：ORA / GSEA / GSVA / ssGSEA。"""

    def test_ora_recovers_planted_set(self):
        code = (
            "import numpy as np\n"
            "from lib import pathway as P\n"
            "genes=[f'g{i}' for i in range(300)]\n"
            "sets={'HIT':[f'g{i}' for i in range(40)],'NOISE':[f'g{i}' for i in range(200,220)]}\n"
            "o=P.ora([f'g{i}' for i in range(30)],sets,background=genes)\n"
            "o=o.sort_values('pvalue')\n"
            "print('TOP',o.iloc[0]['term'])\n"
            "print('P_HIT',o[o.term=='HIT']['pvalue'].iloc[0])\n"
            "print('PADJ_LE_P',bool((o['padj']>=o['pvalue']-1e-12).all()))\n")
        out = _run(code)
        self.assertIn("TOP HIT", out)
        self.assertIn("PADJ_LE_P True", out, "padj 必须 >= p（BH 性质）")

    def test_padj_is_not_boolean(self):
        """反向用例：padj 曾被错取成 reject 布尔值（全 0/1）。

        上游 multiple_testing_correction 返回 (q, rej)，取 [1] 会把 padj
        变成 0/1——p=0.23 的项 padj 变成 0.0，是彻底错误的校正值。
        """
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import pathway as P\n"
            "rng=np.random.default_rng(1)\n"
            "X=pd.DataFrame(np.abs(rng.normal(size=(20,50))),columns=[f'g{i}' for i in range(50)])\n"
            "sets={'S':[f'g{i}' for i in range(10)]}\n"
            "gv=P.gsva(X,sets)\n"
            "d=P.pathway_activity_test(gv,np.repeat(['A','B'],10))\n"
            "pv=d['pvalue'].tolist(); qv=d['padj'].tolist()\n"
            "print('ANY_PADJ_IN_01',bool(all((q==0.0 or q==1.0) for q in qv)))\n"
            "print('PADJ_GE_P',bool(all(q>=p-1e-12 for p,q in zip(pv,qv))))\n")
        out = _run(code)
        self.assertIn("ANY_PADJ_IN_01 False", out, "padj 不应全是 0/1")
        self.assertIn("PADJ_GE_P True", out)

    def test_gsea_detects_top_enriched_set(self):
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import pathway as P\n"
            "rng=np.random.default_rng(0)\n"
            "genes=[f'g{i}' for i in range(200)]\n"
            "r=pd.Series(rng.normal(size=200),index=genes)\n"
            "r[:40]+=2.5\n"
            "sets={'TOP':[f'g{i}' for i in range(40)],'RAND':[f'g{i}' for i in range(100,120)]}\n"
            "g=P.gsea(r,sets,n_perm=300,min_size=10,seed=5)\n"
            "print('TOP',g.iloc[0]['term'])\n"
            "print('ES',round(float(g.iloc[0]['ES']),3))\n")
        out = _run(code)
        self.assertIn("TOP TOP", out)
        es = float([l for l in out.splitlines() if l.startswith("ES")][0].split()[-1])
        self.assertGreater(es, 0.4, f"富集在顶部的基因集 ES 应明显为正，实际 {es}")

    def test_ssgsea_and_gsva_shapes(self):
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import pathway as P\n"
            "rng=np.random.default_rng(3)\n"
            "X=pd.DataFrame(np.abs(rng.normal(size=(12,60))),columns=[f'g{i}' for i in range(60)])\n"
            "sets={'A':[f'g{i}' for i in range(15)],'B':[f'g{i}' for i in range(30,40)]}\n"
            "print('SSGSEA',P.ssgsea(X,sets).shape)\n"
            "print('GSVA',P.gsva(X,sets,kcdf='gaussian').shape)\n"
            "print('GSVA_POIS',P.gsva(X,sets,kcdf='poisson').shape)\n")
        out = _run(code)
        self.assertIn("SSGSEA (12, 2)", out)
        self.assertIn("GSVA (12, 2)", out)


class TestRegularized(unittest.TestCase):
    """分析法⑦：Ridge / LASSO / ElasticNet / Cox-LASSO。"""

    def test_lasso_selects_true_variable_only(self):
        code = (
            "import numpy as np\n"
            "from lib import regularized as R\n"
            "rng=np.random.default_rng(5); n=150\n"
            "X=rng.normal(size=(n,12)); y=X[:,0]*3+rng.normal(0,.5,n)\n"
            "cv=R.cv_elastic_net(X,y,alpha=1.0,n_lambda=40,seed=0)\n"
            "sel=R.select_variables(cv['coef_at_1se'])\n"
            "print('SELECTED',sel)\n"
            "print('COEF0',round(float(cv['coef_at_1se'][0]),2))\n")
        out = _run(code)
        self.assertIn("SELECTED [0]", out, "LASSO 只应保留真正的 x0")
        c0 = float([l for l in out.splitlines() if "COEF0" in l][0].split()[-1])
        self.assertGreater(c0, 2.0, f"x0 系数应接近 3（LASSO 有偏但方向对），实际 {c0}")

    def test_ridge_keeps_all_variables(self):
        code = (
            "import numpy as np\n"
            "from lib import regularized as R\n"
            "rng=np.random.default_rng(5)\n"
            "X=rng.normal(size=(120,12)); y=X[:,0]*3+rng.normal(0,.5,120)\n"
            "cv=R.cv_elastic_net(X,y,alpha=0.0,n_lambda=30,seed=0)\n"
            "print('NONZERO',len(R.select_variables(cv['coef_at_1se'])))\n")
        out = _run(code)
        self.assertIn("NONZERO 12", out, "Ridge 不筛变量，12 个都应保留")

    def test_cox_lasso_sign_is_correct(self):
        """反向用例：风险集方向写反时，系数符号会整体反转。

        这是实际发生过的 bug——把 `cumsum(v[::-1])[::-1]` 当成风险集累加，
        等价于取了「生存时间更短的那一批人」。
        """
        code = (
            "import numpy as np\n"
            "from lib import regularized as R\n"
            "rng=np.random.default_rng(5); n=400; p=10\n"
            "X=rng.normal(size=(n,p)); bt=np.zeros(p); bt[0]=1.2; bt[3]=-0.8\n"
            "t=rng.exponential(np.exp(-(X@bt))); ev=(rng.random(n)<0.8).astype(int)\n"
            "r=R.cox_lasso(X,t,ev,lambdas=[1e-6],max_iter=300)\n"
            "c=r['coef_path'][0]\n"
            "print('C0',round(float(c[0]),3))\n"
            "print('C3',round(float(c[3]),3))\n")
        out = _run(code)
        c0 = float([l for l in out.splitlines() if l.startswith("C0")][0].split()[-1])
        c3 = float([l for l in out.splitlines() if l.startswith("C3")][0].split()[-1])
        self.assertGreater(c0, 0.5, f"真值 +1.2 的系数必须为正，实际 {c0}")
        self.assertLess(c3, -0.3, f"真值 -0.8 的系数必须为负，实际 {c3}")

    def test_lasso_path_shrinks(self):
        """λ 路径必须单调：λ 越小，非零系数越多。"""
        code = (
            "import numpy as np\n"
            "from lib import regularized as R\n"
            "rng=np.random.default_rng(8)\n"
            "X=rng.normal(size=(100,15)); y=X[:,0]*2+rng.normal(0,1,100)\n"
            "r=R.elastic_net(X,y,alpha=1.0,n_lambda=25)\n"
            "nn=r['n_nonzero']\n"
            "print('MONOTONE',bool(np.all(np.diff(nn)>=0)))\n"
            "print('FIRST',int(nn[0]),'LAST',int(nn[-1]))\n")
        out = _run(code)
        self.assertIn("MONOTONE True", out)
        self.assertIn("FIRST 0", out, "最大 λ 处必须全为 0")


class TestMeta(unittest.TestCase):
    """分析法⑧：Meta 分析。"""

    def test_fixed_effect_matches_hand_computation(self):
        """定向对拍：inverse-variance 合并值必须与手算一致。"""
        code = (
            "import numpy as np\n"
            "from lib import meta as M\n"
            "y=np.array([0.5,0.7,0.3]); s=np.array([0.2,0.25,0.15])\n"
            "w=1/s**2\n"
            "manual=float((w*y).sum()/w.sum())\n"
            "se_manual=float(np.sqrt(1/w.sum()))\n"
            "r=M.meta_analysis(y,s,model='fixed')\n"
            "print('DIFF',abs(r['estimate']-manual))\n"
            "print('SE_DIFF',abs(r['se']-se_manual))\n"
            "print('I2',round(r['heterogeneity']['I2'],4))\n")
        out = _run(code)
        d = float([l for l in out.splitlines() if l.startswith("DIFF")][0].split()[-1])
        sd = float([l for l in out.splitlines() if "SE_DIFF" in l][0].split()[-1])
        self.assertLess(d, 1e-12, f"固定效应合并值应与手算一致，差 {d}")
        self.assertLess(sd, 1e-12)

    def test_heterogeneous_data_triggers_random_model(self):
        code = (
            "import numpy as np\n"
            "from lib import meta as M\n"
            "y=np.array([0.1,0.9,0.15,0.85]); s=np.array([0.05,0.05,0.05,0.05])\n"
            "r=M.meta_analysis(y,s,model='auto')\n"
            "print('MODEL',r['model'])\n"
            "print('I2',round(r['heterogeneity']['I2'],3))\n")
        out = _run(code)
        self.assertIn("MODEL random", out, "明显异质时 auto 应选随机效应")
        i2 = float([l for l in out.splitlines() if l.startswith("I2")][0].split()[-1])
        self.assertGreater(i2, 0.5)

    def test_negative_zero_se_raises(self):
        code = (
            "import numpy as np\n"
            "from lib import meta as M\n"
            "try:\n"
            "    M.meta_analysis(np.array([1.,2.]),np.array([0.0,1.0]))\n"
            "    print('NO-RAISE')\n"
            "except ValueError:\n"
            "    print('RAISED')\n")
        out = _run(code)
        self.assertIn("RAISED", out, "se=0 会让权重发散，必须报错")

    def test_leave_one_out_shape(self):
        code = (
            "import numpy as np\n"
            "from lib import meta as M\n"
            "y=np.array([0.3,0.45,0.2,0.55]); s=np.array([0.12,0.15,0.1,0.2])\n"
            "print('ROWS',len(M.leave_one_out(y,s)))\n")
        out = _run(code)
        self.assertIn("ROWS 4", out)


class TestCausal(unittest.TestCase):
    """分析法⑨：观察性因果推断。"""

    def test_methods_remove_confounding(self):
        """定向：朴素差 3.05（混杂），各方法应回到真值 2.0 附近。"""
        code = (
            "import numpy as np\n"
            "from lib import causal as C\n"
            "rng=np.random.default_rng(11); n=900\n"
            "X=rng.normal(size=(n,3))\n"
            "ps=1/(1+np.exp(-(X@np.array([0.8,-0.5,0.3]))))\n"
            "T=(rng.random(n)<ps).astype(int)\n"
            "Y=2.0*T+X@np.array([1.0,-0.7,0.5])+rng.normal(0,.5,n)\n"
            "naive=Y[T==1].mean()-Y[T==0].mean()\n"
            "print('NAIVE',round(float(naive),3))\n"
            "print('PSM',round(C.psm(X,T,Y,caliper=0.2,seed=1)['att'],3))\n"
            "print('AIPW',round(C.aipw(X,T,Y,n_folds=5,seed=1)['ate'],3))\n"
            "print('GCOMP',round(C.g_computation(X,T,Y)['ate'],3))\n")
        out = _run(code)
        naive = float([l for l in out.splitlines() if l.startswith("NAIVE")][0].split()[-1])
        self.assertGreater(abs(naive - 2.0), 0.8, "朴素比较必须明显有偏，否则测不出校正效果")
        for k in ("PSM", "AIPW", "GCOMP"):
            v = float([l for l in out.splitlines() if l.startswith(k)][0].split()[-1])
            self.assertLess(abs(v - 2.0), 0.25, f"{k} 应回到真值 2.0 附近，实际 {v}")

    def test_iptw_improves_balance(self):
        code = (
            "import numpy as np\n"
            "from lib import causal as C\n"
            "rng=np.random.default_rng(11); n=900\n"
            "X=rng.normal(size=(n,3))\n"
            "ps=1/(1+np.exp(-(X@np.array([0.8,-0.5,0.3]))))\n"
            "T=(rng.random(n)<ps).astype(int)\n"
            "Y=2.0*T+X@np.array([1.0,-0.7,0.5])+rng.normal(0,.5,n)\n"
            "r=C.iptw(X,T,Y)\n"
            "b=C.balance_report(X,T,weights=r['weights'])\n"
            "print('MAX_SMD_BEFORE',round(float(b['smd_unweighted'].abs().max()),3))\n"
            "print('MAX_SMD_AFTER',round(float(b['smd_weighted'].abs().max()),3))\n")
        out = _run(code)
        before = float([l for l in out.splitlines() if "BEFORE" in l][0].split()[-1])
        after = float([l for l in out.splitlines() if "AFTER" in l][0].split()[-1])
        self.assertGreater(before, 0.3, "加权前必须明显不平衡")
        self.assertLess(after, 0.1, f"加权后 SMD 必须 <0.1，实际 {after}")

    def test_negative_no_overlap_returns_no_matches(self):
        """完全不重叠（positivity 不成立）时，匹配数应为 0 且明确说明。"""
        code = (
            "import numpy as np\n"
            "from lib import causal as C\n"
            "X=np.array([[10.0],[20.0]])\n"
            "T=np.array([1,0]); Y=np.array([1.0,0.0])\n"
            "r=C.psm(X,T,Y,caliper=1e-6,seed=0)\n"
            "print('NPAIRS',r['n_matched_pairs'])\n")
        out = _run(code)
        self.assertIn("NPAIRS 0", out)


class TestPRS(unittest.TestCase):
    """分析法⑩：多基因风险评分。"""

    def test_score_is_linear_combination(self):
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import prs as P\n"
            "G=pd.DataFrame(np.array([[0.,1.,2.],[1.,1.,0.]]),columns=['a','b','c'])\n"
            "w=pd.DataFrame({'snp':['a','b','c'],'beta':[1.,2.,3.],'p':[1e-9,1e-8,1e-7]})\n"
            "s=P.score(G,w,center=False)\n"
            "manual=np.array([0*1+1*2+2*3, 1*1+1*2+0*3],dtype=float)\n"
            "print('DIFF',float(np.abs(s.to_numpy()-manual).max()))\n")
        out = _run(code)
        d = float([l for l in out.splitlines() if l.startswith("DIFF")][0].split()[-1])
        self.assertLess(d, 1e-12, f"PRS 必须是 β·G 的线性组合，差 {d}")

    def test_clump_keeps_lead_snp(self):
        code = (
            "import numpy as np\n"
            "from lib import prs as P\n"
            "r=P.clump(['a','b','c'],[1,1,1],[1000,2000,900000],\n"
            "          [1e-10,1e-9,1e-12],[0.5,0.4,0.6],window_kb=250)\n"
            "print('KEPT',sorted(r['kept']))\n")
        out = _run(code)
        self.assertIn("KEPT ['a', 'c']", out, "250kb 窗口内只应保留 P 最小的")

    def test_auc_of_strong_score(self):
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import prs as P\n"
            "rng=np.random.default_rng(3); n=500; m=30\n"
            "G=rng.integers(0,3,(n,m)).astype(float)\n"
            "bt=np.zeros(m); bt[:8]=rng.normal(size=8)\n"
            "lin=G@bt; y=(rng.random(n)<1/(1+np.exp(-lin))).astype(int)\n"
            "w=pd.DataFrame({'snp':[f'rs{i}' for i in range(m)],'beta':bt,'p':[1e-9]*m})\n"
            "s=P.score(pd.DataFrame(G,columns=w['snp']),w)\n"
            "e=P.evaluate(s.to_numpy(),y,kind='binary')\n"
            "print('AUC',round(e['auc'],3))\n")
        out = _run(code)
        auc = float([l for l in out.splitlines() if l.startswith("AUC")][0].split()[-1])
        self.assertGreater(auc, 0.6, f"真实信号构造下 AUC 应 >0.6，实际 {auc}")


class TestFineMapping(unittest.TestCase):
    """分析法⑪：精细定位 ABF / SuSiE。"""

    def test_abf_finds_true_causal(self):
        code = (
            "import numpy as np\n"
            "from lib import finemap as F\n"
            "rng=np.random.default_rng(4); p=30\n"
            "base=rng.normal(size=2000)\n"
            "X=np.column_stack([base*0.9+rng.normal(0,0.44,2000) for _ in range(p)])\n"
            "X=(X-X.mean(0))/X.std(0)\n"
            "y=X[:,5]*0.25+rng.normal(0,1,2000)\n"
            "b=np.array([np.corrcoef(X[:,j],y)[0,1] for j in range(p)])\n"
            "se=np.full(p,1/np.sqrt(2000))\n"
            "r=F.abf(b,se,prior_sd=0.2)\n"
            "print('TOP',int(np.argmax(r['posterior'])))\n")
        out = _run(code)
        self.assertIn("TOP 5", out)

    def test_susie_separates_two_causal(self):
        """反向用例：残差减错（把 α 当效应大小）时，第二个因果位点找不到。"""
        code = (
            "import numpy as np\n"
            "from lib import finemap as F\n"
            "rng=np.random.default_rng(4); p=30\n"
            "base=rng.normal(size=2000)\n"
            "X=np.column_stack([base*0.9+rng.normal(0,0.44,2000) for _ in range(p)])\n"
            "X=(X-X.mean(0))/X.std(0); R=np.corrcoef(X,rowvar=False)\n"
            "se=np.full(p,1/np.sqrt(2000))\n"
            "y=X[:,5]*0.25+X[:,20]*0.22+rng.normal(0,1,2000)\n"
            "b=np.array([np.corrcoef(X[:,j],y)[0,1] for j in range(p)])\n"
            "r=F.susie(b,se,R,L=4,prior_variance=0.2,max_iter=120)\n"
            "tops=[c['index'][0] for c in r['reliable_credible_sets']]\n"
            "print('CS_TOPS',sorted(tops))\n"
            "print('N_RELIABLE',r['n_credible_sets_reliable'])\n")
        out = _run(code)
        self.assertIn("CS_TOPS [5, 20]", out, "两个因果位点都应各自成为可信集的 top")
        self.assertIn("N_RELIABLE 2", out)

    def test_negative_ld_shape_mismatch_raises(self):
        code = (
            "import numpy as np\n"
            "from lib import finemap as F\n"
            "try:\n"
            "    F.susie(np.zeros(5),np.ones(5),np.eye(3),L=2)\n"
            "    print('NO-RAISE')\n"
            "except ValueError:\n"
            "    print('RAISED')\n")
        out = _run(code)
        self.assertIn("RAISED", out, "LD 维数不符必须报错")


class TestMutationalSpatialTrajectoryPower(unittest.TestCase):
    """分析法⑫⑬⑭⑮ 的核心断言。"""

    def test_sbs96_channels_and_profile(self):
        code = (
            "import numpy as np, pandas as pd\n"
            "from lib import mutational as MU\n"
            "ch=MU.sbs96_channels()\n"
            "print('NCHAN',len(ch))\n"
            "d=pd.DataFrame({'ref':['C']*10,'alt':['T']*10,\n"
            "                'ctx5':['A']*10,'ctx3':['T']*10})\n"
            "p=MU.sbs96_profile(d)\n"
            "print('SUM',round(float(p.to_numpy().sum()),6))\n"
            "print('ATCHANT',round(float(p.loc['all','A[C>T]T']),3))\n")
        out = _run(code)
        self.assertIn("NCHAN 96", out)
        self.assertIn("SUM 1.0", out, "归一化后谱和为 1")
        self.assertIn("ATCHANT 1.0", out, "所有突变应落在 A[C>T]T 通道")

    def test_nmf_reconstructs_planted_signatures(self):
        code = (
            "import numpy as np\n"
            "from lib import mutational as MU\n"
            "rng=np.random.default_rng(0)\n"
            "W=np.abs(rng.random((20,3))); H=np.abs(rng.random((3,96)))\n"
            "V=W@H\n"
            "r=MU.nmf_decompose(V,k=3,n_init=8,max_iter=400,seed=1)\n"
            "print('RELERR',round(r['relative_error'],4))\n")
        out = _run(code)
        rel = float([l for l in out.splitlines() if l.startswith("RELERR")][0].split()[-1])
        self.assertLess(rel, 0.05, f"NMF 应能重建种下的结构，相对误差 {rel}")

    def test_morans_i_detects_spatial_pattern(self):
        code = (
            "import numpy as np\n"
            "from lib import spatial as SP\n"
            "rng=np.random.default_rng(1)\n"
            "xy=np.column_stack([np.repeat(np.arange(10),10),np.tile(np.arange(10),10)]).astype(float)\n"
            "W=SP.build_knn_graph(xy,n_neighbors=4)\n"
            "pattern=xy[:,0]                      # 强空间梯度\n"
            "random=rng.normal(size=100)\n"
            "print('I_PATTERN',round(SP.morans_i(pattern,W,n_perm=99)['I'],3))\n"
            "print('I_RANDOM',round(SP.morans_i(random,W,n_perm=99)['I'],3))\n")
        out = _run(code)
        ip = float([l for l in out.splitlines() if "I_PATTERN" in l][0].split()[-1])
        ir = float([l for l in out.splitlines() if "I_RANDOM" in l][0].split()[-1])
        self.assertGreater(ip, 0.5, f"空间梯度基因的 Moran's I 应明显为正，实际 {ip}")
        self.assertLess(abs(ir), 0.2, f"随机基因的 Moran's I 应接近 0，实际 {ir}")

    def test_pseudotime_follows_true_order(self):
        code = (
            "import numpy as np\n"
            "from lib import trajectory as TR\n"
            "rng=np.random.default_rng(0); n=60\n"
            "t=np.linspace(0,1,n)\n"
            "X=np.column_stack([t*5+rng.normal(0,.08,n),np.sin(t*3)+rng.normal(0,.08,n)])\n"
            "pt=TR.diffusion_pseudotime(X,root=0)['pseudotime']\n"
            "from scipy.stats import spearmanr\n"
            "rho=spearmanr(pt,np.arange(n)).statistic\n"
            "print('RHO',round(float(rho),3))\n")
        out = _run(code)
        rho = float([l for l in out.splitlines() if l.startswith("RHO")][0].split()[-1])
        self.assertGreater(rho, 0.9, f"伪时间应与真实序高度相关，实际 {rho}")

    def test_power_two_means_matches_known_value(self):
        """定向对拍：n=64/组、d=0.5、α=0.05 的经典功效 ≈ 0.80。"""
        code = (
            "from lib import power as PW\n"
            "p=PW.power_two_means(0.5,1.0,64)['power']\n"
            "print('POWER',round(p,3))\n"
            "n=PW.sample_size_two_means(0.5,1.0,0.8)['n_per_group']\n"
            "print('N',n)\n")
        out = _run(code)
        pw = float([l for l in out.splitlines() if l.startswith("POWER")][0].split()[-1])
        nn = int([l for l in out.splitlines() if l.startswith("N ")][0].split()[-1])
        self.assertAlmostEqual(pw, 0.80, delta=0.02, msg=f"功效应≈0.80，实际 {pw}")
        self.assertEqual(nn, 64, f"每组所需样本量应为 64，实际 {nn}")

    def test_sample_size_survival_uses_events(self):
        code = (
            "from lib import power as PW\n"
            "r=PW.sample_size_survival(1.5,power=0.8)\n"
            "print('EVENTS',r['n_events_required'])\n"
            "print('CALIBER',r['note'][:6])\n")
        out = _run(code)
        ev = int([l for l in out.splitlines() if l.startswith("EVENTS")][0].split()[-1])
        self.assertGreater(ev, 150)
        self.assertLess(ev, 250)

    def test_diagnostic_accuracy_completeness(self):
        code = (
            "import numpy as np\n"
            "from lib import power as PW\n"
            "rng=np.random.default_rng(2)\n"
            "y=rng.integers(0,2,400); s=rng.normal(y*1.5,1,400)\n"
            "d=PW.diagnostic_accuracy(y,s)\n"
            "keys=['sensitivity','specificity','ppv','npv','lr_plus','lr_minus','dor','youden_j']\n"
            "print('ALL_FINITE',bool(all(np.isfinite(d[k]) for k in keys)))\n"
            "print('SENS',round(d['sensitivity'],3))\n")
        out = _run(code)
        self.assertIn("ALL_FINITE True", out)
        sens = float([l for l in out.splitlines() if l.startswith("SENS")][0].split()[-1])
        self.assertGreater(sens, 0.5, "有真实信号时敏感性应 >0.5")


class TestBatchCorrection(unittest.TestCase):
    """ComBat 批次校正（源 scanpy _combat.py, BSD-3）。"""

    def test_removes_batch_effect_while_keeping_biology(self):
        """定向：批次分离比必须大幅下降，而病例/对照的差异必须保留。"""
        out = _run(
            "import sys; sys.path.insert(0,'.')\n"
            "import numpy as np, pandas as pd\n"
            "from lib import batch as B\n"
            "rng=np.random.default_rng(0); n1,n2=40,35\n"
            "bt=np.array(['A']*n1+['B']*n2)\n"
            "grp=np.array(['case']*20+['ctrl']*20+['case']*17+['ctrl']*18)\n"
            "X=rng.normal(0,1,(n1+n2,40)); X[grp=='case',:8]+=1.0\n"
            "X[bt=='A']+=2.0; X[bt=='B']*=1.6\n"
            "df=pd.DataFrame(X,columns=[f'g{j}' for j in range(40)])\n"
            "d0=B.diagnose(df,bt); r=B.correct(df,bt); adj=r['adjusted']\n"
            "d1=B.diagnose(adj,bt)\n"
            "sb=float((df[grp=='case'].iloc[:,:8].mean()-df[grp=='ctrl'].iloc[:,:8].mean()).mean())\n"
            "sa=float((adj[grp=='case'].iloc[:,:8].mean()-adj[grp=='ctrl'].iloc[:,:8].mean()).mean())\n"
            # d0 的 ratio = 「对原始数据校正后」的批次分离 / 校正前 → 这是要断言的量
            "print('RATIO',round(d0['separation_ratio'],4))\n"
            "print('SEP_BEFORE',round(d0['batch_separation_before'],3))\n"
            # 校正后残留的批次分离（d1 的 before 即 adj 自身的分离度）
            "print('SEP_AFTER',round(d1['batch_separation_before'],4))\n"
            "print('BIO_B',round(sb,3)); print('BIO_A',round(sa,3))\n")
        ratio = float([l for l in out.splitlines() if l.startswith("RATIO")][0].split()[-1])
        sep0 = float([l for l in out.splitlines() if l.startswith("SEP_BEFORE")][0].split()[-1])
        sep1 = float([l for l in out.splitlines() if l.startswith("SEP_AFTER")][0].split()[-1])
        bio_b = float([l for l in out.splitlines() if l.startswith("BIO_B")][0].split()[-1])
        bio_a = float([l for l in out.splitlines() if l.startswith("BIO_A")][0].split()[-1])
        self.assertGreater(sep0, 1.0, "构造的批次混杂应当很强（前置条件）")
        self.assertLess(ratio, 0.2, f"校正后批次分离比应大幅下降，实测 {ratio}")
        self.assertLess(sep1, 0.5, f"校正后残留批次分离应接近 0，实测 {sep1}")
        self.assertAlmostEqual(bio_a, bio_b, delta=0.25,
                               msg=f"生物信号被误扣：校正前 {bio_b} → 校正后 {bio_a}")

    def test_reference_batch_is_untouched(self):
        """ref-batch 模式下参考批次必须逐位不变。"""
        out = _run(
            "import sys; sys.path.insert(0,'.')\n"
            "import numpy as np, pandas as pd\n"
            "from lib import batch as B\n"
            "rng=np.random.default_rng(1); bt=np.array(['A']*30+['B']*30)\n"
            "df=pd.DataFrame(rng.normal(size=(60,20)),columns=[f'g{j}' for j in range(20)])\n"
            "df[bt=='B']+=3.0\n"
            "r=B.correct_reference(df,bt,reference='A'); adj=r['adjusted']\n"
            "print('UNCHANGED',bool((adj[bt=='A']==df[bt=='A']).all().all()))\n"
            "print('CHANGED_B',bool(not (adj[bt=='B']==df[bt=='B']).all().all()))\n")
        self.assertIn("UNCHANGED True", out)
        self.assertIn("CHANGED_B True", out)

    def test_negative_collinear_covariate_raises(self):
        """负向：把 batch 自己当协变量 → 设计矩阵奇异，必须报错而非给出假结果。"""
        out = _run(
            "import sys; sys.path.insert(0,'.')\n"
            "import numpy as np, pandas as pd\n"
            "from lib import batch as B\n"
            "rng=np.random.default_rng(2); bt=np.array(['A']*20+['B']*20)\n"
            "df=pd.DataFrame(rng.normal(size=(40,15)))\n"
            "try:\n"
            "    B.correct(df,bt,covariates=pd.DataFrame({'b':bt})); print('RAISED False')\n"
            "except Exception: print('RAISED True')\n")
        self.assertIn("RAISED True", out)

    def test_negative_small_batch_raises(self):
        """负向：某批次只有 1 个样本 → 无法估计批内方差，必须报错。"""
        out = _run(
            "import sys; sys.path.insert(0,'.')\n"
            "import numpy as np, pandas as pd\n"
            "from lib import batch as B\n"
            "rng=np.random.default_rng(3)\n"
            "df=pd.DataFrame(rng.normal(size=(30,12)))\n"
            "bt=np.array(['A']*29+['B'])\n"
            "try:\n"
            "    B.correct(df,bt); print('RAISED False')\n"
            "except Exception: print('RAISED True')\n")
        self.assertIn("RAISED True", out)


class TestCellComm(unittest.TestCase):
    """细胞通讯（源 CellPhoneDB, MIT）。"""

    def test_directionality_and_significance(self):
        """定向：配体在 T、受体在 M → T→M 显著、M→T 不显著（不可对调）。"""
        out = _run(
            "import sys; sys.path.insert(0,'.')\n"
            "import numpy as np, pandas as pd\n"
            "from lib import cellcomm as C\n"
            "rng=np.random.default_rng(3)\n"
            "genes=['CXCL12','CXCR4','CCL2','CCR2']+[f'X{i}' for i in range(60)]\n"
            "X=np.abs(rng.normal(1.0,0.4,(120,len(genes))))\n"
            "df=pd.DataFrame(X,columns=genes)\n"
            "lab=np.array(['T']*40+['M']*40+['B']*40)\n"
            "df.loc[lab=='T','CXCL12']+=6; df.loc[lab=='M','CXCR4']+=6\n"
            "r=C.score(df,lab,[('CXCL12','CXCR4')],n_perm=200,min_cells=10,seed=1)\n"
            "pv=r['pvalues']\n"
            "print('PTM',float(pv.loc['CXCL12|CXCR4','T|M']))\n"
            "print('PMT',float(pv.loc['CXCL12|CXCR4','M|T']))\n"
            "print('NPERM',r['n_perm'])\n")
        ptm = float([l for l in out.splitlines() if l.startswith("PTM")][0].split()[-1])
        pmt = float([l for l in out.splitlines() if l.startswith("PMT")][0].split()[-1])
        self.assertIn("NPERM 200", out)
        self.assertLess(ptm, 0.05, "真实互作方向应显著")
        self.assertGreater(pmt, 0.5, "反方向不应显著（配受体角色不可对调）")

    def test_pvalue_respects_permutation_floor(self):
        """p 值只能是 0 或 ≥1/n_perm（经验 p 的物理下限）。"""
        out = _run(
            "import sys; sys.path.insert(0,'.')\n"
            "import numpy as np, pandas as pd\n"
            "from lib import cellcomm as C\n"
            "rng=np.random.default_rng(4)\n"
            "genes=['CXCL12','CXCR4']+[f'X{i}' for i in range(50)]\n"
            "df=pd.DataFrame(np.abs(rng.normal(1,0.4,(80,len(genes)))),columns=genes)\n"
            "lab=np.array(['A']*40+['B']*40)\n"
            "r=C.score(df,lab,[('CXCL12','CXCR4')],n_perm=100,min_cells=10,seed=2)\n"
            "v=r['pvalues'].to_numpy().ravel()\n"
            "ok=bool(np.all((v==0)|(v>=1.0/100-1e-12)))\n"
            "print('FLOOR_OK',ok)\n")
        self.assertIn("FLOOR_OK True", out)

    def test_missing_genes_are_reported_not_silent(self):
        """负向：配受体基因不在矩阵里 → 必须计数报告，不能静默返回空。"""
        out = _run(
            "import sys; sys.path.insert(0,'.')\n"
            "import numpy as np, pandas as pd\n"
            "from lib import cellcomm as C\n"
            "rng=np.random.default_rng(5)\n"
            "df=pd.DataFrame(np.abs(rng.normal(1,0.4,(80,40))),columns=[f'g{i}' for i in range(40)])\n"
            "lab=np.array(['A']*40+['B']*40)\n"
            "try:\n"
            "    C.score(df,lab,[('NOTAGENE','ALSOMISSING')],n_perm=50,min_cells=10,seed=0)\n"
            "    print('RAISED False')\n"
            "except Exception: print('RAISED True')\n")
        self.assertIn("RAISED True", out)

    def test_negative_all_clusters_too_small_raises(self):
        """负向：所有 cluster 都太小 → 无法置换，必须报错。"""
        out = _run(
            "import sys; sys.path.insert(0,'.')\n"
            "import numpy as np, pandas as pd\n"
            "from lib import cellcomm as C\n"
            "rng=np.random.default_rng(6)\n"
            "df=pd.DataFrame(np.abs(rng.normal(1,0.4,(6,20))),columns=['CXCL12','CXCR4']+[f'g{i}' for i in range(18)])\n"
            "try:\n"
            "    C.score(df,np.array(['A','A','A','B','B','B']),[('CXCL12','CXCR4')],n_perm=50,min_cells=10)\n"
            "    print('RAISED False')\n"
            "except Exception: print('RAISED True')\n")
        self.assertIn("RAISED True", out)


class TestAttributionHonesty(unittest.TestCase):
    """溯源必须如实：新增来源要登记，许可不得虚报。"""

    def test_new_sources_are_recorded(self):
        txt = (PKG / "vendor" / "ATTRIBUTION.md").read_text(encoding="utf-8")
        for token in ("scanpy", "PyWGCNA", "CellphoneDB", "BSD-3-Clause", "MIT"):
            self.assertIn(token, txt, f"ATTRIBUTION.md 缺少 {token}")

    def test_no_gpl_in_vendor(self):
        """GPL 不得进入 vendor（会传染整个 skill）。"""
        bad = []
        for f in (PKG / "vendor").rglob("*.py"):
            head = f.read_text(encoding="utf-8", errors="replace")[:400].upper()
            if "GPL" in head and "NOT GPL" not in head:
                bad.append(str(f))
        self.assertEqual(bad, [], f"疑似 GPL 内嵌件：{bad}")

    def test_ported_files_declare_origin(self):
        """手工移植件（PORTED-FILE）必须写明来源与许可。"""
        for rel in ("vendor/_sc/core.py", "vendor/_wgcna/core.py",
                    "vendor/_batch/core.py", "vendor/_cellcomm/core.py"):
            head = (PKG / rel).read_text(encoding="utf-8")[:600]
            self.assertIn("PORTED-FILE", head, f"{rel} 缺少 PORTED-FILE 声明")
            self.assertTrue(("BSD-3" in head) or ("MIT" in head),
                            f"{rel} 未标注许可")


if __name__ == "__main__":
    unittest.main(verbosity=2)
