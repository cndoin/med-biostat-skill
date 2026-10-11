#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""capabilities.py —— 三层能力探测（Tier0 / Tier1 / 原生加速层）

回答一个问题：**在这台机器上，本 skill 现在能做什么、不能做什么、缺什么。**

  Tier 0    scripts/         纯标准库：取回 / 核验 / 质控 / 位点计数 / 互证 / 报告门禁
  Tier 1    vendor/          需 numpy+pandas(+scipy)：临床统计 / 生存 / 建模 / MR / 差异 / PTM / SVG 图
  原生加速  vendor/native/   单文件可执行（魔改版 biostat-vcf）：百万行汇总统计流式质控

用法：
    python scripts/capabilities.py --json
退出码：0（Tier0 永远可用；Tier1/原生层缺失不算失败，只是提示）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

PKG = Path(__file__).resolve().parents[1]

TIER0 = {
    "doctor.py": "环境自检",
    "capabilities.py": "双层能力探测",
    "fetch_parallel.py": "分块并行取回（续传+零进展检测）",
    "verify_integrity.py": "完整性五道核验",
    "qc_gwas.py": "GWAS 质控重算（口径必填）",
    "loci_count.py": "独立位点计数（LD/距离）",
    "loci_backbone.py": "可信集纯度过滤",
    "annotate_regions.py": "区间基因注释",
    "confounder_audit.py": "混杂十二项审计",
    "cross_verify.py": "两套官方产物逐变异互证",
    "report_pipeline.py": "确定性文档流水线（MD→HTML→门禁）",
    # —— 药物靶点 MR 链（v1.2 新增，全部纯标准库）——
    "cis_instrument.py": "cis 工具变量选择（±100/500kb, P<5e-8, F>10, 非回文）",
    "wald_mr.py": "单工具 Wald ratio MR + F 统计量（拒绝对单工具用 Egger）",
    "abf_coloc.py": "完整 ABF 共定位 H0–H4（含 p12/sdY 敏感性）",
    "testing_family.py": "检验家族分层（主 Bonferroni / 次要 BH）",
    "replication_check.py": "外部复现检查（方向 / 样本重叠 / ancestry）",
    "trigger_check.py": "自动触发自检（description 覆盖度 + 不触发边界）",
}
TIER1 = {
    "core.stats": "多重检验/效应量/缺失/bootstrap/假设检验",
    "core.contract": "数据契约与实验设计体检",
    "core.audit": "溯源审计（manifest）",
    "engines.clinical": "基线表 Table1 / 两组比较 / 风险表",
    "engines.survival": "KM / log-rank / Cox / PH 检验",
    "engines.differential": "差异表达 + 超几何富集",
    "engines.ptm": "PTM 占位率 / 局部 FDR",
    "engines.modeling": "自实现建模（无 sklearn）+ 校准 + 决策曲线",
    "engines.mr_coloc": "孟德尔随机化（多工具）+ 简化共定位",
    "viz.svg_charts": "纯 SVG 图表（无 matplotlib）",
    # —— lib/ 本 skill 自有扩展（v1.2 新增）——
    "lib.increment": "C-index(Harrell) / 连续 NRI / IDI / E-value",
    "lib.competing": "竞争风险累积发生率 CIF（Aalen–Johansen）",
    "lib.mixed": "线性混合效应（认知斜率等重复测量）",
    "lib.impute": "多重插补 MICE + Rubin's rules 合并",
    # —— v1.6：GitHub 开源项目魔改内嵌（详见 vendor/ATTRIBUTION.md）——
    "lib.mr_full": "MR 全套：IVW/加权中位数/Egger/简单众数/加权众数/RAPS/污染混合 + 7 项敏感性（源：pymr，Unlicense）",
    "lib.ophtha": "眼科白内障：IOL 光学(TILP/SE)、logMAR 换算、屈光结局、晶状体基因集（源：PEARL-DGS，MIT）",
    "lib.prep": "数据清洗与变量挑选：缺失报告/离群(IQR·Z·MAD)/低方差/相关去冗余/VIF",
    # —— v1.7：主流分析法补齐（详见 SKILL.md「分析法总表」）——
    "lib.network": "WGCNA 共表达网络：软阈值/邻接/TOM/模块/模块特征基因/kME·hub/模块-性状（源：PyWGCNA，MIT）",
    "lib.singlecell": "单细胞全流程：QC/归一化/HVG/PCA/邻居图/Louvain 聚类/marker/签名打分/t-SNE（源：scanpy，BSD-3）",
    "lib.pathway": "富集四件套：ORA 超几何 / GSEA / GSVA / ssGSEA + 通路活性两组比较（源：GSEApy，BSD-3）",
    "lib.immune": "免疫浸润：签名打分 / NNLS 去卷积 / ESTIMATE 基质-免疫分 + 两组比较",
    "lib.regularized": "正则化回归：Ridge/LASSO/ElasticNet 坐标下降 + CV 选 λ + Cox-LASSO",
    "lib.meta": "Meta 分析：固定/随机效应(DL)、I²/τ²、Egger/Begg、留一、亚组",
    "lib.causal": "观察性因果推断：倾向性评分/PSM/IPTW/G-computation/AIPW(双重稳健)/DiD/中介/阴性对照",
    "lib.prs": "多基因风险评分：clumping / P+T 评分 / AUC·R² 增量 / OR per SD / 分位风险",
    "lib.finemap": "精细定位：Wakefield ABF + SuSiE(IBSS 多因果) + 可信集纯度",
    "lib.mutational": "突变特征：SBS96 谱 / NMF 分解 / 参考谱余弦匹配 / TMB / APOBEC 筛查",
    "lib.spatial": "空间转录组：Moran's I / 空间可变基因 / 邻域富集 / 配体-受体筛查",
    "lib.trajectory": "轨迹推断：扩散伪时间(DPT) / MST 伪时间 / 分支点 / 轨迹相关基因",
    "lib.power": "样本量与功效 + 诊断准确性：两均值/两率/生存/相关 + ROC·LR·DOR·Youden",
    "lib.batch": "批次效应校正：ComBat 经验贝叶斯(位置+尺度)、参考批次模式、批次混杂诊断（源：scanpy，BSD-3）",
    "lib.cellcomm": "细胞通讯：配受体互作均值 + 置换 p 值 + 阳性比例 + 通路归类（源：CellPhoneDB，MIT）",
    # —— v1.7 后续：扩展方法（公开标准方法，非自造）——
    "lib.gxe": "G×E 交互：交互项检验 / 分层效应 / Johnson-Neyman 区间 / 相加交互 RERI·AP·S",
    "lib.epigen": "EWAS 甲基化：beta⇄M 转换 / limma 式 moderated t 差异甲基化 / DMP 汇总 / 细胞组成代理",
    # —— v1.8：跨龄单细胞方向补齐（S0 判决性预检的三件地基）——
    "lib.age_onset": "**年龄失守点（唯一裁决点）**：GAM 拟合表达~年龄 + 置信带越界 / 相对降幅两种口径 + 批量筛查（源：pyGAM，Apache-2.0）",
    "lib.scrublet": "双胞体检测：模拟双胞体 + kNN 贝叶斯打分 + 自动阈值，支持按供体分批（源：swolock/scrublet，MIT）",
    # —— v1.9：供体层级与转化出口（细胞类型 × 年龄 × 可成药 研究框架）——
    "lib.lm": "**moderated t 唯一裁决点**：Smyth 2004 经验贝叶斯方差收缩 + BH 校正，EWAS 与蛋白组共用同一套（保证两侧 p 值可比）",
    "lib.pseudobulk": "供体层级聚合：先按供体加 raw counts 再归一化（默认）/ log 均值（有偏口径）；防伪重复的第一道闸门",
    "lib.specificity": "**细胞类型特异性（唯一裁决点）**：log2 份额富集 ES + 归属标注 + 基因集秩和检验 + 表达率双口径互证（CELLEX/CELLECT 均为 GPL-3，独立实现）",
    "lib.proteomics": "蛋白组差异 + **RNA-蛋白断层（唯一裁决点）**：moderated t + 一致/断层/弱断层三分类 + 二项检验",
    "lib.composition": "供体层级差异丰度（成分数据）：Dirichlet-Multinomial + L1 选变量 + 供体 bootstrap（模型源自 scCODA，BSD-3；推断引擎已换底）",
    "lib.druggable": "可成药映射与干预窗口：基因→药物→给药途径分层 + X% 已上市药靶 + 阳性对照门（离线确定性，不碰网络）",
}
# 每个分析法模块的**可导入性**逐项探测（不靠 tier1_ready 一刀切：
# 某个模块缺依赖时，其余模块必须照常报可用，否则会误报「功能缺失」）
LIB_MODULES = [
    "increment", "competing", "mixed", "impute", "mr_full", "ophtha", "prep",
    "network", "singlecell", "pathway", "immune", "regularized", "meta",
    "causal", "prs", "finemap", "mutational", "spatial", "trajectory", "power",
    "batch", "cellcomm", "gxe", "epigen", "age_onset", "scrublet",
    "lm", "pseudobulk", "specificity", "proteomics", "composition", "druggable",
]
ZERO_DEP_IN_VENDOR = {
    "registry": "函数注册表（中英文检索 + 前置校验，纯标准库）",
    "spec": "TRIPOD+AI / STROBE 报告规范对表",
}


def _probe_lib_modules() -> dict:
    """逐个 import lib/* 模块，返回 {模块名: 是否可用}。

    为什么不用一次 try 包住全部：任何一个模块导入失败都会让整层被判为不可用，
    于是「A 模块缺依赖」被报成「所有分析法都没有」——这类误报会误导调用方
    以为功能缺失而去重复造轮子（v1.6 在 vendor 层踩过同一个坑）。
    """
    import importlib

    out = {}
    for name in LIB_MODULES:
        try:
            importlib.import_module(f"lib.{name}")
            out[f"lib.{name}"] = True
        except Exception:  # noqa: BLE001
            out[f"lib.{name}"] = False
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="双层能力探测")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    vendor_ok, caps = False, {}
    try:
        sys.path.insert(0, str(PKG))
        import vendor  # type: ignore
        caps = vendor.capabilities()
        vendor_ok = True
    except Exception as e:  # noqa: BLE001
        caps = {"import_error": repr(e)}

    tier1_ready = bool(caps.get("tier1_sci_stack_ready"))
    deps = caps.get("deps", {})

    missing = [m for m in ("numpy", "pandas", "scipy") if not deps.get(m, False)]

    # 原生加速层探测：二进制存在性 + 实际子命令（能力漂移即暴露）
    native = {"available": False, "reason": "未探测"}
    try:
        sys.path.insert(0, str(PKG / "tools"))
        import native_accel  # type: ignore
        native = native_accel.probe()
    except Exception as e:  # noqa: BLE001
        native = {"available": False, "reason": f"探测失败：{e!r}"}
    native_ready = bool(native.get("available"))

    report = {
        "ok": True,
        "summary": ("Tier0 就绪；Tier1 " + ("就绪" if tier1_ready else
                    f"未就绪（缺 {'/'.join(missing)}，Tier0 不受影响）")
                    + "；原生加速层 " + ("就绪" if native_ready else "不可用（Tier0 不受影响）")),
        "tier0": {
            "ready": True,
            "engine": "stdlib（纯标准库，跨 agent 裸环境可用）",
            "scripts": TIER0,
        },
        "tier1": {
            "ready": tier1_ready,
            "vendor_imported": vendor_ok,
            "modules": TIER1,
            "zero_dep_in_vendor": ZERO_DEP_IN_VENDOR,
            # 逐模块**真实导入探测**：模块级导入成功才算可用。
            # 用 tier1_ready 一刀切会把「某一个模块缺依赖」误报成「整层不可用」。
            "lib_modules": _probe_lib_modules(),
            "missing_deps": missing,
            "install_hint": "pip install -r vendor/requirements.txt",
        },
        "native": native,
        "vendor_capabilities": caps,
        "degrade_note": ("Tier1 缺失时 Tier0 全部功能照常；"
                         "需要 Tier1 能力时按 install_hint 安装即可。"
                         "原生层不可用时 qc_gwas.py 回退纯标准库路径，绝不伪造结果"),
    }
    C.log(f"Tier0: 就绪（{len(TIER0)} 个脚本）")
    C.log(f"Tier1: {'就绪' if tier1_ready else '未就绪'}；缺 {missing or '无'}")
    C.log(f"原生加速层: {'就绪' if native_ready else '不可用'}"
          f"{'' if native_ready else '（' + str(native.get('reason', '')) + '）'}")
    if vendor_ok:
        for name in ZERO_DEP_IN_VENDOR:
            C.log(f"  可用（零依赖）: vendor.{name}")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
