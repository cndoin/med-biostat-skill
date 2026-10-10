# ATTRIBUTION —— 内嵌工具来源与许可

本目录（`vendor/`）内的文件是**内嵌副本**，由 `tools/vendor_sync.py` 同步生成。
**不要直接编辑这些文件**——改源文件后重跑同步脚本。

## 逐件清单

| 目标 | 来源 | 许可 | 源 sha256 | 说明 |
|---|---|---|---|---|
| `vendor/core/stats.py` | medomics（自建） · `medomics/core/stats.py` | 自建/内部 | `95cddc52e7d40dd2` | 统计基础：多重检验/效应量/缺失/重抽样/检验 |
| `vendor/core/contract.py` | medomics（自建） · `medomics/core/contract.py` | 自建/内部 | `e05f10490625dc9f` | 数据契约与实验设计体检 |
| `vendor/core/audit.py` | medomics（自建） · `medomics/core/audit.py` | 自建/内部 | `34d8fe5d0e1c44e5` | 溯源审计（manifest 落盘） |
| `vendor/engines/clinical.py` | medomics（自建） · `medomics/engine/clinical.py` | 自建/内部 | `dbf340700d5e5244` | 基线表 Table1 / 两组比较 / 风险表 |
| `vendor/engines/survival.py` | medomics（自建） · `medomics/engine/survival.py` | 自建/内部 | `f281f53952814030` | 生存分析：KM / log-rank / Cox / PH 检验 |
| `vendor/engines/differential.py` | medomics（自建） · `medomics/engine/differential.py` | 自建/内部 | `78c7874b09b63d90` | 差异表达 + 超几何富集 |
| `vendor/engines/ptm.py` | medomics（自建） · `medomics/engine/ptm.py` | 自建/内部 | `6286a7c79b0b33a6` | 翻译后修饰：占位率 / 局部 FDR |
| `vendor/engines/modeling.py` | medomics（自建） · `medomics/engine/model.py` | 自建/内部 | `aa6312bc64846e5a` | 自实现建模（无 sklearn）+ 校准 + 决策曲线 |
| `vendor/engines/mr_coloc.py` | medomics（自建） · `medomics/engine/causal.py` | 自建/内部 | `2981f0f125fb1246` | 孟德尔随机化 + 简化共定位 |
| `vendor/viz/svg_charts.py` | medomics（自建） · `medomics/viz/charts.py` | 自建/内部 | `3f9ac53a98179925` | 纯 SVG 图表（无 matplotlib） |
| `vendor/registry.py` | mediscope（自建） · `mediscope/registry.py` | 自建/内部 | `cdf4b4c33437f25f` | 函数注册表：中英文检索 + 前置校验（纯标准库） |
| `vendor/spec.py` | mediscope（自建） · `mediscope/spec.py` | 自建/内部 | `cb1c77369d1df56b` | TRIPOD+AI / STROBE 报告规范对表 |
| `vendor/_deps.py` | mediscope（自建） · `mediscope/_optional.py` | 自建/内部 | `280058bb5a5fd872` | 可选依赖惰性加载与降级 |
| `vendor/_mr/methods.py` | pymr@maxghenis · `_upstream/pymr/src/pymr/methods.py` | Unlicense（公有领域） | `36a03c72bc0f2692` | MR 估计量：IVW / 加权中位数 / MR-Egger / 简单众数 / 加权众数 / MR-PRESSO / MR-RAPS / 污染混合 |
| `vendor/_mr/sensitivity.py` | pymr@maxghenis · `_upstream/pymr/src/pymr/sensitivity.py` | Unlicense（公有领域） | `7f282a8f621c4358` | MR 敏感性：Steiger 过滤 / Cochran Q / Rucker Q / 留一 / 单 SNP / 漏斗不对称 / Radial MR |
| `vendor/_mr/harmonize.py` | pymr@maxghenis · `_upstream/pymr/src/pymr/harmonize.py` | Unlicense（公有领域） | `ae2ee1550c5aa908` | 等位基因协调：列标准化 / 链翻转 / 回文位点剔除 |
| `vendor/_mr/multivariable.py` | pymr@maxghenis · `_upstream/pymr/src/pymr/multivariable.py` | Unlicense（公有领域） | `ece1ba0038d3ae32` | 多变量 MR（MVMR）：IVW 推广 + 工具变量强度诊断 |
| `vendor/_enrich/stats.py` | GSEApy@zqfang · `_upstream/GSEApy/gseapy/stats.py` | BSD-3-Clause | `85b97fa03069e308` | 富集统计量：超几何 ORA p 值 / BH-FDR / 多重检验校正 |
| `vendor/_enrich/algorithm.py` | GSEApy@zqfang · `_upstream/GSEApy/gseapy/algorithm.py` | BSD-3-Clause | `0ae8f98df9a83ea6` | GSEA 核心：加权 KS 富集分数 ES / 置换零分布 / NES / 名义 p / FDR；ssGSEA 的 fast_ssgsea |
| `vendor/_ophtha/pearldgs.py` | pearldgs_toolbox@gdebel · `_upstream/pearldgs_toolbox/pearldgs_toolbox.py` | MIT | `ef9737627bf46e01` | 眼科光学：Gullstrand 眼模型 / 薄透镜 / IOL 屈光力与有效位置（TILP）/ 角膜曲率换算 |

## 手工移植件（PORTED-FILE，非同步生成）

以下文件是**算法照抄、依赖换底**的移植件：上游直接绑 AnnData / sklearn / statsmodels /
numba / pyvis 等依赖，整文件内嵌会带进一堆不可用依赖，故按「掐头去尾取核心」的方式
移植，并在文件头标注 `PORTED-FILE` + 来源 + 许可 + 与上游的已知差异。

| 目标 | 来源 | 许可 | 移植说明 |
|---|---|---|---|
| `vendor/_wgcna/core.py` | mortazavilab/PyWGCNA · `PyWGCNA/wgcna.py` | **MIT** | 取 `pickSoftThreshold`/`scaleFreeFitIndex`/`adjacency`/`TOMsimilarity`/`moduleEigengenes`/`mergeCloseModules`；`statsmodels.ols`→`np.linalg.lstsq`（数值等价）、`sklearn.scale`→numpy 列标准化、去掉 matplotlib/pyvis/reactome2py |
| `vendor/_sc/core.py` | scverse/scanpy · `src/scanpy/**` | **BSD-3-Clause** | 取 QC/归一化/HVG/PCA/邻居/rank_genes_groups/score_genes 的算法；输入输出改为 numpy，去掉 anndata/sklearn/dask 绑定 |
| `vendor/_batch/core.py` | scverse/scanpy · `src/scanpy/preprocessing/_combat.py` | **BSD-3-Clause** | 取 ComBat（Johnson 2007）全套：设计矩阵/池化方差标准化/L-S 模型/逆 Gamma 先验/迭代条件后验；patsy→pandas one-hot、numba→纯 numpy、去 AnnData，改 (样本×特征) 接口 |
| `vendor/_cellcomm/core.py` | ventolab/CellphoneDB · `src/core/methods/cpdb_statistical_analysis_helper.py` | **MIT** | 取 `mean_analysis`/`percent_analysis`/置换 p 值口径；去数据库+complex 绑定与多进程，改「用户传配受体表」的 numpy/pandas 接口 |

### ⚠️ 一处**已更正的错误记录**（2026-10-11）

本文件先前把 **scanpy 记为 GPL**，据此把「单细胞分析」整块排除在技能包之外。
复核 GitHub 官方 SPDX 后确认：**scanpy 是 BSD-3-Clause**（Seurat 才是 GPL-3）。
也就是说这块能力**从来就可以合法内嵌**，是被一条错误记录挡住的。
教训：许可判定必须查**权威来源的 SPDX 字段**，不能凭印象；
一条写错的排除记录会让整条能力线凭空消失。

### 与上游的已知差异（移植件必须标注，否则会被当成 bug）

| 模块 | 差异 | 影响 |
|---|---|---|
| WGCNA | 模块识别用 scipy 层次聚类 + 最大跳跃剪枝，**非** R 的 dynamicTreeCut hybrid | 模块边界可能与 R 版不同；软阈值/邻接/TOM/ME/kME 一致 |
| 单细胞 | 聚类是 **Louvain**（Blondel 2008），**非** Leiden | 簇边界可能与 scanpy leiden 不同 |
| 单细胞 | 邻居图用自适应高斯核，**非** umap 模糊单纯形集 | connectivities 绝对值不可与 scanpy 对比 |
| 单细胞 | t-SNE 为精确梯度（O(n²)），**UMAP 未内嵌** | n>2000 慎用；UMAP 需另取 |
| 单细胞 | HVG 支持 seurat / cell_ranger；seurat_v3 需要的 LOESS 未内嵌 | 与 R Seurat v3 的 HVG 可能不同 |
| 批次校正 | `combat_reference`（参考批次模式）与 `batch_diagnostics` 为新增，非上游 scanpy 代码 | ref-batch 结果与 scanpy 全均值收缩口径不同；诊断是效应量描述非检验 |
| 细胞通讯 | **无 complex（多亚基）展开**；置换为单进程向量化 | 需 complex 时请传入已展开的 pair 表；p 值分布一致，仅速度不同 |

## 评估后未内嵌（去重决策记录）

同一能力只保留一份实现（设计文档附录 B 陷阱 P21：同一逻辑多处实现 → 结论不一致）。
以下文件经评估**不内嵌**，理由是已有等价实现：

| 未内嵌 | 理由 |
|---|---|
| `mediscope/desc.py` | 与 medomics/engine/clinical.py 的 table_one 重复 |
| `mediscope/stats.py` | 与 medomics/core/stats.py 重复（adjust_p/cohens_d/normality） |
| `mediscope/survival.py` | 与 medomics/engine/survival.py 重复（km_estimate/logrank） |
| `mediscope/model.py` | 与 medomics/engine/modeling.py 重复；且硬依赖 sklearn |
| `mediscope/report.py` | matplotlib 硬依赖；改用 medomics/viz/svg_charts.py |
| `mediscope/io.py` | 数据接入可用 vendor/core/contract.py 覆盖 |
| `mediscope/audit.py` | 与 medomics/core/audit.py 角色重复；能力判定并入 spec 层 |
| `medomics/engine/__init__.py` | 空文件，无内容 |

## 第三方开源项目

见 `references/tools.md` 的四档采纳记录。**GPL/LGPL/AGPL 系一律不内嵌**（GWASLab / PLINK2 / ldsc / coloc / DESeq2 等），仅通过 `tools/fetch_tools.py` 从官方源获取后独立调用，不随本 skill 分发。

## 评估后未内嵌（**许可红线**：GPL / LGPL / AGPL / 非开源一律不内嵌）

内嵌会把 GPL 传染给整个 skill。以下项目功能上很想要，但许可不允许。
按用户「搬不了就照抄一个」的要求，**算法公开的那部分已按论文/公开算法等价实现**，
并在代码注释里写明「这是等价实现，不是原项目的包装」。

| 项目 | 许可 | 想要的能力 | 替代做法 |
|---|---|---|---|
| `Seurat` | GPL-3 | 单细胞质控与聚类 | **scanpy（BSD-3）** 算法移植，见 `vendor/_sc/core.py` |
| `WGCNA`（R 包） | GPL-2 | 共表达网络 | **PyWGCNA（MIT）** 算法移植，见 `vendor/_wgcna/core.py` |
| `GSVA`（R 包） | GPL | 通路活性打分 | 按 Hänzelmann 2013 论文算法实现：`lib/pathway.gsva` |
| `glmnet` | GPL-2 | Ridge/LASSO/EN/Cox 路径 | 按 Friedman et al. 2010 (JSS) 公开算法实现：`lib/regularized` |
| `dynamicTreeCut` | GPL-2（R 包） | 动态剪枝 | scipy 层次聚类 + 最大跳跃剪枝（差异已标注） |
| `CellChat` | GPL-3 | 细胞通讯概率 | 只做共表达筛查并标注「非通讯概率」：`lib/spatial.ligand_receptor` |
| `CIBERSORT` / LM22 签名 | 非开源（需授权） | SVR 去卷积 + 官方签名矩阵 | NNLS 去卷积 + 调用方自备签名：`lib/immune.deconvolve` |
| COSMIC 突变特征谱 | 非开源数据（需接受条款） | SBS 参考谱命名 | 需调用方自行传入参考谱：`lib/mutational.match_cosmic` |
| `susieR` | NOASSERTION（许可非标准） | SuSiE 精细定位 | 按 Wang et al. 2020 (JRSS-B) 的 IBSS 实现：`lib/finemap.susie` |
| `metafor` | GPL-2 | Meta 分析 | 按 DerSimonian-Laird / Egger 公开公式实现：`lib/meta` |
| `Python-for-ophthalmologist` | **无 LICENSE** | 经典 vergence IOL 公式（SRK/T、Holladay、Hoffer Q、Haigis） | 仍缺（无 LICENSE = 默认保留全部权利，不可照抄） |
