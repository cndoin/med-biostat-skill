# Changelog

本项目的所有重要变更都记录在此。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 新增

- **眼科 · 经典 IOL 度数计算**（白内障术前最核心的一步）：`lib/ophtha.py` 新增
  **SRK/T**、**Hoffer Q**、**Holladay 1**、**Haigis** 四个第三代薄透镜公式 ——
  由眼轴 + 角膜曲率（+ 前房深度）算 IOL 度数，并做**多公式横向比较**。
  另含 A 常数 ⇄ ACD 常数换算、Haigis 的 d 反解（IOL 常数优化用）。
  按**公开发表论文独立实现**——可搜到的实现许可全部不合格（AGPL-3.0 或无 LICENSE），
  详见 [`vendor/ATTRIBUTION.md`](vendor/ATTRIBUTION.md) 的「按公开文献独立实现」一节。
  ⚠️ 验证到"正反算闭合 / 眼轴单调 / 四公式极差符合文献 / Haigis 对上公开算例"；
  **未**与厂商在线计算器逐例对拍，**临床决策前须用机构计算器复核**。

## [1.7.0] — 2026-10-11

首个公开版本。

### 新增 · 分析法（17 个）

- **WGCNA 共表达网络**：软阈值 / 邻接 / TOM / 模块识别 / 模块特征基因 / kME·hub / 模块–性状（源 PyWGCNA，MIT）
- **单细胞全流程**：QC / 归一化 / HVG / PCA / 邻居图 / Louvain 聚类 / marker 基因 / 基因集打分 / t-SNE（源 scanpy，BSD-3-Clause）
- **通路四件套**：ORA / GSEA / GSVA / ssGSEA（源 GSEApy，BSD-3-Clause）
- **免疫浸润**：签名打分 / NNLS 去卷积 / ESTIMATE 基质–免疫评分
- **正则化回归**：Ridge / LASSO / ElasticNet 坐标下降 + 训练集内 CV 选 λ + Cox-LASSO
- **Meta 分析**：固定 / 随机效应（DerSimonian–Laird）+ I²/τ² + Egger / Begg + 留一
- **观察性因果推断**：倾向性评分匹配 / IPTW / G-computation / AIPW（双重稳健）/ DiD / 中介分析 / 阴性对照
- **多基因风险评分 PRS**：clumping / P+T 打分 / AUC·R² 增量 / OR per SD
- **精细定位**：Wakefield ABF + SuSiE（多因果 IBSS）+ 可信集纯度
- **突变特征**：SBS96 谱 / NMF 分解 / 参考谱余弦匹配 / TMB
- **空间转录组**：Moran's I / 空间可变基因 / 邻域富集
- **轨迹推断**：扩散伪时间（DPT）/ MST 伪时间 / 分支点
- **样本量与功效** + 诊断准确性（ROC / 似然比 / DOR / Youden）
- **批次效应校正 ComBat**：经验贝叶斯位置+尺度收缩、参考批次模式、批次混杂诊断（源 scanpy，BSD-3-Clause）
- **细胞通讯**：配受体互作均值 + 置换 p 值 + 阳性比例 + 通路归类（源 CellPhoneDB，MIT）

### 新增 · 工程

- 跨 agent 安装器 `scripts/install_adapters.py`：单一 `SKILL.md` 来源，生成
  Claude Code / Codex / WorkBuddy / Cursor / Cline / Gemini CLI / GitHub Copilot / AGENTS.md 各端入口
- 自动触发自检 `scripts/trigger_check.py`（29 正向 / 4 负向）
- 双层能力探测 `scripts/capabilities.py`（22 个 lib 模块逐个探测可导入性）
- README 三语化（English / 简体中文 / 日本語）；矢量品牌图（浅色/深色双版）

### 修复

- **VIF 必须在相关剪枝「前」计算** —— 原先在剪枝后算，共线性被自己抹平（实测 r=0.998 的两列在剪枝后 max_vif=1.019）
- **离群检测三种口径的默认 k 各按方法惯例取值**（iqr 1.5 / zscore 3.0 / mad 3.5）—— 原先共用 1.5，会把均匀分布两端误判为离群（约 13%）
- **Cox 偏似然风险集方向写反**，导致系数符号全部相反
- **SuSiE 残差空间用错**，多因果位点无法分离（多个可信集指向同一位点）
- **多重检验校正返回值解包错位**，`padj` 被写成布尔值
- 免疫浸润签名基因零匹配时静默返回全 0 → 改为显式报错

### 变更

- 「不搬运开源项目源码」的旧描述**已更正**：非 GPL 的开源件现按许可内嵌，
  仅 GPL 系（GWASLab / py-twosamplemr / Seurat 等）仍走外部获取后独立调用

### 说明

- 全量自检 **212 / 215 通过**，0 失败 0 错误（3 项因本机缺 `lifelines` 而跳过）
- 公开 API：`lib/` 125 + `scripts/` 77 = 202 个

## [1.6.0] — 2026-10-10

- 按「从 GitHub 开源项目掐头去尾取核心」的原则引入第一批第三方算法：
  **pymr**（Unlicense，孟德尔随机化全套）、**GSEApy**（BSD-3-Clause，富集统计量）、
  **PEARL-DGS**（MIT，眼科 IOL 光学）
- 新增 `lib/mr_full.py`（7 估计量 + 7 敏感性 + MVMR）、`lib/ophtha.py`（IOL 屈光 / logMAR / 屈光结局）、
  `lib/prep.py`（缺失 / 离群 / 低方差 / 去冗余 / VIF / 标准化 / 独热）
- 移植件统一加 `PORTED-FILE` 文件头（来源 + 许可 + 与上游已知差异）
- **更正一条错误记录**：scanpy 是 **BSD-3-Clause**，不是 GPL —— 此前误记导致
  「单细胞分析」整块能力被排除在包外

## [1.5.0] 及更早

内部迭代版本：Tier 0（纯标准库执行层）、Tier 1（vendor 内嵌层）、
原生加速层（源码级 fork `rsomics-vcf`，新增 `sumstats` 子命令，百万行汇总统计流式质控）。
未单独记录变更。
