---
name: biostat-copilot
description: 生信 / 医学研究全流程编排与审计（口径先于数值 · 一致性强于看起来对 · 混杂是主战场 · **数字一律由程序算出，不由 AI 推断**）。**何时自动加载**：用户要动手做、复核或投稿一项医学/生信分析时——如「这个方向能不能做、有没有人做过」「帮我分析这份 GWAS/组学/队列数据」「核对一下我这组结果对不对」「这个药/基因有没有因果证据」「把它写成可投稿的报告」。**覆盖七类形态**：(1) 药物靶点与因果推断——药物靶点 MR、孟德尔随机化（IVW/加权中位数/Egger/众数/RAPS/污染混合 + 异质性、留一、Steiger、Radial）、cis-eQTL/pQTL 工具变量、Wald ratio、共定位 coloc/PP.H4、药物重定位；(2) 遗传关联——GWAS 汇总统计处理与质控、λGC、独立位点与可信集、精细定位、区间基因注释；(3) 组学整合——eQTL/pQTL/TWAS/PWAS、组织与细胞类型富集、单细胞归属；(4) 临床统计——基线表、生存分析、预测模型（区分度+校准度+决策曲线）、C-index/NRI/IDI、竞争风险、混合效应；(5) 差异与通路——差异表达、共表达网络（WGCNA）、富集（ORA/GSEA/GSVA/ssGSEA）、单细胞、批次校正 ComBat、细胞通讯配受体、免疫浸润、空间转录组、拟时序、突变特征；(6) **眼科与白内障（重点方向）**——白内障 GWAS/遗传易感位点、晶状体蛋白基因集（crystallin/MIP/GJA8/EPHA2 等）富集、**人工晶体 IOL 屈光计算（PEARL-DGS 厚透镜 TILP/等效球镜）**、logMAR·Snellen 视力换算、屈光预测误差（ME/MAE/MedAE/±0.5D 内比例）、眼压统计、角膜曲率与眼轴；(7) 数据交付与审计——数据清洗与变量挑选（缺失/离群/共线性 VIF）、数据源分级、**与官方发布的一致性核验（哈希/字节数/逐变异互证）**、混杂十二项审计、外部复现、报告规范对表（STROBE/STROBE-MR/TRIPOD+AI/PRISMA）。**触发词**：生信分析、GWAS、孟德尔随机化、MR、共定位、coloc、eQTL、pQTL、TWAS、富集分析、精细定位、位点、差异表达、WGCNA、共表达网络、生存分析、预测模型、单细胞、聚类、分群、批次效应、批次校正、ComBat、细胞通讯、配受体、免疫浸润、去卷积、空间转录组、拟时序、突变特征、TMB、蛋白组、PTM、药物靶点、倾向性评分、PSM、IPTW、因果推断、Meta 分析、荟萃分析、LASSO、岭回归、变量筛选、样本量、功效、PRS、多基因风险评分、**白内障、眼科、晶状体、人工晶体、IOL、屈光、眼轴、角膜、视力、logMAR、眼压、青光眼、ophthalmology、cataract、intraocular lens、refractive**、数据清洗、bioinformatics、colocalization、fine-mapping、Mendelian randomization、drug target、single-cell、single cell clustering、pseudotime、batch effect、ComBat、cell-cell communication、CellPhoneDB。**不触发**：单纯查文献、单纯问概念、单纯要一张图、与医学/生信无关的任务。This skill should be used when a user plans, executes, audits, or writes up a bioinformatics or medical research analysis end-to-end.
license: MIT
version: 1.7.0
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, WebFetch, WebSearch
agent_created: true
metadata:
  author: 寇豆码
  category: bioinformatics
  source: "生信分析Skill_全量设计文档 v2.0 (2026-10-10)"
  tags: [bioinformatics, gwas, mendelian-randomization, colocalization, eqtl, fine-mapping, clinical-statistics, confounder-audit, reproducibility]
---

# biostat-copilot · 生信研究全流程副驾

把**医学 / 生信研究**从「方向能不能做」管到「结果能不能投出去」。
脚本负责**执行与结构化数据**，AI 负责**路由与判断**；任何数字都不许凭记忆补全。

## 三条核心主张（设计冲突一律以此裁决）

1. **口径先于数值。** 同一个名字在不同口径下是不同的量——λGC 要问「哪个 MAF 分档」，显著位点要问「是变异数还是独立位点数」，MAF 要问「次等位频率还是 ALT 频率」。**缺口径即报错，不给默认值。**
   > 实证：文章里的「49 个」与官方的「61 个」被当成同一个量比较——前者是 chr1 局部原始变异计数，后者是全基因组独立位点数。**49 与 61 本来就不矛盾。**
2. **一致性强于「看起来对」。** 「与官网一致」必须由哈希与字节数证明，不能由「数值差不多」推断。跨文件合并必须做「落到具体变异的数值互证」。
   > 实证：FinnGen 精细定位文件 1,604 个变异回全量文件比对，P 与 beta 最大绝对偏差为 **0**，MAF 偏差 **5.6×10⁻¹⁷**（浮点机器精度）。
3. **混杂是主战场。** 翻车绝大多数不在算法，在混杂（S4 独立一层，十二项逐条过，每项给可执行检测手段）。
   > 实证：白内障最强信号 LOXL1（chr15:73,931,230）与剥脱综合征青光眼**是同一个变异（距离 0 kb）**——这是共病混杂，不是新发现。

## 明确不是什么（入口处直接拦截）

- **不是一键出图模板生成器**：eQTL+pQTL+MR+PheWAS+PPI+分子对接+细胞实验「七件套」；GEO 芯片 DEG+WGCNA+LASSO+免疫浸润；铁死亡/铜死亡/m6A 套模板；纯网络药理学；只报 AUROC 的预测模型。**见到即停下说明理由，不照做。**
- **不是数值生成器**：不许「看起来合理」的数进入交付物。
- **不是方向推销员**：方向被查重判定已覆盖，义务是**叫停**。
- **不是文献综述工具**：单纯查文献 / 问概念 / 要一张图 → 不触发。

## 十层架构（一句话职责）

| 层 | 职责 | 不通过会怎样 |
|---|---|---|
| **S0** 入口与编排 | 决定该不该做、走哪条线、什么条件下停 | 做完才发现方向被占 |
| **S1** 数据层 | 数据是真的、全的、与官方一致、版本固定 | 结论建立在残缺/错版数据上 |
| **S2** 质控层 | 每个指标都有定义、且在该定义下计算 | 拿 A 口径的数去比 B 口径的基准 |
| **S3** 分析层 | 用正确的统计与遗传方法回答假设 | 方法与问题不匹配 |
| **S4** 偏倚层 | 把混杂逐项查出来并量化 | 把混杂当成发现 |
| **S5** 验证层 | 证明结论不是某一次参数下的偶然 | 结论不可复现 |
| **S6** 产出层 | 能读懂、能重算、能投对刊 | 格式退稿 / 数据不可信 |
| **S7** 工程层 | 可重跑、可测试、不静默出错 | 静默失败 |
| **S8** 领域包 | 提供组织参考与判据（按需加载） | 用错参考系 |
| **S9** 红线与门 | 强制止损，防止沉没成本与包装 | 把阴性包装成阳性 |

## 路由决策树

| 用户给的是 | 走哪条线 | 产出 |
|---|---|---|
| **一个想法** | S0（含命题级查重） | 方向裁定（可做 / 不可做 + 依据） |
| **一个数据集 + 一个问题** | S1 → S2 → S3 | 完整分析结果 |
| **一份已有结果，要我核对** | S2 复算 + S4 混杂审计 | 审计报告 + 勘误表 |
| **要我产出一篇** | 全链路 + S6 | 正文 + 图 + 补充材料 + 代码标识 |

**命题级查重（S0 硬前置）**：三条检索路径**同时撞库**，任一命中「命题已被正面回答」→ 触发判定门 ① 停止。
- **概念路径**：这个「问法」本身有没有人提过
- **技术路径**：这个「技术组合」有没有人用过
- **实体路径**：具体基因/位点/数据集有没有被同一问题用过

## 红线（Red Lines）

| # | 红线 | 处置 |
|---|---|---|
| R1 | 使用「七件套」等套路模板 | 拒绝执行并说明 |
| R2 | 数字无出处（脚本 / 可引用官方值 + 口径） | 拒绝写入交付物 |
| R3 | 指标缺口径就输出 | 报错，不给默认值 |
| R4 | 数据与官方一致性只凭「差不多」 | 要求哈希 / 字节 / 逐变异互证 |
| R5 | 把共享对照当独立证据合并 | 检测对照数，禁止合并 |
| R6 | 低纯度可信集不过滤 | 按 `good_cs` 过滤并记录剔除行数 |
| R7 | 用「最近基因」当区域基因 | 改用权威区间注释 |
| R8 | 只报 AUROC | 补齐校准度 + 决策曲线 |
| R9 | 阴性结果包装成阳性 | 按阴性写 |
| R10 | 含数字的报告交给写作代理改写 | 走确定性转换，禁止改写 |
| R11 | 事后更换口径 / 阈值 | 判定门前置，禁止事后调整 |
| R12 | 用截断数据当全量 | 冒烟测试必须带 `--max-*` 且产物写 `truncated=true` |

## 判定门（动手前写好，落盘 `logs/gates.json`）

| 门 | 触发时点 | 判定条件（可检验） | 处置 |
|---|---|---|---|
| **① 命题门** | S0 完成 | 概念/技术/实体三路径任一命中「命题已被正面回答」 | **停止**，不得换措辞继续 |
| **② 数据门** | S1 完成 | 关键源不可得，或三级源承担核心暴露且无法落实 | 换源或降级，**不得用近似数据顶替** |
| **③ 质控门** | S2 完成 | 可用变异低于下限；或指标与官方基准偏差超阈值且原因未查明 | 先排查口径，**未查明不得进 S3** |
| **④ 阳性对照门** | S3 首轮 | 阳性对照不成立 | **停止主分析**，先修参考数据，不强行解释 |
| **⑤ 增量门** | S3 主结论后 | 新方法相对已有方法一致性高于上限（无新信息） | 停止，方向无增量 |
| **⑥ 阴性门** | 全程 | 得「P>0.05 且无解释力」 | **不得包装成阳性**，按阴性写 |

> 门是有效的，但**门的输入必须先经审计**——一个错的输入会让门误触发（实证：原产物误报「阳性对照为 0」，本会误触门 ④）。

## 什么情况下读哪个 reference（按需加载，零常驻成本）

| 触发场景 | 读 |
|---|---|
| 选数据源、判可达性、查版本时效 | `references/data_sources.md` |
| 输出任何指标前的口径核对 | `references/qc_definitions.md` |
| 逐项过混杂、写混杂审计表 | `references/confounders.md` |
| 选检验、算效应量、建预测模型 | `references/stats_criteria.md` |
| 定图表数量、对报告规范、选刊 | `references/reporting_standards.md` |
| 采纳开源工具、查许可 | `references/tools.md` |
| **动手前先扫一遍已知陷阱** | `references/pitfalls.md` |
| 眼科 / 晶状体 / 视网膜方向 | `references/domain/ophthalmology.md` |

**规则**：先读 `pitfalls.md` 再动手；输出任何指标前读 `qc_definitions.md` 对应行。

## 脚本调用（Windows 本机示例）

```bash
PY="C:/Users/35680/.workbuddy/binaries/python/versions/3.13.12/python.exe"
SK="C:/Users/35680/Desktop/科研0.1/biostat-copilot/scripts"

# 环境自检
"$PY" "$SK/doctor.py" --json

# 双层能力探测（Tier0 纯标准库 / Tier1 内嵌引擎）
"$PY" "$SK/capabilities.py" --json

# 自动触发自检（description 覆盖度 + 不触发边界）
"$PY" "$SK/trigger_check.py" --json

# 分块并行取回（小分块是硬要求；带断点续传与零进展检测）
"$PY" "$SK/fetch_parallel.py" --url "<URL>" --out raw/x.gz --chunk-mb 8 --concurrency 16 --json

# 完整性五道核验
"$PY" "$SK/verify_integrity.py" --file raw/x.gz --expect-bytes <N> --expect-md5 <MD5> --json

# 质控重算（口径必填）
"$PY" "$SK/qc_gwas.py" --input raw/x.gz --bucket maf_0.01_0.5 --json

# 独立位点计数（LD 与距离两口径 + 窗口敏感性）
"$PY" "$SK/loci_count.py" --input raw/x.gz --p-threshold 5e-8 --window-kb 1000,3000 --json

# 区间基因注释 / 可信集清洗
"$PY" "$SK/loci_backbone.py" --finemap <snp.filter.tsv> --good-cs --json

# 混杂十二项审计
"$PY" "$SK/confounder_audit.py" --project <项目dir> --json

# 两份官方产物逐变异互证
"$PY" "$SK/cross_verify.py" --full raw/x.gz --subset finemap/x.snp.filter.tsv --json

# 药物靶点 MR 链（也可用 pipeline/run_drug_target_mr.py 一条命令跑完）
"$PY" "$SK/cis_instrument.py" --eqtl eqtl.tsv --genes genes.tsv --window-kb 100 --json
"$PY" "$SK/wald_mr.py" --instruments mr_input.tsv --json
"$PY" "$SK/abf_coloc.py" --trait1 t1.tsv --trait2 t2.tsv --p12-list 1e-6,5e-6,1e-5 --json
"$PY" "$SK/testing_family.py" --input tests.tsv --json
"$PY" "$SK/replication_check.py" --discovery d.tsv --replication r.tsv --json
```

- 用 Bash 工具调用（PowerShell 不回显 stdout）；Git Bash 缺 `ls/head/tail/dirname`
- 所有命令支持 `--json`；**AI 决策时一律加 `--json` 读结构化输出**
- **stdout 只有数据，人类提示走 stderr**；解析失败就重跑并贴原始输出，不要靠猜
- 路径用正斜杠，含空格加引号

## 工作流 A：方向裁定（用户给「一个想法」）

1. 抽取研究对象 / 数据约束 / 期望产出 → 映射到六类形态之一。
2. **命题级查重**：概念 / 技术 / 实体三路径撞库（WebSearch + PubMed/Europe PMC）。
3. 命中 → 判定门 ① 停止，写 `logs/gates.json`，说明被谁覆盖。
4. 未命中 → 写 `logs/00_intake.json`（形态、路由、判定门清单）+ 预算与退出条件。
5. 产出方向裁定：可做 / 不可做 + 依据 + 预设判定门。

## 工作流 B：数据 → 结果（用户给「数据集 + 问题」）

1. `doctor.py --json` 确认环境；读 `references/data_sources.md` 定来源层级。
2. `fetch_parallel.py` 取回（小分块 + 续传 + 零进展检测）→ `verify_integrity.py` 五道核验。
3. `logs/provenance.json` 冻结版本 + 哈希 + 获取日期。
4. 先读 `qc_definitions.md` → `qc_gwas.py` 重算质控（口径必填）→ 判定门 ③。
5. `loci_count.py` 独立位点（LD 优先，距离退化须报窗口 + 敏感性）；`loci_backbone.py` 可信集纯度过滤 + 区间注释。
6. `confounder_audit.py` 十二项 → `logs/confounder_audit.json`（**无法查的写「无法查」，不能写「未见异常」**）。
7. 阳性对照 + 阴性对照 → 判定门 ④；内部/外部/正交三级验证。

## 工作流 C：审计已有结果（用户要「帮我核对」）

1. 复算：所有关键数字用脚本重算，与原文逐一对照。
2. 口径核对：每个指标问「定义是什么、在哪定义」。
3. 混杂十二项：逐条给「已查 / 结论 / 无法查及原因」。
4. 出审计报告 + 勘误表；命中的门逐条落盘。

## 工作流 D：产出（用户要「产出一篇」）

1. 报告规范对表（`reporting_standards.md`）：STROBE / STROBE-MR / TRIPOD+AI / PRISMA。
2. 图表纪律：主图 4–8 张，每张推进一步论证。
3. `report_pipeline.py`：Markdown → 模板 HTML → 门禁 → DOCX（**确定性转换，禁止写作代理改数字**）。
4. 可重算：每张图的数字都能由脚本重跑得到；附 source data。

## 工作流 E：药物靶点 MR（药物分析主力）

dhp / ptgds / lpdc 三个真实项目共用同一条链。**这条链可一条命令确定性跑完**：

```bash
"$PY" pipeline/run_drug_target_mr.py --config pipeline/config.example.json --json
```

链：`cis_instrument → wald_mr → abf_coloc → testing_family → replication_check`

| 步骤 | 脚本 | 关键约束（来自 PROTOCOL） |
|---|---|---|
| cis 工具变量 | `cis_instrument.py` | 基因边界 ±100 kb（敏感性 500 kb / full cis）、P<5e-8、**F>10**、**排除回文双等位**、一基因一组织一 lead |
| 单工具 MR | `wald_mr.py` | Wald ratio + delta 法 se；**拒绝对单工具使用 MR-Egger / 异质性 / 留一法** |
| 共定位 | `abf_coloc.py` | 完整 Wakefield ABF 的 H0–H4 分解；**必须用完整未筛选区域统计量，不得只取显著子集**；p12 ∈ {1e-6,5e-6,1e-5}、sdY ∈ {0.5,1,1.5} 敏感性 |
| 多重检验 | `testing_family.py` | 主检验 **Bonferroni**、次要 / 探索性 **BH-FDR**，每族独立校正 |
| 外部复现 | `replication_check.py` | 方向一致性 / 样本重叠 / ancestry；**不合并跨阶段效应量** |

**产物** `run_manifest.json`：config 哈希 + 参数 + 脚本哈希 + 每步产物哈希 + 耗时。
用途是证明「这组数字确实来自这次运行」。**AI 只负责写配置与读结果，不产生任何数值。**

药物 MR 的四个核心模块（cis_instrument / wald_mr / abf_coloc / testing_family）**全部是纯标准库**，
在无 numpy 的裸环境也能跑通整条链。

## 三层能力（Tier0 / Tier1 / 原生加速层）

| 层 | 位置 | 依赖 | 能力 |
|---|---|---|---|
| **Tier 0** | `scripts/` | **纯标准库** | 取回 / 核验 / 质控 / 位点计数 / 可信集过滤 / 区间注释 / 混杂审计 / 逐变异互证 / 报告门禁。**任何 agent 的裸 Python 都能跑** |
| **Tier 1** | `vendor/` + `lib/` | numpy + pandas + scipy | 临床统计（Table1/比较/风险表）/ 生存（KM/log-rank/Cox/PH）/ 建模（无 sklearn，含校准+决策曲线）/ 孟德尔随机化（多工具）+ 共定位 / 差异表达+富集 / PTM / 纯 SVG 图表 / 函数注册表 / TRIPOD+AI·STROBE 对表 / **C-index·NRI·IDI·E-value**（`lib/increment`）/ **竞争风险 CIF**（`lib/competing`）/ **线性混合效应**（`lib/mixed`）/ **MR 全套 7 法 + 7 项敏感性**（`lib/mr_full`，源 pymr）/ **眼科白内障**（`lib/ophtha`，源 PEARL-DGS）/ **数据清洗与变量挑选**（`lib/prep`） |
| **原生加速层** | `vendor/native/` + `tools/native_accel.py` | 无（单文件可执行） | **百万级汇总统计流式质控**（`sumstats`）：λGC + MAF 分档 + 12 类异常计数，rayon 并行，直接读 `.tsv` / `.tsv.gz`。**魔改自 `rsomics-vcf`（MIT/Apache-2.0），只做加法** |

**降级规则**：Tier1 缺依赖时**不影响 Tier0**；`vendor/registry`、`vendor/spec` 是纯标准库，即使无 numpy 也可用。
原生层不可用（二进制缺失 / 架构不符）时**不影响 Tier0**，`qc_gwas.py --native` 会**显式告警并回退**，绝不伪造结果。
装 Tier1：`pip install -r vendor/requirements.txt`，或 `python tools/fetch_tools.py --fetch-all --tier A`。
能力探测：`python scripts/capabilities.py --json`。

### 原生加速层（`vendor/native/`）—— 用法与口径红线

- 探测：`python tools/native_accel.py --list`（会真跑 `--help` 比对二进制自报的子命令，**能力漂移即非零退出**）。
- 调用：`python tools/native_accel.py --input gwas.tsv.gz --json`。
- 接入：`python scripts/qc_gwas.py --input gwas.tsv.gz --bucket maf_0.01_0.5 --native --json`。
- 原样透传（保证「包装层不是功能上限」）：`python tools/native_accel.py official -- <任意上游子命令>`。

⚠️ **口径红线（最容易被踩）**：原生层的 λGC 是 **Wald 均值口径**
`mean((β/se)²)/0.4549364231195727`；Tier0 `qc_gwas.py` 的 λGC 是
**p 值中位数口径** `median(χ²(p))/0.4549364`。**两者数值不同、都合法，但绝不可混用或相互替换。**
因此：原生路径只在用户显式 `--native` 时启用，且结果必带
`lambda_caliber` 与 `engine`（`native:biostat-vcf/sumstats` vs `stdlib:qc_gwas.py`）两个可核字段。
上游退出码：`0` 通过 / `1` 校验阻断 / `2` 配置错 / `3` 上游错 / `4` IO 错（`3` **不是**阻断）。

### 内嵌工具来源（务必先读）

- `vendor/` 内文件由 `tools/vendor_sync.py` 从上游同步生成（带 `# VENDORED-FILE` 头），**不要直接编辑**；改源后重跑同步脚本。**修复也必须写进同步脚本的 `REWRITES`**，否则下次同步会被覆盖。
- 逐件来源 / 许可 / 哈希：`vendor/ATTRIBUTION.md`（含"评估后未内嵌"的去重决策记录）。
- **开源项目为何没有内嵌源码**：`vendor/opensource/README.md`（GWASLab 的"许可不一致"已结案：v3.3.x 及以前 MIT、v3.4.39 起 GPL-3.0-only）。
- GPL / 重型 / 外部二进制工具：`tools/fetch_tools.py` 装到 `~/.biostat-copilot/tools/`，**只作对拍、不随 skill 分发**。

## 分析法总表（AI 调用的第一站：先查这张表，再动手）

**35 个分析法 · 全部由程序确定性计算，AI 只负责选方法与读结果，不产生任何数值。**
用户问「用什么方法分析」或「帮我做 X 分析」时，**先在这张表里定位**，再跳到对应模块。

### 一、因果与遗传（10 个）

| # | 分析法 | 入口 | 关键口径 / 必须一起报告的量 |
|---|---|---|---|
| 1 | **孟德尔随机化 MR** | `lib.mr_full.run_all` / `diagnose` | 7 估计量（IVW/加权中位数/Egger/简单众数/加权众数/RAPS/污染混合）+ 7 敏感性（Cochran Q/Rucker Q/留一/单 SNP/Steiger/漏斗/Radial）；单工具**只配 Wald ratio**，拒绝 Egger 与异质性 |
| 2 | 药物靶点 cis-MR 链 | `scripts/cis_instrument → wald_mr → abf_coloc → testing_family` | 基因±100 kb、P<5e-8、**F>10**、排除回文；一基因一组织一 lead |
| 3 | 共定位 coloc | `scripts/abf_coloc.py` | Wakefield ABF 的 H0–H4；**必须用完整未筛选区域统计量**；p12∈{1e-6,5e-6,1e-5} |
| 4 | **精细定位** | `lib.finemap.susie` / `abf` | SuSiE（多因果 IBSS）+ 可信集；**必须给 LD 矩阵**，无 LD 的精细定位是自欺欺人；报告 `prior_variance` |
| 5 | 多基因风险评分 PRS | `lib.prs.clump/score/evaluate` | 阈值只能在**验证集**选；报告 OR per SD / AUC 增量 / 祖源匹配性 |
| 6 | GWAS 汇总统计质控 | `scripts/qc_gwas.py` / 原生 `sumstats` | λGC 两口径**不可混用**（Wald 均值 vs p 中位数），结果必带 `lambda_caliber` |
| 7 | 独立位点与可信集 | `scripts/loci_count.py` / `loci_backbone.py` | 「变异数」≠「独立位点数」；低纯度可信集按 `good_cs` 过滤 |
| 8 | **观察性因果推断** | `lib.causal.aipw/iptw/psm/g_computation` | 首选 **AIPW（双重稳健+交叉拟合）**；平衡用 **SMD<0.1**，不用 p 值；报告 positivity |
| 9 | 中介分析 | `lib.causal.mediate` / `bootstrap_indirect` | 间接效应 CI 必须 **bootstrap**（a×b 非正态，Sobel 在小样本失真） |
| 10 | DiD / 阴性对照 | `lib.causal.did` / `negative_control_outcome` | DiD 依赖平行趋势（不可直接检验）；对照结局显著≠0 → 残余混杂警报 |

### 二、组学与通路（10 个）

| # | 分析法 | 入口 | 关键口径 |
|---|---|---|---|
| 11 | 差异表达 | `vendor.engines.differential` / `lib.singlecell.markers` | logFC 用 **expm1 差值**（scanpy 口径）；必须多重检验校正 |
| 12 | **WGCNA 共表达网络** | `lib.network.wgcna` | 报告 **power + deep_split + 聚类规则**，否则无法复现；kME 挑 hub |
| 13 | ORA 过表征 | `lib.pathway.ora` | **背景基因集必须给**（不给默认 20000 会让 p 系统性偏小） |
| 14 | **GSEA** | `lib.pathway.gsea` | NES/p 由置换得到，报告 `n_perm`（最小 p = 1/n_perm） |
| 15 | **GSVA / ssGSEA** | `lib.pathway.gsva` / `ssgsea` | 把 基因×样本 转成 通路×样本；GSVA 数值不可与 R 版逐位对比 |
| 16 | **单细胞全流程** | `lib.singlecell.standard_pipeline` | 聚类是 **Louvain 非 Leiden**；连接度非 umap 模糊集；结果必须带差异说明 |
| 17 | 免疫浸润 | `lib.immune.deconvolve` / `signature_score` / `estimate_score` | 打分是**相对富集**（跨细胞类型不可比）；去卷积才是比例 |
| 18 | 拟时序 / 轨迹 | `lib.trajectory.diffusion_pseudotime` / `mst_pseudotime` | 伪时间是**相对顺序**；起点必须人工指定；不证明因果 |
| 19 | 空间转录组 | `lib.spatial.morans_i` / `spatial_variable_genes` | Moran's I 的 p 必须**置换检验**；spot 数据不能做细胞分辨率通讯 |
| 20 | 突变特征 | `lib.mutational.sbs96_profile` / `nmf_decompose` | NMF 秩 k 需稳定性检验；COSMIC 参考谱需自供（非开源数据） |

### 三、临床统计与建模（11 个）

| # | 分析法 | 入口 | 关键口径 |
|---|---|---|---|
| 21 | 生存分析 KM / Cox | `vendor.engines.survival` | 报告事件数而非样本数；PH 假设需检验 |
| 22 | 竞争风险 CIF | `lib.competing.cif` | 有竞争事件时不该用 KM（会高估） |
| 23 | 预测模型 + 校准 + DCA | `vendor.engines.modeling` | **只报 AUROC 触发红线 R8** |
| 24 | C-index / NRI / IDI / E-value | `lib.increment` | 增量指标需与基线模型比 |
| 25 | **正则化回归** | `lib.regularized.cv_elastic_net` / `cox_lasso` | λ 必须**训练集内 CV**选，默认 `lambda_1se`；全数据选 λ 会泄漏 |
| 26 | **Meta 分析** | `lib.meta.meta_analysis` | 固定/随机(DL) + **I²/τ² 必须报**；异质时禁用固定效应 |
| 27 | 发表偏倚 | `lib.meta.egger_test` / `begg_test` / `leave_one_out` | 研究数<10 时功效很低，不显著≠无偏倚 |
| 28 | 多重插补 MICE | `lib.impute.mice` + `pool_rubin` | 缺失是否填补由调用方显式决定，不默认丢行 |
| 29 | 多重检验校正 | `scripts/testing_family.py` | 主检验 Bonferroni、探索性 BH，**每族独立校正** |
| 30 | 混合效应 / 重复测量 | `lib.mixed.mixed_slope` | 需 statsmodels |
| 31 | 样本量与功效 | `lib.power.power_*` / `sample_size_*` | 生存功效由**事件数**决定；禁止事后功效分析 |

### 四、数据工程与眼科（4 个）

| # | 分析法 | 入口 | 关键口径 |
|---|---|---|---|
| 32 | 数据清洗与变量挑选 | `lib.prep.prep_pipeline` | **VIF 在相关剪枝前算**；离群三口径默认 k 不同（iqr1.5/z3.0/mad3.5） |
| 33 | **IOL 度数计算 + 屈光位置** | `lib.ophtha.iol_power_comparison`（术前算度数：SRK/T · Hoffer Q · Holladay 1 · Haigis）/ `tilp_from_refraction`（术后光学位置，源 PEARL-DGS, MIT） | 四公式**各用各的常数、不可互换**，角膜指数也不同（1.3375 / 1.3375 / 4÷3 / 1.3315）；度数公式为**独立实现**（无可内嵌的干净实现），未对拍厂商计算器，临床决策前须复核 |
| 34 | 视力与屈光结局 | `lib.ophtha.logmar_from_snellen` / `refractive_outcome` | 6/12=0.3010；PE = actual − predicted；±0.5D 内比例是 JCRS 标准项 |
| 35 | 晶状体基因集富集 | `lib.ophtha.lens_gene_overlap` + `lib.pathway.ora` | **交集个数 ≠ 富集**，必须做超几何检验 |

### 五、预处理与细胞互作（2 个）

| # | 分析法 | 入口 | 关键口径 |
|---|---|---|---|
| 36 | **批次效应校正 ComBat** | `lib.batch.correct` / `correct_reference` / `diagnose` | EB 收缩（位置+尺度），**非**简单中心化；输入须 log 域；批次与分组共线时禁用；`diagnose` 的 separation ratio 下降**不等于**生物学保留 |
| 37 | **细胞通讯（配受体）** | `lib.cellcomm.score` / `summary` / `pathway_summary` | p 值是**置换细胞标签**的经验 p（最小 = 1/n_perm）；互作有**方向性**（ligand→receptor 不可对调）；须用归一化表达；`EXAMPLE_LR_PAIRS` 仅演示，生产自供官方表 |

### 六、扩展方法（4 个）

| # | 分析法 | 入口 | 关键口径 |
|---|---|---|---|
| 38 | **G×E 基因-环境交互** | `lib.gxe.interaction_test` / `stratified_effect` / `johnson_neyman` / `reri_additive_interaction` | **交互 ≠ 分层**（两层都显著也不代表有交互）；**相乘交互 ≠ 相加交互**（RERI 是另一个量）；连续变量做交互前**必须中心化**，否则主效应变成「另一变量 = 0 时的效应」 |
| 39 | **EWAS 甲基化差异** | `lib.epigen.diff_methylation` / `dmp_summary` | 回归建在 **M 值**上、报告在 **beta 尺度**（beta 两端方差不齐）；Δβ 与 FDR **双阈值**（大样本下 Δβ=0.01 也会「显著」）；**细胞组成是头号混杂** |
| 40 | **双向 / 两步法中介 MR** | `lib.mr_full.bidirectional` / `two_step_mediation` | 反向 MR 必须用**结局的独立 IV**（拿同一批 IV 反着算等于没算）；**双向都显著 ≠ 反向因果**（可能是水平多效性）；间接效应 CI 必须 delta 法或 bootstrap |
| 41 | **细胞类型注释** | `lib.singlecell.annotate_cells` | marker **一个都没匹配上即报错**（基因命名不一致会让打分静默变 0）；注释是**假设**不是结论，结果带 `ratio_to_runner_up` 与 `ambiguous`；打分先 z-score 再平均 |

### 开源来源速查（谁提供了算法）

| 分析法组 | 上游开源项目 | 许可 | 内嵌位置 |
|---|---|---|---|
| 孟德尔随机化全套 | pymr@maxghenis | Unlicense（公有领域） | `vendor/_mr/` + `lib/mr_full.py` |
| 富集 / GSEA / ssGSEA | GSEApy@zqfang | BSD-3-Clause | `vendor/_enrich/` + `lib/pathway.py` |
| 眼科 IOL 光学 | pearldgs_toolbox@gdebel（PEARL-DGS） | MIT | `vendor/_ophtha/` + `lib/ophtha.py` |
| 共表达网络 WGCNA | mortazavilab/PyWGCNA | MIT | `vendor/_wgcna/` + `lib/network.py` |
| 单细胞全流程 | scverse/scanpy | BSD-3-Clause | `vendor/_sc/` + `lib/singlecell.py` |
| 批次校正 ComBat | scverse/scanpy（`_combat.py`） | BSD-3-Clause | `vendor/_batch/` + `lib/batch.py` |
| 细胞通讯 | ventolab/CellphoneDB | MIT | `vendor/_cellcomm/` + `lib/cellcomm.py` |
| 其余（正则化/Meta/因果/PRS/SuSiE/空间/轨迹…） | 自研（口径对齐公开文献，无 GPL 依赖） | 本 skill | `lib/*.py` |

完整许可与「未内嵌」决策记录见 `vendor/ATTRIBUTION.md`（含一处**已更正的错误**：
scanpy 曾被误记为 GPL）。

### 方法选择的三条硬规则

1. **问题决定方法，不是数据决定方法。** 「想筛变量」→LASSO；「想保留全部变量」→Ridge；
   高维共线性组学数据 → Elastic Net。选错模型是这类分析最常见的错误。
2. **一个方法回答一个问题。** ORA/GSEA/GSVA 回答的是三个不同问题，不能互相替代。
3. **口径不随结果输出 = 没做。** power、deep_split、n_perm、prior_sd、背景基因集、
   λ 的选择方式——这些不写出来，别人无法复现。

## 眼科与白内障（重点方向）

能力在 `lib/ophtha.py`，**光学核心来自开源项目 PEARL-DGS（MIT）**，不是自造公式。

```python
from lib import ophtha as O

# ⓪ 术前 IOL 度数计算 —— 四个经典薄透镜公式横向比较（白内障最核心的那步）
r = O.r_from_K(43.0)                                   # K=43.0 D → 7.85 mm
O.iol_power_comparison(23.5, r, a_const=119.0, pacd=5.0, sf=1.5,
                       haigis_abc=(1.0, 0.4, 0.1), acd_mm=3.0)
#   典型眼 → SRK/T 21.83 · Hoffer Q 20.83 · Holladay 1 21.30 · Haigis 21.17 D
#   单公式入口：O.srkt_power / hofferq_power / holladay1_power / haigis_power
#   正算（度数→术后屈光）：同名 *_refraction
#   常数换算：O.acd_const_from_a_constant(119.0) → 5.5887
#   Haigis 常数优化：O.haigis_solve_d(24, 7.5, 21, -2) → 5.4615 mm
#
# ⚠️ 这四个公式是**按公开论文独立实现**的，不是内嵌开源件 —— 能搜到的 Python/R
#    实现要么是 AGPL-3.0，要么没有 LICENSE，**没有一个许可干净到可以内嵌**。
#    已验证：正反算闭合 / 度数随眼轴单调 / 四公式极差 0.995 D（符合文献）/
#    Haigis 对上一个公开算例。**未**与厂商在线计算器逐例对拍（那需要厂商常数）。
#    → 科研统计（反推公式表现、常数优化）可用；**临床决策前请用机构计算器复核**。

# ① IOL 光学：术后等效球镜 ⇄ IOL 理论位置（TILP）。两者互逆，往返误差 < 1e-9 D
tilp = O.tilp_from_refraction(se=-1.0, al_mm=23.5, rco1_mm=7.8, rco2_mm=6.5)   # -> mm
back = O.refraction_from_tilp(tilp, al_mm=23.5, rco1_mm=7.8, rco2_mm=6.5)      # -> -1.0 D

# ② 视力换算（国际标准定义）：6/6=0、6/12=0.3010、6/60=1.0
O.logmar_from_snellen(6, 12)      # -> 0.3010

# ③ 屈光预测误差（JCRS 标准报告项）
O.refractive_outcome(predicted_se, actual_se)
#   ME / SD / MAE / MedAE / ±0.25·0.5·1.0D 内比例；符号约定 PE = actual - predicted

# ④ 眼压描述统计（>21 mmHg 计数）
O.iop_stats(iop_list)

# ⑤ 晶状体 / 白内障基因集（仅集合运算；**富集必须走超几何检验**）
O.lens_gene_overlap(genes, "age_related_cataract_gwas")
```

基因集：`crystallin_alpha/beta/gamma`、`lens_membrane_channel`（MIP/GJA3/GJA8）、
`lens_development_tf`（PAX6/SOX2/PITX3/FOXE3/MAF/HSF4）、
`age_related_cataract_gwas`（32 个，含 EPHA2/CASP7/CRY 家族）、
`congenital_cataract_panel`（32 个）、`lens_epithelium_uv_damage`（紫外线/氧化应激）。

⚠️ **交集个数 ≠ 富集**。数出 5 个重叠基因什么也说明不了——基因集越大重叠必然越多。
富集必须做超几何检验：`vendor/_enrich/stats.py::calc_pvalues`（源 GSEApy，BSD-3）。

⚠️ **白内障遗传分析的混杂主战场**：LOXL1（chr15:73,931,230）与剥脱综合征青光眼
是**同一个变异（距离 0 kb）**——见到该位点先判共病混杂，不要当新发现。

## 数据清洗与变量挑选（`lib/prep.py`）

建模**之前**必须跑；结论错有一半是数据错（缺失被静默丢弃、离群把均值拉飞、
高相关变量重复计入）。

```python
from lib import prep
rep = prep.prep_pipeline(matrix, names=[...])
# 逐项：缺失报告 / 低方差列 / 离群（MAD）/ 相关去冗余 / VIF 共线性 / 可执行建议
```

设计约束（都是踩过的坑）：
- **不修改输入数据**，只产出报告与索引。自动改数据是最难追责的一类操作。
- **不默认丢行**。缺失是否填补由调用方显式决定（默认丢弃行是临床数据最常见的静默偏差）。
- **VIF 在相关剪枝之前算**。先剪枝再算 VIF 必然接近 1，会把"共线性严重"
  这个事实自己抹掉（实测 r=0.998 的两列剪枝后 max_vif=1.019，看着完全正常）。
- **离群三种口径的默认 k 各不相同**（iqr 1.5 / zscore 3.0 / mad 3.5），
  因为量纲不同。曾把 Tukey 的 1.5 用在 z-score 上，均匀分布两端都被判成离群（约 13%）。

## 接入开源项目：包装而非重写（wrapper + 金标准对拍）

要引入一个开源生信项目（当它许可干净、且能力确实需要时），**唯一正确的形态是包装层**，
不是把算法重写一遍：

| 错误做法 | 为什么错 |
|---|---|
| 复制 GPL 源码进 skill | 许可传染，违反本文红线 |
| 照文档重写算法 | 数值必然漂移，且无法证明一致 |
| 「大概等价就行」 | 差异会伪装成科学结论 |
| 只加个 AI 友好接口 | 上层看不见底层错误 → 静默失败 |

**包装层五条不变量**（`tools/wrapper_core.py` 已实现并被自检守护）：

1. **失败必须可见** —— 很多科研 CLI 出错也 `exit 0`。判成败要三条件：
   退出码合法 **AND** 日志无 `Error:` 行 **AND** 期望产物真的存在。
2. **产物存在是最硬的证据** —— 不依赖对日志格式的猜测。
3. **参数翻译不得静默丢参** —— 双向断言：解析器每个 flag 都要有翻译项，翻译表不能有死条目。
4. **必须有 `official` 原样透传入口** —— 这是「包装层不是功能上限」的实现方式。
   注意：argparse 即使声明了 REMAINDER **仍会**校验顶层 flag，透传会被自己拒掉，
   必须在 argparse **之前**手动分离 `official`。
5. **对不上就标红，不许放宽断言换全绿** —— 已知差异必须显式写出理由。

**已落地的样板**：`tools/calibrate_lifelines.py`（MIT，对拍 C-index）。
真实跑通 13 组（含 0%/30%/50% 三档删失 × 4 档并列强度），与 lifelines **逐位一致**（abs_diff = 0.0）。
门禁**已被证明会失败**：注入退化实现 → `overall=FAIL`、13/13 differs、exit=1。

```bash
"$PY" "$SK/tools/calibrate_lifelines.py"          # 金标准对拍（缺 lifelines → SKIP 而非假失败）
"$PY" "$SK/tools/fetch_tools.py" --fetch lifelines # 装到隔离目录，不随 skill 分发
```

## 自动触发（何时会被加载）

**本 skill 的自动触发完全由 frontmatter 的 `description` 决定** —— agent 启动时读到所有 skill 的
`name` + `description`，据此判断是否加载，**不需要任何手动点击**。

| Agent | 发现机制 | 备注 |
|---|---|---|
| **WorkBuddy** | 扫描 `~/.workbuddy/skills/` → 写入 `.skill-list-cache.json` → 注入会话 skill 索引 | 装完需**新会话**才会进入索引 |
| **Claude Code** | 扫描 `~/.claude/skills/*/SKILL.md` 的 frontmatter | 同一份包 |
| **Codex** | 扫描 `~/.codex/skills/` | 同一份包 |
| Cursor / Cline / Gemini / Copilot | 项目内规则文件（`AGENTS.md` / `.cursor/rules/*.mdc` / `.clinerules/` / `GEMINI.md` / `copilot-instructions.md`） | 由 `install_adapters.py` 生成 |

**触发可靠性是可测的**（不靠感觉）：

```bash
"$PY" "$SK/trigger_check.py" --json
# 期望：正向覆盖 12/12、误触发 0、边界声明 有
```

该检查**能失败**：首次运行时确有 1 条漏触发（「数据与官方是否一致」缺「哈希/互证」概念），已补进 description。
**改了能力范围后先跑它，再重装。**

## 跨 Agent 安装

本 skill 单一来源是 `SKILL.md`；各端入口由 `scripts/install_adapters.py` 生成，**不手写多份**。

```bash
PY="C:/Users/35680/.workbuddy/binaries/python/versions/3.13.12/python.exe"
"$PY" scripts/install_adapters.py --target all --project "<你的项目目录>" --json
```

支持 `claude`（`~/.claude/skills/`）、`codex`（`~/.codex/skills/`）、`agents`（AGENTS.md）、`cursor`、`cline`、`gemini`、`copilot`。
详见 `README.md`。

## 自检与质量门

```bash
"$PY" tests/run_all.py          # 全量自检（零依赖，纯标准库）
"$PY" scripts/doctor.py --json  # 环境与依赖报告
```

六个质量维度：功能 / 数值一致性（与权威实现对拍）/ 抗幻觉 / 抗静默失败 / 安全 / 口径纪律。
**门禁必须被证明会失败才算有效** —— `tests/test_guards_negative.py` 制造退化检测器与假数据，验证门禁会 FAIL。
