# 质控口径定义（qc_definitions）

**铁律：口径先于数值。** 每个指标在输出前必须先回答「定义是什么、在哪里定义」。
**缺口径即报错，不给默认值。** 脚本输出每条指标都必须携带 `caliber` 字段。

## 口径速查表

| 指标 | 必须声明的口径 | 常见误用 | 本 skill 的字段名 |
|---|---|---|---|
| λGC | 计算所用的变异集合与 **MAF 分档** | 把局部值当全基因组值；不报分档 | `lambda_gc.caliber` |
| 全基因组显著 | 阈值（常用 5×10⁻⁸）+ **单位是变异还是位点** | 两者混用 | `gw_significant.unit` |
| 独立位点 | **聚类方法**（LD / 距离）+ 窗口 / r² 阈值 | 报数不报口径 | `loci.cluster_method` |
| MAF | **次等位频率**还是 **ALT 频率** | 跨文件直接相减 | `maf.definition` |
| 效应等位 | EA 是否等于 ALT | 不声明，导致方向可疑 | `effect_allele.is_alt` |
| 可信集 | 精度指标（`cs_avg_r2` / `cs_min_r2`）+ **纯度过滤** | 不过滤低纯度集 | `credible_set.filter` |
| 组织特异性 | 指标（表达广度 / τ 指数）+ 阈值 | 单指标 + 无阈值敏感性 | `tissue_specificity.metric` |
| 富集 P | 检验方法与置换次数（或 LD 分数回归模型） | 只报名义 P | `enrichment.model` |
| q 值（蛋白组） | **LEVEL** + **CONTEXT** | 跨样本直接用未过滤 q 值 | `qvalue.level_context` |
| PTM 位点 | Delta-mass FDR + A-Score 阈值 | 不做特异 FDR | `ptm.delta_mass_fdr` |
| 生存结局 | 事件定义 + 随访窗口 + 竞争风险处理 | 忽略竞争风险 | `survival.outcome_def` |
| 预测模型 | 区分度 + 校准度 + 决策曲线 + 验证方式 | 只报 AUROC | `model.three_piece` |
| 版本 | 数据版本号 + 获取日期 + 哈希 | 只写「FinnGen 数据」 | `provenance.*` |

## GWAS 类

| 项目 | 要求 |
|---|---|
| 病例 / 对照数 | **三处来源交叉核对**（manifest / 质控表 / API） |
| λGC | 必须同报口径；提供方只给一个数而无分档说明时，标注**「口径未公开，不可与他处直接互比」** |
| QQ 与分位点 | 保留分位点表（供作图与复算） |
| MAF / INFO / MAC / HWE | 报过滤阈值，并做**换阈值的敏感性** |
| 变异数 ≠ 独立位点数 | 两个数都报；独立位点数须说明聚类口径 |
| 表型与对照定义 | 编码体系、排除标准、最小年龄 / 最小随访 |
| 技术性健全性 | 等位基因频率越界、sebeta 非正、p 越界、重复变异、病例-对照 AF 差 |

## 表达组类

文库与批次、离群样本、标准化方法、多重检验、探针 / 基因 ID 版本对齐。

## 蛋白组类

- q 值的 **LEVEL**（前体 / 肽段 / 蛋白组）与 **CONTEXT**（单针 / 实验级 / 全局）**必须声明**
- PTM 开放搜索须做 **Delta-mass 特异 FDR** 与 **A-Score > 0.99**
- 脱酰胺等「既是制备伪影又是真实生物学」的修饰需**三道防线**

## 临床类

缺失机制、变量字典、结局事件定义与随访窗口、竞争风险。

## 口径缺口的处置

脚本遇到缺口径的输入：

- `--strict`（默认）：**报错退出**（exit code 3），错误信息写 stderr，JSON 里 `ok=false` + `missing_caliber=[...]`。
- `--allow-missing-caliber`：允许继续，但每条结果强制写 `caliber="UNSPECIFIED"`，并在 `logs/` 记一条 `caliber_gap` 警告。
- **禁止**静默给默认值。
