<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand-dark.svg">
  <img src="assets/brand.svg" alt="biostat-copilot" width="760">
</picture>

**面向生信与医学研究的「证据优先」工作流技能 —— 43 个主流分析法，AI 直接可跑。**

[![版本](https://img.shields.io/badge/version-1.8.0-0d9488)](VERSION)
[![许可证: MIT](https://img.shields.io/badge/License-MIT-8b5cf6.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-2563eb)](#环境要求)
[![自检](https://img.shields.io/badge/self--check-216%20%2F%20219-16a34a)](tests/run_all.py)
[![Checks](https://github.com/cndoin/med-biostat-skill/actions/workflows/checks.yml/badge.svg)](https://github.com/cndoin/med-biostat-skill/actions/workflows/checks.yml)
[![Agent](https://img.shields.io/badge/agents-Claude%20Code%20%7C%20Codex%20%7C%20WorkBuddy%20%7C%20Cursor-18181b)](#装进你的-ai-助手)

[English](README.md) · **简体中文** · [日本語](README.ja.md)

[快速开始](#快速开始) · [分析法总览](#分析法总览) · [装进你的 AI 助手](#装进你的-ai-助手) · [许可证](#许可证)

</div>

![biostat-copilot 总览](assets/banner.png)

## 这是什么

一个**跨 Agent 的科研工作流 Skill**。装进 Claude Code / Codex / WorkBuddy 等助手后，
当你提出「帮我分析这份 GWAS」「这个基因和白内障有没有因果关系」「核对一下我这组结果对不对」这类请求时，
它会被**自动加载**，然后按固定的方法学框架把分析跑完 —— 而不是让 AI 凭感觉写代码。

它解决的核心问题不是「AI 会不会写代码」，而是**口径**。

- λGC 有两套合法口径，混用就得到两个数；
- 缺失值「填 / 不填 / 怎么填」会改变所有下游结论；
- 「和官网一致」若只凭「数值差不多」判断，等于没核验。

所以本 skill 把**口径**当成一等公民：不写口径的结果一律视为无效，能算的必须由脚本算、能验的必须由哈希验。

## 三条核心主张

| # | 主张 | 含义 |
|---|---|---|
| 1 | **口径先于数值** | 同一个名字在不同口径下是不同的量。缺口径即报错，不给默认值。 |
| 2 | **一致性强于「看起来对」** | 「与官网一致」必须由哈希与字节数证明，不能由「数值差不多」推断。 |
| 3 | **混杂是主战场** | 翻车绝大多数不在算法，在混杂。偏倚层是独立一层，十二项逐条过。 |

## 分析法总览

| 组 | 数量 | 包含 |
|---|---|---|
| 因果与遗传 | 10 | 孟德尔随机化全套（7 估计量 + 7 敏感性）、药物靶点 cis-MR 链、共定位 coloc、精细定位 SuSiE、PRS、GWAS 质控、独立位点、观察性因果推断（PSM / IPTW / AIPW）、中介分析、DiD |
| 组学与通路 | 10 | 差异表达、WGCNA 共表达网络、ORA、GSEA、GSVA / ssGSEA、单细胞全流程、免疫浸润、拟时序、空间转录组、突变特征 |
| 临床统计与建模 | 11 | 生存分析、竞争风险 CIF、预测模型（校准 + DCA）、C-index / NRI / IDI、正则化回归（LASSO / Cox-LASSO）、Meta 分析、发表偏倚、多重插补 MICE、多重检验校正、混合效应、样本量与功效 |
| 数据工程与眼科 | 4 | 数据清洗与变量挑选、IOL 屈光计算、视力 / 屈光结局、晶状体基因集富集 |
| 预处理与细胞互作 | 2 | ComBat 批次效应校正、细胞通讯（配受体） |
| 扩展方法 | 4 | 基因-环境交互（G×E）、EWAS 甲基化差异、双向 / 中介 MR、细胞类型注释 |

每个分析法的入口函数、关键口径、典型错误见 [`SKILL.md`](SKILL.md) 的「分析法总表」。

## 快速开始

### 环境要求

| 项 | 要求 |
|---|---|
| Python | **3.9+**（Tier 0 只需标准库） |
| 操作系统 | Windows / macOS / Linux |
| 可选科学栈 | `numpy` + `pandas` + `scipy`（仅 Tier 1 需要） |

```bash
git clone https://github.com/cndoin/med-biostat-skill.git
cd med-biostat-skill
```

### 1. 先跑自检（不需要任何第三方依赖）

Tier 0 是纯标准库的，clone 下来即可运行：

```bash
python tests/run_all.py            # 全量自检
python scripts/capabilities.py     # 双层能力探测：本机哪层可用
python scripts/doctor.py           # 环境与依赖报告
```

### 2. 装 Tier 1 依赖（可选，装了才有完整能力）

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r vendor/requirements.txt
```

装完后 `scripts/capabilities.py` 会把 Tier 1 标为就绪。**缺依赖不会导致假失败** ——
探测不到的用例会被判为 SKIP 而不是 FAIL。

### 3. 试试自动触发自检

```bash
python scripts/trigger_check.py
```

它用 29 个正向 + 4 个负向样例，验证「哪些问法会让 skill 自动加载、哪些不该」。

## 装进你的 AI 助手

本 skill 遵循 **单一 `SKILL.md` 来源 + 生成器产出各端入口** 的原则，不手写多份入口
（避免「同一逻辑多处实现」漂移）。所有入口都由 `scripts/install_adapters.py` 生成。

```bash
# 装到个人级 skill 目录（各 agent 自动读取）
python scripts/install_adapters.py --target claude       # → ~/.claude/skills/biostat-copilot/
python scripts/install_adapters.py --target codex        # → ~/.codex/skills/biostat-copilot/
python scripts/install_adapters.py --target workbuddy    # → ~/.workbuddy/skills/biostat-copilot/

# 装到某个项目目录（项目级规则文件）
python scripts/install_adapters.py --target cursor  --project /path/to/proj   # .cursor/rules/
python scripts/install_adapters.py --target cline   --project /path/to/proj   # .clinerules/
python scripts/install_adapters.py --target gemini  --project /path/to/proj   # GEMINI.md
python scripts/install_adapters.py --target copilot --project /path/to/proj   # .github/copilot-instructions.md
python scripts/install_adapters.py --target agents  --project /path/to/proj   # AGENTS.md

# 一次全装
python scripts/install_adapters.py --target all --project /path/to/proj
```

| Agent / 工具 | 入口形态 | 安装目标 |
|---|---|---|
| Claude Code | `SKILL.md` | `claude` |
| Codex | `SKILL.md` | `codex` |
| WorkBuddy | `SKILL.md` | `workbuddy` |
| Cursor | `.cursor/rules/<name>.mdc` | `cursor` |
| Cline | `.clinerules/<name>.md` | `cline` |
| Gemini CLI | `GEMINI.md` | `gemini` |
| GitHub Copilot | `.github/copilot-instructions.md` | `copilot` |
| 通用（AGENTS.md 系） | `AGENTS.md` | `agents` |

**装完怎么确认生效？** 对助手说一句「帮我做生存分析和预测模型」，若它开始按 skill 的框架
（先定口径 → 跑脚本 → 出带口径的结果）回应，即已加载。也可在本机跑
`python scripts/trigger_check.py` 验证触发词覆盖。

> 安装器是**就地覆盖同步**（幂等、可安全重跑），不会销毁目标目录里的其他内容。

## 三层能力

| 层 | 位置 | 依赖 | 能做什么 |
|---|---|---|---|
| **Tier 0** | `scripts/` | Python 标准库 | 取回 / 核验 / 质控 / 位点计数 / 可信集过滤 / 区间注释 / 混杂审计 / 逐变异互证 / 报告门禁 |
| **Tier 1** | `vendor/` + `lib/` | numpy + pandas + scipy | 43 个分析法中的绝大多数；纯 SVG 图表；函数注册表；报告规范对表 |
| **原生加速层** | `vendor/native/` | 无（单文件可执行） | 百万级 GWAS 汇总统计流式质控（3M 行 551 ms，约 11× Python） |

`Tier 1` 的 22 个模块是**逐个探测可导入性**的：缺一个依赖不影响其余模块被判为可用。

## 算法来自哪里

**核心原则：不自己写算法。** 所有统计与光学算法都取自 GitHub 开源项目，
按「掐头去尾取核心 + 依赖换底」的方式内嵌，并在文件头标注来源、许可与已知差异。

| 上游项目 | 许可 | 提供的能力 | 内嵌位置 |
|---|---|---|---|
| [pymr](https://github.com/maxghenis/pymr) | Unlicense（公有领域） | 孟德尔随机化全套 | `vendor/_mr/` |
| [GSEApy](https://github.com/zqfang/GSEApy) | BSD-3-Clause | 富集统计量、GSEA 核心、ssGSEA | `vendor/_enrich/` |
| [scanpy](https://github.com/scverse/scanpy) | BSD-3-Clause | 单细胞全流程、ComBat 批次校正 | `vendor/_sc/` `vendor/_batch/` |
| [PyWGCNA](https://github.com/mortazavilab/PyWGCNA) | MIT | 共表达网络 WGCNA | `vendor/_wgcna/` |
| [CellPhoneDB](https://github.com/ventolab/CellphoneDB) | MIT | 细胞通讯（配受体） | `vendor/_cellcomm/` |
| [pearldgs_toolbox](https://github.com/gdebel/pearldgs_toolbox) | MIT | 眼科 IOL 光学 | `vendor/_ophtha/` |
| rsomics-vcf | MIT OR Apache-2.0 | GWAS 原生加速（源码级 fork） | `vendor/native/` |

**GPL 系项目一律不内嵌**（会传染），改走外部获取 + 独立调用：
`python tools/fetch_tools.py --fetch lifelines`。

逐件的来源、许可、哈希与「与上游的已知差异」见 [`vendor/ATTRIBUTION.md`](vendor/ATTRIBUTION.md)；
第三方许可全文随源码放在各 `vendor/_*/LICENSE`，汇总清单见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 明确不是什么

- **不是一键出图模板生成器** —— 七件套 / GEO 芯片 DEG+WGCNA+LASSO+免疫浸润 / 铁死亡铜死亡套模板，入口处直接拦截。
- **不是数值生成器** —— 任何数字必须来自可重跑的脚本，或可引用的官方发布值并标注口径。
- **不是方向推销员** —— 方向被查重判定已覆盖时，义务是**叫停**，不是换个说法继续包装。

## 目录结构

```
med-biostat-skill/
├── SKILL.md              入口总纲（路由 · 红线 · 判定门 · 43 个分析法总表）
├── VERSION
├── README.md             English
├── README.zh-CN.md       简体中文（本文件）
├── README.ja.md          日本語
├── references/           知识层（按需加载，零常驻成本）
├── scripts/              Tier0 执行层（纯标准库）+ 跨 agent 安装器
├── lib/                  Tier1 本 skill 自有扩展（22 个模块）
├── vendor/               Tier1 内嵌工具层（来源见 ATTRIBUTION.md）
├── pipeline/             确定性分析管道（AI 不参与数值）
├── tools/                vendor_sync · fetch_tools · wrapper_core · native_accel
├── assets/               品牌图 · banner · 模板
├── adapters/             各 agent 入口模板
└── tests/                自检层（功能 · 稳定 · 安全 · 门禁负向 · vendor 层 · 分析法）
```

## 自检与质量门

```bash
python tests/run_all.py            # 全量：216 / 219 通过（3 项因缺 lifelines 跳过）
python tests/run_all.py --json     # 机器可读
python scripts/capabilities.py     # 双层能力报告
python scripts/trigger_check.py    # 自动触发覆盖（29 正向 / 4 负向）
```

**门禁必须被证明会失败才算有效** —— `tests/test_guards_negative.py` 专门制造
退化检测器（恒定输出、语义反转）与假数据，验证门禁会 FAIL。
「0 失败」如果来自「门禁是摆设」，那比没有门禁更危险。

## 许可证

本项目使用 **MIT License**，见 [`LICENSE`](LICENSE)。

内嵌的第三方组件保留其原始许可（Unlicense / BSD-3-Clause / MIT）：
许可全文随源码置于各 `vendor/_*/LICENSE`，汇总清单见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)，
来源与哈希的完整记录见 [`vendor/ATTRIBUTION.md`](vendor/ATTRIBUTION.md)。
