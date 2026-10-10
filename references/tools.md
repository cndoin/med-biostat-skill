# 开源工具采纳记录（tools）

**四档采纳策略。** 每个被采纳的项目都要记录：名称、版本、许可、用途、是否可作为依赖分发、对拍脚本路径。

| 档 | 定义 | 判断标准 | 运行时依赖 |
|---|---|---|---|
| **A 直接依赖** | 装起来就用，其输出直接进入结果 | 成熟、边界清晰、许可干净，且我们不想自己写错 | 是 |
| **B 对拍金标准** | 不作为依赖，但用它的输出来验证我们的实现 | 领域内有权威性，但我们不需要它的运行时 | 否（只在测试时） |
| **C 只借方法学** | 读它的方法与参数，自己实现 | 思想好，但依赖重 / 语言不通 / 与本机环境冲突 | 否 |
| **D 明确不采纳** | 写进红线，禁止使用 | 已被审稿人识别为套路，或方法学上有已知缺陷 | — |

**为什么必须分档**：每个依赖都是长期负债（版本漂移、许可传染、跨平台安装）。
**B 档性价比最高** —— 不引入运行时依赖，却能回答「我怎么知道我的实现是对的」。
**C 档解决方法学落后**（最典型：富集方法）。

## 逐层采纳清单

| 层 | A 直接依赖 | B 对拍金标准 | C 只借方法学 |
|---|---|---|---|
| S1 数据 | pyarrow/Parquet；pysam/htslib（tabix） | — | aria2c 分片续传；Snakemake 增量重跑；DVC 数据版本化 |
| S2 质控 | pandera/pydantic（数据契约） | **GWASLab**（build/等位基因/sanity）；**munge_sumstats**；**PLINK2 --clump**；**ldsc** | EasyQC 的 QC 检查清单 |
| S3 遗传与因果 | — | susieR/FINEMAP；**TwoSampleMR + MR-PRESSO**；**coloc/hyprcoloc** | PolyFun 注释加权；GCTA-COJO 条件分析 |
| S3 组学整合 | scanpy/Seurat/anndata | ldsc（遗传相关） | **LDSC-SEG**；sc-linker τ*；scDRS；**Cauchy 组合** |
| S3 临床统计 | lifelines/statsmodels/sklearn/tableone | R survival/rms | 校准曲线与 DCA 结构 |
| S3 差异与通路 | PyDESeq2 | PyDESeq2 ↔ DESeq2 | clusterProfiler 报告结构 |
| S5 验证 | — | ldsc、coloc | — |
| S6 产出 | 文档流水线（MD→HTML→DOCX） | — | gwaslab 的 Miami/regional plot；Quarto |
| S7 工程 | **polars/duckdb/pyarrow**；**pytest+hypothesis**；ruff/mypy | — | — |

## 许可与合规红线

> **许可信息以各项目官方仓库为准。** 下表标注「未核实」的，采用前必须逐一查证 —— SOP 上不允许「大概是 MIT 吧」。

- **GWASLab 许可不一致 —— 已结案（2026-10-10 联网核实）**：**不是标注冲突，是版本切换**。
  v3.3.x 及以前 = **MIT**；**v3.4.39 起 = GPL-3.0-only**（仓库内 `LICENSE` 与 `LICENSE_before_v3.4.39` 并存即切换点；
  PyPI 3.3.4 标 MIT、4.1.8 标 GPL-3.0-only、bioconda 标 GPL-3.0-only；`mkdocs.yml` 里的 "MIT" 是残留标注）。
  **处置**：维持 **B 档**（对拍，不打包、不分发）。依赖亦极重（pandas+scipy+pysam+scikit-allel+pyensembl+matplotlib…）。
  完整核实记录见 `vendor/opensource/README.md`。
- **GWASPipe 是 MIT，但仍不可内嵌**：其核心依赖是 `gwaslab`（GPL-3.0）→ 通过依赖链传染。
- **GPL / LGPL / AGPL 系**（coloc、ldsc、PLINK2、FUSION、S-PrediXcan、DESeq2、edgeR、limma、WGCNA、scikit-survival 等）：只作 B 档对拍或独立调用，**不得把其代码并入本 skill 分发**。
- **许可干净的优选**（MIT / BSD / Apache-2.0）：polars、duckdb、pyarrow、pytest、hypothesis（MPL-2.0）、pandera、pydantic、lifelines、statsmodels、scikit-learn、scanpy、Seurat、anndata、PyDESeq2、snakemake。

## D 档（明确不采纳，写进红线）

| 项目 | 不采纳原因 |
|---|---|
| 一键式 Web 平台（无参数控制的黑箱） | 无法追溯口径、无法做敏感性 |
| 模板化「七件套」 | 已进入审稿人识别清单 |
| **无协变量的 MAGMA** | 零富集模拟中假阳性失控 |
| **DEPICT 的 1e-5 阈值** | 同上，固定阈值法不可靠 |
| 只报 AUROC 的预测建模流程 | 违反模型三件套 |

## 本 skill 自身的依赖与内嵌

**三层取用路径**（详见 `vendor/opensource/README.md`）：

| 路径 | 位置 | 说明 |
|---|---|---|
| ① 自建工具内嵌 | `vendor/` | 由 `tools/vendor_sync.py` 同步（来源/许可/哈希见 `vendor/ATTRIBUTION.md`） |
| ② 许可干净的依赖 | `vendor/requirements.txt` | `pip install -r` 即可 |
| ③ GPL / 重型 / 外部二进制 | `tools/fetch_tools.py` | 装到 `~/.biostat-copilot/tools/`，**只作对拍、不随 skill 分发** |

**Tier 0 核心引擎纯 Python 标准库**（可跨 agent 裸环境运行）。
Tier 1 内嵌引擎需以下依赖，有则启用、无则回退：

| 包 | 用途 | 缺失时的回退 |
|---|---|---|
| numpy / pandas / scipy | Tier1 引擎基础 | Tier1 不可用，Tier0 全部照常 |
| statsmodels | Cox 比例风险（PHReg）、校准斜率 | 生存/建模对应项降级 |
| duckdb / polars | 列式扫描（Parquet） | 标准库 csv + 分块 |
| pytest / hypothesis | 属性化测试 | 内置 `tests/run_all.py`（unittest） |
| matplotlib / scikit-learn | **不需要** | 本 skill 自实现（`vendor/viz/svg_charts.py` / `vendor/engines/modeling.py`） |

详见 `scripts/capabilities.py --json` 的能力报告与 `scripts/doctor.py --json` 的依赖报告。
