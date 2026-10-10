# 开源项目评估记录（opensource）

**结论先行**：本 skill 核查了设计文档 §11.5 的开源清单，**没有把任何开源项目的源码内嵌进来**——
不是因为不想，而是因为**许可与依赖两条路都走不通**。本文件是那次核查的可审计记录。

如果日后要重新评估，**先读这里**，避免重复劳动或误判。

---

## 一、为什么不能内嵌源码

| 障碍 | 说明 |
|---|---|
| **许可传染** | 生信工具生态由 GPL/LGPL/AGPL 主导。这些许可要求衍生作品同样以同许可发布，**并入本 skill 分发即触发传染**（设计文档 §11.4 红线）。 |
| **依赖过重** | 许可干净的多是重型框架（pysam 需 C 扩展；scanpy/PyDESeq2 依赖 anndata+scikit-learn 等），搬运的成本与风险高于收益。 |
| **版本漂移** | 内嵌副本会与上游脱节。本 skill 已有 `tools/vendor_sync.py` 机制，但那只适合**自建**工具（源在本地可控）。 |

**正确的替代做法**（已实现）：
- 许可干净的 → 通过 `vendor/requirements.txt` 声明为**依赖**，`pip install` 即可；
- GPL / 重型 / 需外部二进制的 → 通过 `tools/fetch_tools.py` 获取到**隔离目录**，只作对拍，不随 skill 分发；
- 方法学 → 只借思路，**自己实现**（本 skill 的 `scripts/` 与 `vendor/` 均已如此）。

---

## 二、本次联网核实（2026-10-10）

### GWASLab —— 许可"不一致"已结案

设计文档 v2.0 记录「PyPI 标 MIT / GitHub 标 GPL-3.0，须核实」。**本次核实结论：不是标注冲突，是版本切换。**

| 证据 | 内容 |
|---|---|
| GitHub 仓库文件 | 同时存在 `LICENSE` 与 `LICENSE_before_v3.4.39` 两个文件 → v3.4.39 是许可切换点 |
| PyPI `gwaslab==3.3.4` | License: **MIT** |
| PyPI `gwaslab==4.1.8` | License: **GPL-3.0-only** |
| bioconda `gwaslab 4.1.6` | License: **GPL-3.0-only** |
| 仓库内 `mkdocs.yml` | copyright 仍写 "MIT license" → **残留标注，与主许可矛盾，不可采信** |

**处置**：维持 **B 档**（对拍，不打包、不分发）。许可确认方法学结论：`GPL-3.0-only`（v3.4.39 起）。
依赖极重（pandas+scipy+pysam+scikit-allel+pyensembl+gtfparse+matplotlib+seaborn+adjustText），
即便回退到 MIT 的 v3.3.x 也不适合内嵌。

### GWASPipe —— MIT，但不可内嵌

| 项 | 内容 |
|---|---|
| 许可 | **MIT**（官方 README 声明） |
| 障碍 | 核心依赖是 `gwaslab`（GPL-3.0）→ **通过依赖链传染**，内嵌它等于间接引入 GPL |
| 处置 | 不内嵌；方法学（等位基因排序 `order_alleles`）可参考 |

---

## 三、其余项目（设计文档记录，本次未逐一复核）

> **标注"未核实"者，采用前必须从官方仓库确认**——SOP 不允许"大概是 MIT 吧"。

### 不可内嵌（GPL / LGPL / AGPL 系）

| 项目 | 许可 | 用途 | 档位 |
|---|---|---|---|
| PLINK 2 | GPL-3.0 | 独立位点的行业定义（`--clump`） | B |
| ldsc | GPL-3.0 | 遗传力 / 遗传相关 / LDSC-SEG 方法学来源 | B |
| coloc | GPL-3.0（未核实） | 共定位 PP.H4 | B |
| TwoSampleMR | MIT（未核实，R 包） | MR 对拍 | B |
| MR-PRESSO | 未核实（GPL 系？） | 多效性检测 | B |
| susieR / FINEMAP | 未核实 | 精细定位对拍 | B |
| FUSION / S-PrediXcan | GPL-3.0（未核实） | TWAS 对拍 | B |
| DESeq2 | LGPL-3.0 | 差异表达对拍 | B |
| edgeR / limma | GPL-2.0 | 差异表达对照 | B |
| WGCNA | GPL-2.0（未核实） | 共表达网络 | C |
| clusterProfiler | Artistic-2.0（未核实） | 通路富集（本机已有对应 skill） | A/C |
| DEPICT | 未核实 | **D 档：不采纳**（1e-5 阈值假阳性失控） | D |

### 许可干净，但以**依赖**方式使用（不内嵌源码）

| 项目 | 许可 | 用途 |
|---|---|---|
| numpy / pandas / scipy / statsmodels | BSD-3 | Tier1 引擎基础 |
| lifelines | MIT | 生存分析对拍 |
| scikit-learn | BSD-3 | 建模对照（本 skill 的 `vendor/engines/modeling.py` 不依赖它） |
| scanpy / anndata | BSD-3 | 单细胞（本机 `seurat-sc` skill 覆盖） |
| PyDESeq2 | MIT | 差异表达（与 DESeq2 对拍） |
| pysam | MIT | htslib 绑定（需 C 扩展） |
| tableone | MIT | 基线表对拍 |
| pytest / hypothesis | MIT / MPL-2.0 | 测试（本 skill 内置 `tests/run_all.py` 已零依赖覆盖） |
| polars / duckdb / pyarrow | MIT / MIT / Apache-2.0 | 列式加速（Tier1 可选） |

---

## 四、本 skill 最终的三条取用路径

```
① 自建工具内嵌         vendor/            ← tools/vendor_sync.py 同步，见 ../ATTRIBUTION.md
② 许可干净的依赖        vendor/requirements.txt → pip install
③ GPL / 重型 / 外部二进制 tools/fetch_tools.py   → 装到 ~/.biostat-copilot/tools/（不随 skill 分发）
④ 方法学               references/tools.md 的四档表 + 自实现代码
```
