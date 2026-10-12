# 第三方组件声明（Third-Party Notices）

本项目的自有代码使用 **MIT License**（见 [`LICENSE`](LICENSE)）。

`vendor/` 目录下包含若干**第三方开源组件**。它们保留各自的原始许可，
许可全文随源码放在对应的 `vendor/_*/LICENSE` 文件中。按「算法照抄、依赖换底」
方式移植的文件，文件头均标注 `PORTED-FILE` + 来源 + 许可 + 与上游的已知差异。

| 组件 | 上游项目 | 许可 | 内嵌位置 | 许可全文 |
|---|---|---|---|---|
| pymr | [maxghenis/pymr](https://github.com/maxghenis/pymr) | Unlicense（公有领域） | `vendor/_mr/` | [`_mr/LICENSE`](vendor/_mr/LICENSE) |
| GSEApy | [zqfang/GSEApy](https://github.com/zqfang/GSEApy) | BSD-3-Clause | `vendor/_enrich/` | [`_enrich/LICENSE`](vendor/_enrich/LICENSE) |
| scanpy | [scverse/scanpy](https://github.com/scverse/scanpy) | BSD-3-Clause | `vendor/_sc/` | [`_sc/LICENSE`](vendor/_sc/LICENSE) |
| scanpy（ComBat 部分） | [scverse/scanpy](https://github.com/scverse/scanpy) | BSD-3-Clause | `vendor/_batch/` | [`_batch/LICENSE`](vendor/_batch/LICENSE) |
| PyWGCNA | [mortazavilab/PyWGCNA](https://github.com/mortazavilab/PyWGCNA) | MIT | `vendor/_wgcna/` | [`_wgcna/LICENSE`](vendor/_wgcna/LICENSE) |
| CellPhoneDB | [ventolab/CellphoneDB](https://github.com/ventolab/CellphoneDB) | MIT | `vendor/_cellcomm/` | [`_cellcomm/LICENSE`](vendor/_cellcomm/LICENSE) |
| pearldgs_toolbox | [gdebel/pearldgs_toolbox](https://github.com/gdebel/pearldgs_toolbox) | MIT | `vendor/_ophtha/` | [`_ophtha/LICENSE`](vendor/_ophtha/LICENSE) |
| rsomics-vcf | rsomics（源码级 fork） | MIT OR Apache-2.0 | `vendor/native/` | 见上游仓库 |
| scrublet | [swolock/scrublet](https://github.com/swolock/scrublet) | MIT（自动阈值取自 scikit-image，BSD-3） | `vendor/_scrublet/` | 见上游仓库 |
| pyGAM | [dswah/pyGAM](https://github.com/dswah/pyGAM) | Apache-2.0 | `vendor/_gam/` | 见上游仓库 |
| scCODA | [theislab/scCODA](https://github.com/theislab/scCODA) | BSD-3-Clause | `vendor/_sccoda/` | [`_sccoda/LICENSE`](vendor/_sccoda/LICENSE) |
| **harmony-pytorch** | [lilab-bcb/harmony-pytorch](https://github.com/lilab-bcb/harmony-pytorch) | BSD-3-Clause | `vendor/_harmony/` | [`_harmony/LICENSE`](vendor/_harmony/LICENSE) |

**harmony-pytorch 的移植说明**（与上一个 AS IS 许可证系列一致，附 Article 3 义务）：
本 skill 取用其 `harmony/harmony.py` 的**算法结构**（模糊 k-means + 多样性惩罚 +
岭回归校正的交替迭代），**底座已替换**：torch.Tensor → numpy，`sklearn.cluster.KMeans`
→ 自实现 k-means++ + Lloyd，并去掉 psutil / threadpoolctl 这两个纯性能设施
（它们不参与任何数值计算）。许可文件为上游 BSD-3-Clause 全文，版权归
The Broad Institute, Inc.、The General Hospital Corporation 与 Genentech, Inc.。
**未做的一件事（如实标注）**：未与 torch 原始实现做逐位对拍（本机无 torch），只做性质比对。

## 未内嵌的项目（GPL 等）

为避免许可传染，以下项目**不内嵌源码**，仅在需要时由 `tools/fetch_tools.py`
从官方源获取后独立调用，不随本仓库分发：

| 项目 | 许可 | 处理方式 |
|---|---|---|
| GWASLab | GPL-3.0 | 等价能力自研（`scripts/qc_gwas.py` + 原生 `sumstats`） |
| py-twosamplemr | GPL-3.0 | 改用 pymr（Unlicense） |
| Seurat | GPL-3 | 改用 scanpy（BSD-3）算法移植 |
| PLINK2 / ldsc / coloc(R) | GPL | `tools/fetch_tools.py` 外部获取，独立调用 |

## 说明

- 每个移植件与上游的**已知差异**（例如「聚类用 Louvain 而非 Leiden」、
  「无 complex 多亚基展开」）都在该文件头的 PORTED 注释里逐一标注。
- 来源、哈希与去重决策的完整记录见 [`vendor/ATTRIBUTION.md`](vendor/ATTRIBUTION.md)。
- 其中包含一处**已更正的错误记录**：scanpy 曾被误记为 GPL，据此把「单细胞」
  整块能力排除在包外；经核对 GitHub 官方 SPDX 字段，scanpy 实为 **BSD-3-Clause**
  （Seurat 才是 GPL-3）。
