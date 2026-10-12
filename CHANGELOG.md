# Changelog

本项目的所有重要变更都记录在此。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [1.10.0] — 2026-10-12

### 新增 · 多组学整合与保真 5 个（分析法总数 48 → **53**）

> 上一版补齐了 P0（不补就断链）；这一版补齐 **P1（补了才扛得住审稿）**：
> 整合缺失会让「供体 == 批次」的假象贯穿全部下游分析，而整合过度会静默抹掉
> 年龄信号 —— 两者都不报错。所以这一版的主题是 **先整合，再实测有没有整过头**。

- **`lib/integration.py` —— Harmony 批次整合 + 保真诊断**（含 `vendor/_harmony/core.py`）。
  源 `lilab-bcb/harmony-pytorch`（**BSD-3-Clause**；`slowkow/harmonypy` 是 GPL-3、
  `immunogenomics/harmony` 是 NOASSERTION，均已排除）。算法逐句照抄、**底座换掉**：
  torch → numpy、`sklearn.KMeans` → 自实现 k-means++ + Lloyd，去 psutil/threadpoolctl。
- **`lib/nullcontrol.py` —— 随机匹配基因集零分布（阴性对照唯一裁决点）**。
  回答「换成随便一组基因会不会一样显著」：按表达水平的**分位数分箱**等配额无放回抽样，
  箱内候选不足时向相邻层外扩并把 `max_expansion` 报出来，**不**悄悄有放回凑数。
- **`lib/paga.py` —— PAGA 簇间连通性**：`score = log(观测连边/期望连边)`，
  期望按有向 kNN 图推（`k·n_a·(n_b−δ)/(N−1)`，行和 = `k·n_a`，测试里用这条恒等式断言自洽）。
- **`lib/figs.py` —— Fig3 失守曲线图**：纯 SVG、确定性（同输入同一个字节，可进 git diff）、
  无 matplotlib；**`detected=False` 时不画失守竖线**。

### 修复（都是自检查出来的真 bug，不是文档改动）

1. **`fidelity` 把信号最强判成没信号**（最隐蔽的一个）。
   协变量被 embedding 完美解释时 R²=1 → F=∞ → p=NaN，而 informative 判据是
   `p < α`，于是 **R²=1 被判成「整合前就没有信息」、retention 直接失去定义**。
   方向完全反了，而且只在信号极强时出现。现改为 p=0（极端显著）。
2. **Harmony `correction` 用了正在被修改的 Z 去算系数矩阵**（上游用的是原始 X），
   等于按累计步长多次校正、会过冲。修为：系数归原始 X、累加归 Z。
3. **Harmony 的早停会被目标上升误触发**：各轮目标算在不同的 embedding 上、跨轮不可比。
   实测 2 批次大偏移下默认参数**只跑 1 轮**就收工（mixedness 0.891→0.777，没整干净），
   跑满 10 轮才到 0.568 且生物团纯度反而更高。核心件保留上游默认但报 `stopped_by`，
   **包装层默认 `stop_on_increase=False`**。
4. **默认 θ=2.0 在大偏移下不够**：三批次实测 batch_mixing 的 normalized 只从 2.52
   降到 1.49，θ=3 收敛到 1.18、θ≥4 饱和。结论写进 docstring 与测试：**别信默认值，调完实测**。
5. `exp(-2/σ(1-cos))` 在 σ 很小时整行下溢为 0 → 行 L1 归一化变成 0/0 扩散 NaN。
   改为先减行最大值（数学等价，因为随后是逐元素乘 ω + 行归一）。
6. 单细胞批次（每批仅 1 个细胞）不再静默给 garbage：直接报错。

### 关于「未证实的部分」（如实标注，不得省略）

- `_harmony` **未与 torch 原版画面对拍**（本机无 torch），只做性质比对；
- PAGA 用的是本仓库的自适应高斯核 connectivities 而非 umap 模糊单纯形集，
  **绝对值不可与 scanpy `tl.paga` 逐位对比**，只能比拓扑；
- Harmony 不保证留住任何协变量信号 —— 这事只能由 `fidelity` 实测。

## [1.9.0] — 2026-10-12

### 新增 · 供体层级与转化出口 5 个（分析法总数 43 → **48**）

> 起因：对《白内障遗传风险「细胞类型 × 年龄 × 可成药图谱」研究框架》做
> **工具满足度核查**（41 个可核项：21 项工具 + 9 类验证 + 6 道判定门 + 5 项口径），
> 结论是 S4 因果链完全就绪、S0 判决可跑，但 **S2 遗传落点 / S3 蛋白组正交 /
> S5 药物映射三块硬缺口**，分别卡住 Fig2 / Fig4 / Fig6。本版补齐这五件。
> 核查报告见项目根目录 `研究框架_工具满足度核查_20261012.md`。

- **`lib/pseudobulk.py` —— 供体层级聚合**（防伪重复的第一道闸门）。
  细胞不是独立样本，直接拿细胞当 n 会把样本量虚增两三个数量级。
  默认 `sum_counts`（先加 raw counts 再归一化），`mean_log` 标注为有偏口径。
- **`lib/specificity.py` —— 细胞类型特异性（唯一裁决点）**。
  ⚠️ 核查发现 **CELLEX 与 CELLECT 均为 GPL-3.0**（此前一度被误记为 MIT），
  按红线**不能内嵌也不能照抄**，故按公开描述的特异性度量思路**独立实现**，
  数值不可与 CELLECT 官方产物对比，只能与论文定性结论比对。
- **`lib/proteomics.py` —— 蛋白组差异 + RNA-蛋白断层（唯一裁决点）**。
  limma 与 MSstats 均为 GPL，不能内嵌；检验走 `lib.lm` 的 moderated t，
  与 EWAS **共用同一引擎**，保证两侧 p 值可比（"是否同向"依赖这件事）。
- **`lib/composition.py` + `vendor/_sccoda/core.py` —— 供体层级差异丰度**。
  模型来自 theislab/scCODA（**BSD-3-Clause**），上游依赖 tensorflow + tfp + arviz
  不能内嵌，故**推断引擎换底**：HMC 后验 → L1 选变量（λ 由 BIC 网格选）
  + 选中支持集上无惩罚 MLE + 供体 bootstrap。差异已在该文件头如实标注。
  ⚠️ miloR 是 GPL-3，**未覆盖** → 邻域级局部差异检验做不了，须写进局限。
- **`lib/druggable.py` —— 可成药映射与干预窗口**（转化出口）。
  只吃本地已落盘药库（网络 API 不可复现，不进主链路）；必报
  `pct_approved_drug_target`；给药途径按**眼局部 / 全身**分层计数。

### 重构 · moderated t 收敛为唯一裁决点（P21）

- 新增 **`lib/lm.py`**：`lib.epigen`（EWAS）与 `lib.proteomics`（蛋白组）原先
  各自需要一套"OLS + Smyth 2004 fitFDist 方差收缩"。若各写一份，
  两处的 d0 / 自由度 / BH 校正必然漂移，同一个基因在两个组学里会得到
  **不可比的 p 值** —— 而"RNA-蛋白是否同向"恰恰依赖两侧 p 值可比。
  现统一委托 `lib.lm.moderated_lm` / `lib.lm.bh_adjust`。

### 修复 · 真实 bug

1. **特异性度量的单调性反了**（新模块自检时发现，未流入发布版）：
   初版基线写成"以表达量自身为权重"（ΣE²/ΣE），结果**只在一种细胞表达**的基因
   基线正好等于最大值 → ES=0，而 [10,1,1,1] 反而得 +0.34，**越特异分越低**。
   改为 `share` 口径（log2 份额富集）后单调性正确，并补上方向性测试断言。
2. **`int(dof)` 在完全收缩时抛 OverflowError**：d0 可以是 `inf`
   （各特征方差真的一致时稳定触发，不是边角情况），`int(inf)` 崩溃。
   现 `dof` 报 float。
3. **L1 惩罚把效应量系统性拉向 0**：实测真值 0.35 被估成 0.325。
   改为**选择归 L1、估计归无惩罚 MLE**（选完变量后在支持集上重拟合），
   估计回到 0.337，且 bootstrap CI 覆盖真值。
4. **scCODA 移植件的整数计数校验只查行合计**：`Y*1.5`（合计恰好为整数）
   能蒙混过关，而 DM 的 `gammaln(y+1)` 对非整数也有定义 —— **喂归一化过的
   假计数不会报错，只会静默给出错误结果**。现逐格校验。
5. **bootstrap 的 B 下限从 20 抬到 100**：实测零假设下 B=40 会冒假阳性
   （bootstrap p 的分辨率是 2/B），B=200 时 48 次零假设检验 0 假阳性。
6. **参考细胞类型的 p 值被报成 0**：参考行系数恒为 0（可辨识性约束），
   `2·min(frac_pos, frac_neg)` 算出 0.0，会被误读成"参考类型显著变化"。
   现置 NaN。
7. **干预窗口表把实验药标成已上市**：(基因 × 药物) 行用的是基因级
   `is_approved_target`，一个基因只要有任一上市药，它名下的实验药全被标 True。
   现改为**药物级**标志。

## [1.8.0] — 2026-10-11

### 新增 · 眼科 · 经典 IOL 度数计算

- **白内障术前最核心的一步**：`lib/ophtha.py` 新增 **SRK/T**、**Hoffer Q**、
  **Holladay 1**、**Haigis** 四个第三代薄透镜公式 —— 由眼轴 + 角膜曲率
  （+ 前房深度）算 IOL 度数，并做**多公式横向比较**。
  另含 A 常数 ⇄ ACD 常数换算、Haigis 的 d 反解（IOL 常数优化用）。
  按**公开发表论文独立实现**——可搜到的实现许可全部不合格（AGPL-3.0 或无 LICENSE），
  详见 [`vendor/ATTRIBUTION.md`](vendor/ATTRIBUTION.md) 的「按公开文献独立实现」一节。
  ⚠️ 验证到"正反算闭合 / 眼轴单调 / 四公式极差符合文献 / Haigis 对上公开算例"；
  **未**与厂商在线计算器逐例对拍，**临床决策前须用机构计算器复核**。

### 新增 · 扩展方法 4 个（分析法总数 37 → 41）

- `lib/gxe.py` —— **G×E 基因-环境交互**：交互项检验、分层效应、Johnson–Neyman 区间、
  相加交互 RERI / AP / S（Rothman）
- `lib/epigen.py` —— **EWAS 甲基化差异**：beta ⇄ M 值转换、limma 式 moderated t
  （按 Smyth 2004 论文实现，limma 本身是 LGPL 不内嵌）、DMP 汇总、细胞组成粗代理
- `lib/mr_full.py` —— **双向 MR** 与 **两步法中介 MR**（间接效应 delta 法；
  反向分析必须用结局的独立 IV）
- `lib/singlecell.py::annotate_cells` —— **细胞类型注释**（marker z-score 打分 +
  不确定度标注；marker 全不匹配即报错）

### 新增 · 跨龄单细胞方向的三件地基（分析法总数 41 → **43**）

> 起因：为「跨龄晶状体单细胞 × 遗传风险定位」方向做工具清单对照核查，
> 发现三件**卡住判决性预检**的缺口（表达率统计 / 年龄失守点 / 双胞体检测）。
> 缺口不补，预检判据「≥60% 风险基因可检出、≥10 个呈年龄趋势」根本算不出来。

- **`lib/age_onset.py` —— 年龄失守点（全项目唯一裁决点）**。
  「某个基因几岁开始失守」这个数**只允许由本模块产生**，别处不得另写一份实现
  （否则同一份数据会给出两个年龄）。两种口径并存、默认 `ci_departure`：
  · `ci_departure` —— 最早使 **95% CI 上界 < 基线**且此后**持续**成立的年龄；
  · `relative_drop` —— 最早使 μ(t) ≤ 基线×(1−f) 且此后持续成立的年龄。
  **红线**：趋势不显著（p ≥ alpha）时**一律返回 `onset_age=None`**
  ——报一个年龄等于把噪声当发现；常数表达标 `degenerate`，
  不给「零宽置信带」造成的假显著。
  算法为 `vendor/_gam/core.py`（移植自 **dswah/pyGAM，Apache-2.0**：
  `b_spline_basis` 的 De Boor 递推 / `derivative` P 样条惩罚 /
  `_pirls` 的 QR+SVD 稳定求解 / `edof`·`scale` 估计 / `_estimate_GCV_UBRE` 的 GCV）。
  实测：真值 45 岁失守 → 检出 45.7；纯噪声 → `None`；上传升趋势 → `up`。

- **`lib/scrublet.py` + `vendor/_scrublet/core.py` —— 双胞体检测**
  （源 **swolock/scrublet，MIT**；自动阈值取自 scikit-image 的 `threshold_minimum`，BSD-3）。
  **双胞体在轨迹推断里会伪装成「过渡态细胞」，不清掉直接污染伪时间与失守曲线**。
  支持**按供体分批**（不同样本捕获效率不同，混检会把样本差异当双胞体信号）。
  实测（人工构造 ground truth）：整块 **AUC 0.976**、分批 **AUC 0.918**。

- **`lib/singlecell.expression_rate` / `detectable_genes` —— 表达率统计**。
  预检判据「≥N% 风险基因可检出」中**「可检出」必须由表达率定义**，不能凭印象。
  口径随结果输出：检出阈值（默认 0 = 非零即检出）、分母（全组细胞，不丢行不下采样）、
  `mean_expr`（含阴性）与 `mean_expr_positive`（仅阳性）**两者都给**。

### 修复（本轮实测发现）

- **`lib.age_onset` 趋势不显著时仍给出失守点**：纯噪声数据会因局部波动在某个年龄
  「CI 偏离基线」，据此报出的 onset 会被下游当成发现。→ 加 `alpha` 门槛，
  不显著一律 `None`（实测修复前噪声数据报 28.3 岁，修复后 `None`）。
- **`lib.age_onset` 基线用单点拟合值 → onset 系统性偏早**：样条在端点方差最大，
  单点基线受端点效应影响。→ 基线改为「年龄网格最年轻的前 10% 段的拟合值均值」，
  实测把真值 40 岁的失守点从 32.3 岁拉回 34.7 岁，真值 45 岁的场景检出 45.7 岁。
- **`lib.scrublet` 基因过滤后剩太少仍静默继续**：v-score 筛的是「高于泊松噪声的
  变异」，数据几乎无过散布时会把真实信号基因一并滤掉。实测某次实验 400 个基因
  只剩 32 个背景基因，PCA 建在噪声上、双胞体分数**看起来仍在 0–1 之间**却毫无判别力
  （AUC 从 >0.9 掉到 0.68）。→ 加 `min_genes_after_filter` 门禁，不足即**报错**。

### 工程

- `vendor/ATTRIBUTION.md` 补登两个手工移植件（`_scrublet` MIT / `_gam` Apache-2.0）
  及「许可不合格故未内嵌」的记录（`miloR`、`ldsc` 均 GPL-3.0 → 差异丰度检验仍为缺口）。
- `tests/test_methods.py` 新增 3 个测试类 15 个用例（含定向数值、口径敏感性、负向用例）。

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
