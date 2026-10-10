# 贡献指南

感谢考虑为本项目贡献。这个仓库的规则比一般开源项目**更硬**——因为每一条都是踩出来的。

## 三条不可协商的原则

### 1. 口径先于数值

任何新增指标必须**同时输出它是在什么口径下算出来的**。缺口径就报错，不给默认值。
实现见 `scripts/_common.py` 的 `require_caliber` / `CaliberError`。

反面教材（真实发生过）：λGC 有「Wald 均值」与「p 值中位数」两套合法口径，
混用会在同一个文件上得到两个不同的数，而且**两个都"看起来对"**。

### 2. 算法不自造

统计 / 光学算法只有两条合法来源：

| 来源 | 要求 |
|---|---|
| 从开源项目内嵌 | 许可必须干净（见表）；「掐头去尾取核心 + 依赖换底」；文件头标 `PORTED-FILE` + 来源 + 许可 + **与上游的已知差异** |
| 按公开发表的论文实现 | 必须列出完整文献引用；必须在 `vendor/ATTRIBUTION.md` 登记；必须说明验证级别与**未验证的部分** |

**许可只认权威来源的 SPDX 字段**（GitHub API 的 `license.spdx_id`），不凭印象。
一条写错的许可记录，会让整条能力线凭空消失——本仓库就发生过：scanpy 曾被误记为 GPL，
导致「单细胞分析」整块被排除，而它其实是 BSD-3-Clause。

### 3. 失败不得伪装

- 缺依赖要判 **SKIP**，不是 FAIL（`tests/_harness.py` 的 `find_sci_python` / `skip_reason_sci`）。
- Python 未捕获异常的退出码**恰好也是 1**，所以不能只看退出码——
  要用 `_harness.assert_no_crash()` 按 stderr 有无 traceback 判定。
- 结论对不上时**标红，不放宽断言**。

## 许可红线

| 许可 | 能否内嵌 |
|---|---|
| MIT / BSD-2-Clause / BSD-3-Clause / Apache-2.0 / Unlicense | ✅ 可以 |
| LGPL / GPL-2 / GPL-3 / AGPL-3 | ❌ 一律不内嵌（会传染） |
| 没有 LICENSE 文件 | ❌ 默认「保留全部权利」，等同不可用 |

GPL 系项目走 `tools/fetch_tools.py` 从官方源获取后**独立调用**，不随本仓库分发。

## 提交前必须跑

```bash
python tests/run_all.py            # 全绿：SKIP 可以有，FAIL 不行
python scripts/trigger_check.py    # 改了 SKILL.md 的 description 就必须跑
python scripts/capabilities.py     # 改了 lib/ 或 vendor/ 就应该跑
```

**新增能力必须带负向用例**——构造一个已知会错的输入，断言它**报错**而不是给出假结果。
「能跑通」不算通过。

**新增门禁必须先证明它会失败**——把被保护的文件删掉、把配置改坏，看门禁是否报 FAIL。
否则你无法区分「门禁有效」与「门禁是摆设」，而一个摆设门禁比没有门禁更危险
（它会让「0 失败」变得毫无意义）。

## 提交信息

写清三件事：**改了什么 / 为什么改（原代码错在哪）/ 影响范围**。
只写 `fix bug` 等同于没写。subject 用 Conventional Commits 英文前缀
（`feat:` / `fix:` / `perf:` / `docs:` / `test:` / `chore:`），正文可中文。

## 目录约定

| 目录 | 职责 | 能否直接编辑 |
|---|---|---|
| `scripts/` | Tier 0，纯标准库 | ✅ |
| `lib/` | 本仓库自有扩展（包装层） | ✅ |
| `vendor/` | 内嵌的第三方与移植件 | ⚠️ 同步生成的不要改；移植件见文件头说明 |
| `references/` | 知识层，按需加载 | ✅ |
| `tests/` | 自检层 | ✅ |

## 问题与讨论

有疑问就开 issue。特别是：如果你发现有哪个数字对不上、哪个口径说不通，
**那大概率是真 bug**，欢迎直接报。
