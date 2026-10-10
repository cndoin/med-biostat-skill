"""lib —— 本 skill 自有的分析扩展层（Tier 1，与 vendor/ 的同步副本区分）

  * `vendor/`  内文件由 tools/vendor_sync.py 从上游同步（带 VENDORED-FILE 头）
  * `lib/`     本 skill 自有实现（带 SKILL-OWNED 头），直接编辑

放置理由：这些模块补的是**项目 PROTOCOL 明确要求、但上游工具没覆盖**的判据
（C-index/NRI/IDI、E-value、竞争风险、线性混合效应）。
"""
