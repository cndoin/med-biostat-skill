"""_sccoda —— 移植自 theislab/scCODA（BSD-3-Clause）的成分数据模型。

算法见 `core.py`（文件头有 PORTED-FILE 声明、来源许可与**与上游的已知差异**）。
**推断引擎已换底**（HMC → 惩罚 MAP + 供体 bootstrap），调用方必须把
`result["inference"]` 那段说明一起报告。
"""
