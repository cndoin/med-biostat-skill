"""vendor —— 内嵌工具层（Tier 1）。

设计原则
--------
1. **单一来源**：本目录下所有文件由 `tools/vendor_sync.py` 从上游同步生成，
   带 `# VENDORED-FILE` 头，**不要直接编辑**（改源文件后重跑同步脚本）。
2. **优雅降级**：Tier 1 需要 numpy / pandas / scipy；缺失时本包**不报错**，
   而是把对应能力标记为不可用（`capabilities()`），Tier 0（scripts/，纯标准库）照常工作。
3. **唯一裁决点**：与上游重复的能力只在此保留一份，避免同一逻辑多处实现。

能力分层
--------
* Tier 0（`scripts/`，纯标准库）：取回/核验/质控/位点计数/互证/报告门禁 —— 永远可用。
* Tier 1（本目录，需 numpy/pandas/scipy）：
  - `core.stats`      多重检验 / 效应量 / 缺失 / bootstrap / 假设检验
  - `engines.clinical`  基线表 Table 1 / 两组比较 / 风险表
  - `engines.survival`  KM / log-rank / Cox / 比例风险检验
  - `engines.differential` 差异表达 + 超几何富集
  - `engines.ptm`      翻译后修饰占位率 / 局部 FDR
  - `engines.modeling` 自实现建模（无 sklearn）+ 校准 + 决策曲线
  - `engines.mr_coloc` 孟德尔随机化 + 简化共定位
  - `viz.svg_charts`   纯 SVG 图表（无 matplotlib）
* 零依赖子模块（即使无 numpy 也可用）：
  - `registry`  函数注册表（中英文检索 + 前置校验，纯标准库）
  - `spec`      TRIPOD+AI / STROBE 报告规范对表
"""

from __future__ import annotations

from pathlib import Path

__version__ = "1.0.0"

_HERE = Path(__file__).resolve().parent

# ---- 依赖探测（用内嵌的 _deps，缺依赖不抛错）----
try:
    from . import _deps as _d
    HAS_NUMPY = _d.has("numpy")
    HAS_PANDAS = _d.has("pandas")
    HAS_SCIPY = _d.has("scipy")
except Exception:  # pragma: no cover - 极端情况
    HAS_NUMPY = HAS_PANDAS = HAS_SCIPY = False

# Tier 1 科学栈就绪 = numpy + pandas（scipy 为增强，缺则部分功能降级）
SCI_STACK_READY = bool(HAS_NUMPY and HAS_PANDAS)

# 真实导入成功的模块名集合（由下方条件导出填充）。
# capabilities() 逐模块查它，而不是用 SCI_STACK_READY 一刀切。
_IMPORTED: set = set()

# 零依赖子模块（总是可用）
from . import registry as registry          # noqa: E402,F401
from . import spec as spec                  # noqa: E402,F401

# 条件导出：Tier 1 引擎（缺依赖时保持 None，由调用方查 capabilities()）
stats = clinical = survival = differential = ptm = modeling = mr_coloc = svg_charts = None

if SCI_STACK_READY:
    try:
        from .core import stats as stats            # noqa: F811
        from .core import contract as contract      # noqa: F401
        from .core import audit as audit            # noqa: F401
        from .engines import clinical as clinical    # noqa: F811
        from .engines import survival as survival    # noqa: F811
        from .engines import differential as differential  # noqa: F811
        from .engines import ptm as ptm              # noqa: F811
        from .engines import modeling as modeling    # noqa: F811
        from .engines import mr_coloc as mr_coloc    # noqa: F811
        from .viz import svg_charts as svg_charts    # noqa: F811
        # 记录真正可用的模块（含零依赖的两个）
        _IMPORTED.update({"stats", "clinical", "survival", "differential",
                          "ptm", "modeling", "mr_coloc", "svg_charts"})
    except Exception as _e:  # noqa: BLE001
        # 依赖在但引擎 import 失败 → 记录，不让整个包崩
        _IMPORT_ERROR = repr(_e)
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None


def capabilities() -> dict:
    """报告内嵌层的能力就绪度，供 scripts/capabilities.py 与文档使用。"""
    return {
        "version": __version__,
        "tier0_stdlib": True,
        "tier1_sci_stack_ready": SCI_STACK_READY,
        "deps": {"numpy": HAS_NUMPY, "pandas": HAS_PANDAS, "scipy": HAS_SCIPY},
        "zero_dep_modules": ["registry", "spec"],
        "tier1_modules": {
            # 逐模块**真实导入探测**，而不是靠 SCI_STACK_READY 一刀切。
            # 原写法 bool((name in ("stats",)) and SCI_STACK_READY) 恒只报
            # stats 一项：科学栈齐备时其余 7 个模块其实也全部可用，
            # 报告却说不可用 —— 能力误报会让人以为功能缺失而重复造轮子。
            name: (name in _IMPORTED)
            for name in ["stats", "clinical", "survival", "differential",
                         "ptm", "modeling", "mr_coloc", "svg_charts"]
        },
        "import_error": _IMPORT_ERROR,
        "install_hint": "pip install -r vendor/requirements.txt",
    }
