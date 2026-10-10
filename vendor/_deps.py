# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: mediscope（自建）
# 原路径: mediscope/_optional.py
# 许可: 自建/内部
# 说明: 可选依赖惰性加载与降级
# 源 sha256: 280058bb5a5fd872
# VENDORED-FILE-END
"""可选依赖与惰性导入。

设计目标：缺任何一个可选依赖，都不应影响其余功能可用。
（此手法参考 OmicVerse 的 _optional.py / _lazy_loader.py，
本实现为独立编写，仅借鉴思路。）
"""

from __future__ import annotations

import importlib
from functools import lru_cache

# 可选依赖 -> 该依赖解锁的能力说明（供报错时提示用户）
OPTIONAL_DEPS = {
    "sklearn": "预测建模、交叉验证、正则化回归",
    "matplotlib": "图形输出（报告中的曲线与箱线图）",
    "statsmodels": "Cox 比例风险模型、多重检验、方差分析",
    "scipy": "统计检验、分布函数",
}


@lru_cache(maxsize=None)
def has(module: str) -> bool:
    """判断可选依赖是否可用。结果缓存，避免重复 import 开销。"""
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def require(module: str, feature: str = "") -> None:
    """断言依赖可用，否则抛出带安装指引的错误。

    Parameters
    ----------
    module : str
        模块名。
    feature : str
        该模块支撑的功能名，用于生成可读的报错信息。
    """
    if has(module):
        return
    hint = OPTIONAL_DEPS.get(module, "")
    feature_txt = f"（{feature}）" if feature else ""
    raise ImportError(
        f"缺少可选依赖 `{module}`{feature_txt}。\n"
        f"该依赖用于：{hint or '相关功能'}。\n"
        f"安装：pip install {module}"
    )


def lazy(module: str, feature: str = ""):
    """按需导入并返回模块对象。"""
    require(module, feature)
    return importlib.import_module(module)


def available() -> dict:
    """返回所有可选依赖的可用状态，供审计与报告写入环境信息。"""
    return {name: has(name) for name in OPTIONAL_DEPS}
