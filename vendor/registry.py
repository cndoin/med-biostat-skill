# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: mediscope（自建）
# 原路径: mediscope/registry.py
# 许可: 自建/内部
# 说明: 函数注册表：中英文检索 + 前置校验（纯标准库）
# 源 sha256: cdf4b4c33437f25f
# VENDORED-FILE-END
"""函数注册表。

用途：让每一个分析函数都自带元数据（别名、分类、前置条件、产出物），
从而支持三件事：(1) 用中文或英文检索函数；(2) 执行前自动校验数据是否具备；
(3) 导出注册表用于生成文档与审计。

（装饰器注册 + 数据契约的思路参考 OmicVerse 的 _registry.py，
本实现为独立编写，接口按临床表格数据的场景重新设计，不再依赖 AnnData。）
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# requires / produces 允许的键
#   columns : 必需/新增的列名
#   roles   : 必需/新增的变量角色（outcome / time / event / exposure / ...）
#   min_rows: 最少观测数
_ALLOWED_KEYS = {"columns", "roles", "min_rows", "attrs"}
_AUTO_FIX_LEVELS = {"auto", "escalate", "none"}


class Registry:
    """全局函数注册表。"""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._by_full: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, List[str]] = {}

    # ---------------------------------------------------------------- 注册

    def register(
        self,
        func: Callable,
        aliases: List[str],
        category: str,
        description: str,
        requires: Optional[Dict[str, Any]] = None,
        produces: Optional[Dict[str, Any]] = None,
        auto_fix: str = "none",
        examples: Optional[List[str]] = None,
        related: Optional[List[str]] = None,
    ) -> Callable:
        if not aliases or not all(str(a).strip() for a in aliases):
            raise ValueError("注册函数必须提供至少一个非空别名。")
        if not category or not str(category).strip():
            raise ValueError("注册函数必须提供分类。")
        if not description or not str(description).strip():
            raise ValueError("注册函数必须提供描述。")
        if auto_fix not in _AUTO_FIX_LEVELS:
            raise ValueError(f"auto_fix 取值非法：{auto_fix}，应为 {_AUTO_FIX_LEVELS}")

        for label, spec in (("requires", requires), ("produces", produces)):
            if spec is None:
                continue
            if not isinstance(spec, dict):
                raise TypeError(f"{label} 必须是字典。")
            bad = set(spec) - _ALLOWED_KEYS
            if bad:
                raise ValueError(f"{label} 含非法键 {bad}，允许：{sorted(_ALLOWED_KEYS)}")

        full_name = f"{func.__module__}.{func.__name__}"
        try:
            signature = str(inspect.signature(func))
        except Exception:
            signature = "(df, **kwargs)"

        entry = {
            "function": func,
            "full_name": full_name,
            "short_name": func.__name__,
            "module": func.__module__,
            "aliases": [str(a).strip() for a in aliases],
            "category": category,
            "description": description,
            "signature": signature,
            "docstring": inspect.getdoc(func) or "",
            "requires": requires or {},
            "produces": produces or {},
            "auto_fix": auto_fix,
            "examples": list(examples or []),
            "related": list(related or []),
        }

        self._by_full[full_name] = entry
        for alias in entry["aliases"]:
            self._entries[alias.lower()] = entry
        self._entries[entry["short_name"].lower()] = entry
        self._entries[full_name.lower()] = entry

        self._categories.setdefault(category, [])
        if full_name not in self._categories[category]:
            self._categories[category].append(full_name)
        return func

    # ---------------------------------------------------------------- 检索

    def find(self, query: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """按别名、描述、文档字符串检索函数。支持中英文。"""
        q = str(query).strip().lower()
        if not q:
            return []
        results: List[Dict[str, Any]] = []
        seen = set()

        def _push(entry):
            if entry["full_name"] not in seen:
                seen.add(entry["full_name"])
                results.append(entry)

        if q in self._entries:
            _push(self._entries[q])

        for match in get_close_matches(q, list(self._entries), n=5, cutoff=threshold):
            _push(self._entries[match])

        for entry in self._by_full.values():
            if entry["full_name"] in seen:
                continue
            haystack = " ".join([
                entry["description"],
                entry["docstring"],
                " ".join(entry["aliases"]),
                " ".join(entry["related"]),
                entry["short_name"],
            ]).lower()
            if q in haystack:
                _push(entry)
        return results

    def categories(self) -> List[str]:
        return sorted(self._categories)

    def list_in(self, category: str) -> List[Dict[str, Any]]:
        return [self._by_full[n] for n in self._categories.get(category, [])]

    def entry(self, query: str) -> Optional[Dict[str, Any]]:
        found = self.find(query)
        return found[0] if found else None

    # ------------------------------------------------------------ 前置校验

    def check(self, query: str, df) -> Dict[str, Any]:
        """校验数据是否满足某函数的前置条件。

        返回 ``satisfied`` 与缺失项清单，以及可执行的补全建议。
        """
        entry = self.entry(query)
        if entry is None:
            return {"satisfied": False, "reason": f"未找到函数：{query}", "missing": []}

        req = entry.get("requires") or {}
        missing: List[str] = []
        meta = (getattr(df, "attrs", {}) or {}).get("variable_meta", {}) or {}
        roles = {m.get("role") for m in meta.values() if isinstance(m, dict)}

        for col in req.get("columns", []):
            if col not in getattr(df, "columns", []):
                missing.append(f"缺少列：{col}")
        for role in req.get("roles", []):
            if role not in roles:
                missing.append(f"缺少角色声明：{role}")
        if req.get("min_rows") and len(df) < int(req["min_rows"]):
            missing.append(f"观测数不足：{len(df)} < {req['min_rows']}")

        satisfied = not missing
        advice = ""
        if not satisfied:
            advice = (
                f"请先补全：{'；'.join(missing)}。"
                "可用 Mediscope.set_role() 声明变量角色，或 Mediscope.audit() 查看数据全貌。"
            )
        return {
            "satisfied": satisfied,
            "function": entry["full_name"],
            "missing": missing,
            "advice": advice,
        }

    # ---------------------------------------------------------------- 导出

    def export(self, path: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_functions": len(self._by_full),
                "categories": self.categories(),
            },
            "functions": {
                name: {k: v for k, v in e.items() if k != "function"}
                for name, e in self._by_full.items()
            },
        }
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def format_found(self, results: List[Dict[str, Any]], verbose: bool = False) -> str:
        if not results:
            return "未找到匹配的函数。"
        lines = [f"找到 {len(results)} 个匹配函数："]
        for i, e in enumerate(results, 1):
            lines.append(f"\n{i}. {e['full_name']}{e['signature']}")
            lines.append(f"   说明：{e['description']}")
            lines.append(f"   别名：{', '.join(e['aliases'])}    分类：{e['category']}")
            if verbose and e["examples"]:
                lines.append("   示例：")
                lines.extend(f"     {x}" for x in e["examples"])
        return "\n".join(lines)


# 全局单例
_REGISTRY = Registry()


def register(
    aliases: List[str],
    category: str,
    description: str,
    requires: Optional[Dict[str, Any]] = None,
    produces: Optional[Dict[str, Any]] = None,
    auto_fix: str = "none",
    examples: Optional[List[str]] = None,
    related: Optional[List[str]] = None,
):
    """注册一个分析函数的装饰器。

    Examples
    --------
    >>> @register(aliases=["基线表", "table1"], category="描述统计",
    ...           description="生成分层基线特征表")
    ... def table_one(df, group=None):
    ...     ...
    """

    def decorator(func: Callable) -> Callable:
        _REGISTRY.register(
            func=func,
            aliases=aliases,
            category=category,
            description=description,
            requires=requires,
            produces=produces,
            auto_fix=auto_fix,
            examples=examples,
            related=related,
        )
        return func

    return decorator


def get_registry() -> Registry:
    return _REGISTRY
