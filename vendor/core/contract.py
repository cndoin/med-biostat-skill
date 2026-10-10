# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/core/contract.py
# 许可: 自建/内部
# 说明: 数据契约与实验设计体检
# 源 sha256: e05f10490625dc9f
# VENDORED-FILE-END
"""数据契约层。

借鉴 SDRF（Sample and Data Relationship Format）的思想：把实验设计显式声明成一张表，
而不是散落在脚本里的隐式假设。契约在进入任何分析之前校验，失败即中止。

一个合规的样本表最少需要：
    sample_id   唯一标识，不可重复、不可为空
    group       分组变量（暴露/疾病/亚型），建模与比较的主变量
可选但强烈建议：
    batch       批次（采集批次、质谱批次、测序 lane）
    site        中心（多中心研究必需，用于外部验证切分）
    age / sex   人口学协变量
    time / event 生存分析的随访时间与结局指示
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "group")
RECOMMENDED_COLUMNS = ("batch", "site", "age", "sex")

SURVIVAL_COLUMNS = ("time", "event")


class ContractError(ValueError):
    """契约校验失败。携带全部问题，而不是第一条就抛出。"""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("数据契约校验失败（%d 项）：\n  - %s" % (
            len(self.problems), "\n  - ".join(self.problems)))


@dataclass
class DesignReport:
    """实验设计体检结果。"""

    n_samples: int
    n_groups: int
    group_counts: Dict[str, int]
    batch_group_table: Optional[pd.DataFrame] = None
    min_cell: Optional[int] = None
    balanced: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_groups": self.n_groups,
            "group_counts": self.group_counts,
            "min_cell": self.min_cell,
            "balanced": self.balanced,
            "warnings": list(self.warnings),
        }


def _is_number(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def validate(metadata: pd.DataFrame,
             need_survival: bool = False,
             need_site: bool = False) -> List[str]:
    """返回问题列表；空列表表示通过。"""
    problems: List[str] = []

    if not isinstance(metadata, pd.DataFrame):
        raise ContractError(["metadata 必须是 pandas.DataFrame"])

    missing = [c for c in REQUIRED_COLUMNS if c not in metadata.columns]
    if missing:
        problems.append("缺少必需列：%s" % "、".join(missing))
        # 必需列缺失时后续检查无法进行
        return problems

    sid = metadata["sample_id"]
    if sid.isna().any():
        problems.append("sample_id 存在空值：%d 个" % int(sid.isna().sum()))
    if sid.duplicated().any():
        dup = sid[sid.duplicated()].unique()[:5]
        problems.append("sample_id 重复：%s" % ", ".join(map(str, dup)))

    grp = metadata["group"]
    if grp.isna().any():
        problems.append("group 存在空值：%d 个" % int(grp.isna().sum()))
    vc = grp.value_counts(dropna=True)
    if len(vc) < 2:
        problems.append("group 至少需要 2 个水平，当前 %d 个" % len(vc))
    elif int(vc.min()) < 3:
        problems.append("最小组样本量 %d < 3，无法估计组内方差" % int(vc.min()))

    if need_survival:
        for c in SURVIVAL_COLUMNS:
            if c not in metadata.columns:
                problems.append("生存分析需要列 %s" % c)
        if "event" in metadata.columns:
            ev = pd.to_numeric(metadata["event"], errors="coerce")
            if ev.notna().any() and not set(ev.dropna().unique()) <= {0, 1}:
                problems.append("event 必须为 0/1 指示变量")
        if "time" in metadata.columns:
            tt = pd.to_numeric(metadata["time"], errors="coerce")
            if tt.notna().any() and (tt.dropna() < 0).any():
                problems.append("time 存在负值")

    if need_site and "site" not in metadata.columns:
        problems.append("多中心外部验证需要 site 列")

    for c in ("age", "time"):
        if c in metadata.columns and not _is_number(metadata[c]):
            problems.append("列 %s 应为数值型" % c)

    return problems


def check_design(metadata: pd.DataFrame) -> DesignReport:
    """实验设计体检：组间均衡、批次混杂、空单元。"""
    vc = metadata["group"].value_counts()
    rep = DesignReport(
        n_samples=int(len(metadata)),
        n_groups=int(len(vc)),
        group_counts={str(k): int(v) for k, v in vc.items()},
    )

    imbalance = int(vc.max()) / max(int(vc.min()), 1)
    if imbalance > 3:
        rep.warnings.append(
            "组间样本量失衡（最大/最小 = %.1f），效应估计对抽样波动敏感" % imbalance)

    if "batch" in metadata.columns:
        tab = pd.crosstab(metadata["batch"], metadata["group"])
        rep.batch_group_table = tab
        rep.min_cell = int(tab.min().min()) if tab.size else None
        if rep.min_cell == 0:
            rep.balanced = False
            rep.warnings.append(
                "存在空的 批次×分组 单元：该批次未覆盖全部分组，批次效应与分组效应不可分离")
        elif rep.min_cell < 3:
            rep.warnings.append(
                "最小的 批次×分组 单元仅 %d 例，批次校正不稳定" % rep.min_cell)
        # 卡方独立性近似检验
        try:
            from scipy.stats import chi2_contingency
            chi2, p, _, _ = chi2_contingency(tab.values)
            if p < 0.05:
                rep.warnings.append(
                    "批次与分组不独立（卡方 p = %.3g）：存在批次混杂，校正后仍需谨慎解释" % p)
        except Exception:
            pass

    miss = metadata.isna().mean()
    bad = miss[miss > 0.3]
    for col, frac in bad.items():
        rep.warnings.append("列 %s 缺失率 %.0f%%，超过 30%%" % (col, frac * 100))

    return rep


def fingerprint(obj) -> str:
    """对任意可序列化对象求稳定指纹，用于溯源。"""
    if isinstance(obj, pd.DataFrame):
        payload = pd.util.hash_pandas_object(obj, index=True).values.tobytes()
    elif isinstance(obj, np.ndarray):
        payload = np.ascontiguousarray(obj).tobytes()
    elif isinstance(obj, (dict, list, tuple)):
        payload = repr(sorted(obj.items()) if isinstance(obj, dict) else list(obj)).encode()
    else:
        payload = str(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """统一入口：始终以 utf-8-sig 读取，避免 BOM 造成的列名污染。"""
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
