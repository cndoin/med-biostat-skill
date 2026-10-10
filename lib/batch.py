"""批次效应校正（ComBat）+ 批次诊断（包装层）。

算法在 `vendor/_batch/core.py`（移植自 scanpy `_combat.py`，BSD-3-Clause，
原始方法 Johnson, Li & Rabinovic 2007）。本模块只做接口收敛与口径提示。

**何时用**
- 多个测序批次 / 中心 / 平台合并后，PCA 上批次可分（用 `diagnose` 量化）；
- 下游要做差异表达或聚类，而批次与生物学分组**不完全共线**。

**何时不要用**
- 批次与暴露/分组**完全共线**（如所有病例在 A 中心）—— 校正会连生物学
  效应一起扣掉；这种情况只能做分层或把批次当随机效应建模。
- 只是想「去掉离群样本」—— 那是 QC，不是批次校正。

**口径红线**
1. 输入须为**已 log 变换**的表达（对数域上位置/尺度才可加性校正）。
2. 校正参数只能**在训练/校正集上估计**；不要用别人的 gamma/delta 套新数据。
3. `diagnose` 的 separation ratio 下降**不等于**生物学保留 —— 必须同时看生物学
   分组在校正后是否仍可分（本模块不代替那一步）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from vendor._batch import core as _core

__all__ = ["correct", "correct_reference", "diagnose", "pca_scores"]


def correct(X, batch: Sequence[Any],
            covariates: Optional[Any] = None) -> Dict[str, Any]:
    """ComBat 校正。返回 dict，``adjusted`` 为校正后 DataFrame。"""
    return _core.combat(X, batch, covariates=covariates)


def correct_reference(X, batch: Sequence[Any], reference: Any,
                      covariates: Optional[Any] = None) -> Dict[str, Any]:
    """以 ``reference`` 批次为基准的 ComBat（参考批次保持原尺度不动）。"""
    return _core.combat_reference(X, batch, reference, covariates=covariates)


def diagnose(X, batch: Sequence[Any], n_components: int = 2) -> Dict[str, Any]:
    """批次混杂诊断：校正前后「批次间/批次内」离散度之比。"""
    return _core.batch_diagnostics(X, batch, n_components=n_components)


def pca_scores(X, n_components: int = 2, center: bool = True):
    """样本 PCA 得分（诊断可视化用）。"""
    return _core.pca_scores(X, n_components=n_components, center=center)
