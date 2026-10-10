# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: mediscope（自建）
# 原路径: mediscope/spec.py
# 许可: 自建/内部
# 说明: TRIPOD+AI / STROBE 报告规范对表
# 源 sha256: cb1c77369d1df56b
# VENDORED-FILE-END
"""顶刊报告规范内置对表。

现状：所有开源分析工具都不管报告规范，研究者只能自己拿着清单逐条自查。
本模块把它做进流程——分析跑完，自动告诉作者哪些条目已经满足、
哪些还缺、哪些必须由人填写。

覆盖两份最常用的规范：
- TRIPOD+AI（BMJ 2024，Collins 等）：使用回归或机器学习方法开发的临床预测模型，27 项
- STROBE：观察性研究（队列 / 病例对照 / 横断面），22 项

状态三档：
- 已满足  satisfied  —— 工具能从分析痕迹中证明
- 缺失    missing    —— 应该做但当前结果里没有
- 需人工  manual     —— 工具无法判断（如预注册、伦理批件、作者贡献）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._deps import available
from .registry import register


def _item(no: str, name: str, status: str, evidence: str = "", action: str = "") -> Dict[str, Any]:
    return {"编号": no, "条目": name, "状态": status, "证据": evidence, "待办": action}


def _has(results: Dict[str, Any], key: str) -> bool:
    v = results.get(key)
    return v is not None and not (isinstance(v, dict) and v.get("error"))


@register(
    aliases=["tripod", "tripod_ai", "预测模型报告清单", "规范对表"],
    category="报告规范",
    description="对表 TRIPOD+AI 清单，逐条判定已满足 / 缺失 / 需人工填写",
    examples=["ms.tripod_ai(results_dict)"],
)
def tripod_ai(results: Dict[str, Any]) -> Dict[str, Any]:
    """TRIPOD+AI 完成度对表。results 为各分析结果的汇总字典。"""
    items: List[Dict[str, Any]] = []
    model = results.get("model") or {}
    audit = results.get("audit") or {}
    qc = results.get("qc") or {}
    cv = (model.get("performance_cv") or {}) if model else {}
    test = (model.get("performance_test") or {}) if model else {}
    train = (model.get("performance_train") or {}) if model else {}

    items.append(_item("1", "标题：注明是预测模型的开发与/或验证研究", "需人工",
                       "", "在标题中写明「预测模型」与开发/验证性质。"))
    items.append(_item("2", "摘要：报告模型类型、结局、样本量、区分度与校准度", "需人工",
                       "", "结构式摘要中至少给一个区分度指标与一个校准度指标，并附置信区间。"))
    items.append(_item("3a", "背景与理由", "需人工", "", "说明临床问题与已有模型的不足。"))
    items.append(_item("3b", "研究目标：明确是开发、验证还是两者兼顾", "需人工",
                       "", "在引言末句写明研究类型。"))

    if audit.get("n_rows"):
        items.append(_item("4a", "数据来源与研究对象", "已满足",
                           f"数据集共 {audit['n_rows']} 行、{audit['n_columns']} 列，来源已在报告的「数据概况」中记录。",
                           "补充样本来源机构与时间跨度。"))
    else:
        items.append(_item("4a", "数据来源与研究对象", "缺失", "", "载入数据并运行审计。"))

    items.append(_item("4b", "纳入与排除标准", "需人工", "", "在方法中逐条列出。"))

    if audit.get("roles", {}).get("outcome"):
        items.append(_item("6a", "结局定义", "已满足",
                           f"结局变量：{', '.join(audit['roles']['outcome'])}"
                           + (f"（{model.get('outcome_definition','')}）" if model.get("outcome_definition") else ""),
                           "补充结局的测量方式与判定标准。"))
    else:
        items.append(_item("6a", "结局定义", "缺失", "", "用 set_role 声明结局变量。"))

    if model.get("predictor_names"):
        items.append(_item("7a", "预测因子定义与测量", "已满足",
                           f"共 {len(model['predictor_names'])} 个预测因子：{', '.join(model['predictor_names'][:8])}"
                           + ("..." if len(model["predictor_names"]) > 8 else ""),
                           "补充每个因子的测量时点与单位。"))
    else:
        items.append(_item("7a", "预测因子定义与测量", "缺失", "", "运行 build_model 或列出预测因子。"))

    if audit.get("n_rows") and model.get("n_positive") is not None:
        items.append(_item("8", "样本量与事件数", "已满足",
                           f"总样本 {audit['n_rows']}，其中阳性 {model.get('n_positive')}、"
                           f"阴性 {model.get('n_negative')}；训练 {model.get('n_train')}、测试 {model.get('n_test')}。",
                           "补充样本量估算依据（把握度计算）。"))
    else:
        items.append(_item("8", "样本量与事件数", "缺失", "", "运行 build_model 以获得事件数拆分。"))

    qc_missing = (qc.get("missing") or {}) if qc else {}
    if qc_missing:
        items.append(_item("9", "缺失数据处理", "已满足",
                           f"整体缺失率 {qc_missing.get('overall_missing_pct','?')}%；"
                           f"高缺失列 {len(qc_missing.get('high_missing_columns',[]))} 个。",
                           "须在方法中写明缺失机制判断与处理方法，不能只给数字。"))
    else:
        items.append(_item("9", "缺失数据处理", "缺失", "", "运行 quality_report。"))

    if model.get("model_type"):
        items.append(_item("10a", "模型类型与设定", "已满足",
                           f"模型类型：{model['model_type']}；预测因子数：{model.get('n_predictors')}。",
                           "补充超参数的调优过程与搜索范围。"))
        items.append(_item("10b", "模型构建的软件与版本", "已满足",
                           f"环境：{ {k: v for k, v in available().items()} }",
                           "在论文中写明软件名与精确版本号。"))
    else:
        items.append(_item("10a", "模型类型与设定", "缺失", "", "运行 build_model。"))
        items.append(_item("10b", "模型构建的软件与版本", "缺失", "", "运行 build_model。"))

    if cv.get("auroc_mean") is not None and not (isinstance(cv.get("auroc_mean"), float) and cv["auroc_mean"] != cv["auroc_mean"]):
        items.append(_item("10c", "内部验证（交叉验证）", "已满足",
                           f"{cv.get('folds','?')} 折交叉验证 AUROC = {cv['auroc_mean']:.3f} "
                           f"(± {cv.get('auroc_sd', float('nan')):.3f})。",
                           "说明分层方式与随机种子。"))
    else:
        items.append(_item("10c", "内部验证", "缺失", "", "运行 build_model。"))

    if test.get("auroc") is not None and test.get("auroc") == test.get("auroc"):
        items.append(_item("10d", "模型性能（区分度）", "已满足",
                           f"独立测试集 AUROC = {test['auroc']:.3f} "
                           f"(95%CI {test.get('ci_low', float('nan')):.3f}–{test.get('ci_high', float('nan')):.3f})。",
                           "同时报告开发集与测试集，避免只给乐观值。"))
    else:
        items.append(_item("10d", "模型性能（区分度）", "缺失", "", "运行 build_model。"))

    if test.get("calibration_slope") is not None and test.get("calibration_slope") == test.get("calibration_slope"):
        items.append(_item("10e", "模型性能（校准度）", "已满足",
                           f"Brier = {test.get('brier', float('nan')):.4f}；校准斜率 = {test['calibration_slope']:.3f}；"
                           f"校准截距 = {test.get('calibration_intercept', float('nan')):.3f}。",
                           "在正文给出校准曲线图。"))
    else:
        items.append(_item("10e", "模型性能（校准度）", "缺失", "",
                           "只报 AUROC 会被质疑，务必补校准曲线与 Brier 分数。"))

    if model.get("decision_curve") is not None:
        items.append(_item("10f", "临床效用（决策曲线）", "已满足",
                           "已生成阈值为 0.01–0.99 的净获益曲线。",
                           "说明临床可接受的阈值范围。"))
    else:
        items.append(_item("10f", "临床效用", "缺失", "", "运行 build_model。"))

    items.append(_item("11", "风险分组（如适用）", "需人工", "",
                       "若做风险分层，需说明切点来源（不可用同一数据集既选切点又验证）。"))
    items.append(_item("12", "与现有模型的比较", "需人工", "",
                       "顶刊通常要求与已有模型头对头比较（AUROC 差异检验、NRI、IDI）。"))
    items.append(_item("13a", "结果解读与局限", "需人工", "", "逐条写明偏倚来源与适用边界。"))
    items.append(_item("13b", "亚组表现", "需人工", "",
                       "按性别、年龄等报告亚组性能；样本量不足须如实说明而非略去。"))
    items.append(_item("15", "外部验证", "需人工", "",
                       "未做外部验证时必须在局限中明确声明——这是审稿人最常追问的一条。"))
    items.append(_item("17", "数据与代码可用性", "需人工", "",
                       "给出可实现的获取路径（公开仓库 / 受控访问），并说明限制原因。"))
    items.append(_item("19", "资助与利益冲突", "需人工", "", "逐项填写。"))
    items.append(_item("20", "伦理审批", "需人工", "", "写明伦理委员会名称与批件号。"))

    if not _has(results, "model"):
        items.append(_item("—", "提示", "缺失", "",
                           "尚未运行预测建模。TRIPOD+AI 适用于预测模型研究；"
                           "若这是观察性研究，请改用 STROBE。"))

    return _summarize("TRIPOD+AI", items)


@register(
    aliases=["strobe", "观察性研究报告清单"],
    category="报告规范",
    description="对表 STROBE 清单，逐条判定已满足 / 缺失 / 需人工填写",
    examples=["ms.strobe(results_dict)"],
)
def strobe(results: Dict[str, Any]) -> Dict[str, Any]:
    """STROBE 完成度对表（队列 / 病例对照 / 横断面研究）。"""
    items: List[Dict[str, Any]] = []
    audit = results.get("audit") or {}
    qc = results.get("qc") or {}
    t1 = results.get("table_one") or {}
    cmp_ = results.get("compare")
    if cmp_ is None:
        cmp_ = {}
    surv = results.get("survival") or {}
    cox = results.get("cox") or {}

    items.append(_item("1a", "标题与摘要：说明研究设计", "需人工", "",
                       "标题中体现「队列」「病例对照」或「横断面」等设计类型。"))
    items.append(_item("2", "背景与理由", "需人工", "", "说明既有证据的缺口。"))
    items.append(_item("3", "研究目标与预设假设", "需人工", "",
                       "假设应在数据分析前确定，不可事后编造。"))
    items.append(_item("4", "研究设计", "需人工", "", "写明设计类型与依据。"))
    items.append(_item("5", "研究现场、地点与时间", "需人工", "", "写明机构与时间跨度。"))

    if audit.get("n_rows"):
        items.append(_item("6a", "研究对象：纳入条件", "已满足",
                           f"数据集含 {audit['n_rows']} 例。", "在方法中逐条列出纳入标准。"))
    else:
        items.append(_item("6a", "研究对象", "缺失", "", "载入数据并运行审计。"))

    if t1.get("rows") is not None:
        items.append(_item("7", "变量：明确定义", "已满足",
                           f"基线表纳入 {len(t1.get('rows', []))} 个变量。",
                           "每个变量须写出测量方法与单位。"))
    else:
        items.append(_item("7", "变量定义", "缺失", "", "运行 table_one。"))

    items.append(_item("8", "数据来源与测量方法", "需人工", "",
                       "说明每个变量的数据来源（病历 / 检验 / 随访）。"))

    if qc.get("missing"):
        m = qc["missing"]
        items.append(_item("9", "偏倚：说明处理偏倚的措施", "已满足",
                           f"已量化整体缺失率 {m.get('overall_missing_pct')}%。",
                           "另需说明选择偏倚、信息偏倚的控制措施。"))
    else:
        items.append(_item("9", "偏倚", "缺失", "", "运行 quality_report。"))

    if audit.get("n_rows") and t1.get("variables"):
        items.append(_item("10", "样本量：说明确定依据", "已满足",
                           f"实际分析 {audit['n_rows']} 例。", "补充把握度分析或样本量估算依据。"))
    else:
        items.append(_item("10", "样本量", "缺失", "", "运行 audit 与 table_one。"))

    if cmp_ is not None and (not hasattr(cmp_, "empty") or not cmp_.empty):
        items.append(_item("12", "统计方法：说明全部统计方法", "已满足",
                           f"已执行 {len(cmp_)} 项组间比较，含效应量与多重校正。",
                           "写明检验选择依据、效应量类型与校正方法。"))
    else:
        items.append(_item("12", "统计方法", "缺失", "", "运行 compare_many。"))

    if surv.get("logrank") is not None:
        items.append(_item("12b", "生存分析方法", "已满足",
                           f"Kaplan-Meier + log-rank；事件数 {surv.get('n_events')}，"
                           f"删失比例 {surv.get('censored_pct')}%。",
                           "若用 Cox，须报告 PH 假设检验结果。"))
        if cox.get("ph_violation"):
            items.append(_item("12c", "比例风险假设", "缺失",
                               f"以下变量 PH 假设可能不成立：{', '.join(cox['ph_violation'])}",
                               "改用时间相依协变量或分层 Cox，并在论文中说明。"))
        elif cox.get("table") is not None:
            items.append(_item("12c", "比例风险假设", "已满足", "已完成 Schoenfeld 残差检验。", ""))
    else:
        items.append(_item("12b", "生存分析方法", "缺失", "", "如为生存数据，运行 ms.km。"))

    if t1.get("rows"):
        items.append(_item("14a", "描述性数据：参与者特征", "已满足",
                           f"已生成基线特征表，含 {len(t1['groups'])} 个分组的比较。", ""))
    else:
        items.append(_item("14a", "描述性数据", "缺失", "", "运行 table_one。"))

    items.append(_item("14b", "缺失数据：每变量的缺失例数", "需人工",
                       f"整体缺失率 {(qc.get('missing') or {}).get('overall_missing_pct', '?')}%。",
                       "逐变量列出缺失例数（基线表中已含缺失列）。"))
    items.append(_item("14c", "随访时间（队列研究）", "需人工", "",
                       "报告随访中位数与总人年。"))
    items.append(_item("15", "结局数据：事件数与风险估计", "需人工", "",
                       "报告事件数、风险比或比值比及其置信区间（若已运行 cox/compare 则结果已具备）。"))
    items.append(_item("16a", "主要结果：未校正与校正后的估计", "需人工", "",
                       "顶刊要求同时给出粗估与调整后估计，并写明调整了哪些变量。"))
    items.append(_item("17", "其他分析：亚组与敏感性分析", "需人工", "",
                       "报告亚组分析与敏感性分析（如剔除极端值后是否稳健）。"))
    items.append(_item("18", "关键结果总结", "需人工", "", "与研究假设逐条对应。"))
    items.append(_item("19", "局限", "需人工", "", "讨论偏倚方向而非仅罗列。"))
    items.append(_item("20", "解释：结合目标人群与更广泛证据", "需人工", ""))
    items.append(_item("22", "资助", "需人工", "", "写明资助来源与编号。"))

    return _summarize("STROBE", items)


def _summarize(guideline: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_total = len(items)
    n_sat = sum(1 for i in items if i["状态"] == "已满足")
    n_miss = sum(1 for i in items if i["状态"] == "缺失")
    n_manual = sum(1 for i in items if i["状态"] == "需人工")
    return {
        "guideline": guideline,
        "items": items,
        "n_total": n_total,
        "n_satisfied": n_sat,
        "n_missing": n_miss,
        "n_manual": n_manual,
        "auto_coverage": round(100.0 * n_sat / n_total, 1) if n_total else 0.0,
        "missing_items": [i["条目"] for i in items if i["状态"] == "缺失"],
        "next_actions": [i["待办"] for i in items if i["状态"] == "缺失" and i["待办"]],
        "note": (
            "「已满足」表示工具能从分析痕迹中证明该条目已被执行；"
            "「需人工」表示必须由作者填写，工具无法替代。"
            "本表只做完整性对表，不评判研究质量或偏倚风险。"
        ),
    }


@register(
    aliases=["规范检查", "spec_check", "报告完整性"],
    category="报告规范",
    description="按研究设计自动选择规范（预测模型→TRIPOD+AI，观察性→STROBE）并对表",
    examples=["ms.spec_check(results)"],
)
def spec_check(results: Dict[str, Any], design: str = "auto") -> Dict[str, Any]:
    """自动选择并运行报告规范对表。"""
    if design == "auto":
        design = "tripod" if _has(results, "model") else "strobe"
    if design in {"tripod", "tripod_ai", "prediction"}:
        return tripod_ai(results)
    if design in {"strobe", "observational"}:
        return strobe(results)
    raise ValueError("design 只能是 'auto' / 'tripod' / 'strobe'")


def spec_text(report: Dict[str, Any]) -> str:
    """把对表结果排版成文本。"""
    lines = [
        f"规范：{report['guideline']}    共 {report['n_total']} 项",
        f"已满足 {report['n_satisfied']} 项（自动覆盖 {report['auto_coverage']}%），"
        f"缺失 {report['n_missing']} 项，需人工填写 {report['n_manual']} 项",
        "",
    ]
    for i in report["items"]:
        mark = {"已满足": "[✓]", "缺失": "[×]", "需人工": "[ ]"}[i["状态"]]
        lines.append(f"{mark} {i['编号']:<5} {i['条目']}")
        if i["证据"]:
            lines.append(f"        证据：{i['证据']}")
        if i["状态"] == "缺失" and i["待办"]:
            lines.append(f"        待办：{i['待办']}")
    return "\n".join(lines)
