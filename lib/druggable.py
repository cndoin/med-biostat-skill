"""druggable.py —— 基因 → 药物 → 给药途径 → 干预窗口 的**离线确定性**映射。

**为什么只做离线版**
--------------------
DGIdb / ChEMBL / Open Targets 是**网络 API**，不是可内嵌的代码。
本 skill 的红线是"数字必须由程序确定性算出"：网络响应会变、会超时、
会限流，同一个基因今天查 19,300 条明天查 19,301 条，结果不可复现。
故主链路只吃**已落盘的本地药库表**，在线查询留给调用方自行落盘后再喂进来。

（API 口径已在 `drug_emulation_cataract/scripts/_probe_*.py` 实测钉死：
 DGIdb v5 用 `Drug.name` + `first/after` 分页；Open Targets v4 必须用
 `targets(ensemblIds:)` 且 `target(ensemblId:"symbol:X")` **静默返回 null**；
 ChEMBL 要用 `drug_indication?mesh_heading__icontains=` 而非
 `mechanism?mechanism_of_action__icontains=`。落盘时照这些口径取数。）

**必报指标**
------------
``pct_approved_drug_target``：**X% 的效应基因是已上市药物的靶点**。
这是转化出口那句话的算术本体，摘要末句要带这个数字。

**给药途径 = 干预窗口口径**
--------------------------
眼局部给药（滴眼液 / 前房注射）与全身给药是**两个完全不同的干预窗口**：
前者可以长期、患者自付；后者受全身暴露与安全性限制。
本模块强制把 route 分成 `topical` / `systemic` / `unknown` 三类计数，
不允许把两者混成一个"有药可用"的比例——那会把一个口服化疗药
说成"白内障可用"。

**红线**
--------
1. **词表必须由调用方确认**。`approved` / `topical` 在具体药库里的写法
   千差万别（approved / launched / marketed；topical / ophthalmic / ocular）。
   本模块给出默认集合，但**必须报告实际匹配到的取值**（结果里带
   `status_values_seen` / `route_values_seen`），否则"0% 可成药"可能只是
   词表没对上，而不是真的没有药。
2. **基因命名不一致必须报错**，不能静默返回空映射（那会让下游把
   "一个都没匹配上"当成"真的不可成药"，是反向的假结论）。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["map_genes", "druggable_report", "intervention_window",
           "positive_control", "DEFAULT_APPROVED", "DEFAULT_TOPICAL"]


DEFAULT_APPROVED: tuple = ("approved", "launched", "marketed", "approved_eye")
DEFAULT_TOPICAL: tuple = ("topical", "ophthalmic", "ocular", "eye",
                          "intravitreal", "intracameral", "subconjunctival")


def _require(df: pd.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{what} 缺少必需列 {missing}；现有列 {list(df.columns)[:12]}。"
            f"列名对不上时不要猜——猜错会让下游拿到一张全空的映射表，"
            f"然后把「没匹配上」当成「不可成药」")


def map_genes(genes: Sequence[str], drugs,
              gene_col: str = "gene",
              drug_col: str = "drug",
              status_col: Optional[str] = "status",
              route_col: Optional[str] = "route",
              approved_values: Iterable[str] = DEFAULT_APPROVED,
              topical_values: Iterable[str] = DEFAULT_TOPICAL) -> Dict[str, Any]:
    """基因 → 药物映射（离线）。

    :param drugs: 本地药库表（DataFrame 或路径由调用方读入后的对象）。
        必需列：基因列、药名列；可选：status、route
    :return: dict
        - ``table``: 逐基因 DataFrame(gene, n_drugs, n_approved, n_topical,
          approved, topical, systemic, unknown_route, drugs)
        - ``status_values_seen`` / ``route_values_seen``: 实际出现的取值（**必报**）
        - ``n_genes_matched`` / ``n_genes_unmatched``
    """
    D = pd.DataFrame(drugs)
    if len(D) == 0:
        raise ValueError("药库表为空；空表会让下游算出「0% 可成药」这种假结论")
    _require(D, [gene_col, drug_col], "药库表")

    genes_in = [str(g) for g in genes]
    if not genes_in:
        raise ValueError("基因列表为空")

    approved_set = {str(v).strip().lower() for v in approved_values}
    topical_set = {str(v).strip().lower() for v in topical_values}

    D = D.copy()
    D["_gene"] = D[gene_col].astype(str).str.strip().str.upper()
    D["_drug"] = D[drug_col].astype(str).str.strip()

    if status_col and status_col in D.columns:
        D["_status"] = D[status_col].astype(str).str.strip().str.lower()
        status_seen = sorted(set(D["_status"]) - {""})
    else:
        D["_status"] = ""
        status_seen = []
    if route_col and route_col in D.columns:
        D["_route"] = D[route_col].astype(str).str.strip().str.lower()
        route_seen = sorted(set(D["_route"]) - {""})
    else:
        D["_route"] = ""
        route_seen = []

    D["_approved"] = D["_status"].isin(approved_set)
    D["_topical"] = D["_route"].isin(topical_set)

    # 同一 (基因, 药) 重复只记一次：DGIdb 里一个药会被多个来源重复收录，
    # 不去重会把"有 3 条记录"当成"有 3 种药"。
    n_raw = len(D)
    D = D.drop_duplicates(subset=["_gene", "_drug"], keep="first")

    query_up = {g.upper(): g for g in genes_in}
    sub = D[D["_gene"].isin(query_up.keys())]
    if len(sub) == 0:
        raise ValueError(
            f"基因列表 {len(genes_in)} 个与药库**一个都匹配不上**"
            f"（示例：{genes_in[:3]}；药库基因示例："
            f"{list(D['_gene'].unique()[:3])}）。通常是 Symbol vs Ensembl 命名不一致。"
            f"不报错会让下游把「没匹配上」当成「不可成药」——那是反向的假结论")

    rows = []
    for g_up, g in query_up.items():
        s = sub[sub["_gene"] == g_up]
        n_t = int(s["_topical"].sum())
        n_ap = int(s["_approved"].sum())
        n_sys = int((~s["_topical"]).sum())
        rows.append({
            "gene": g,
            "n_drugs": int(len(s)),
            "n_approved": n_ap,
            "n_topical": n_t,
            "n_systemic": n_sys,
            "is_druggable": bool(len(s) > 0),
            "is_approved_target": bool(n_ap > 0),
            "has_topical": bool(n_t > 0),
            "drugs": sorted(set(s["_drug"]))[:20],
            # **药物级**标志：干预窗口表是 (基因 × 药物) 行，若只用基因级的
            # is_approved_target，会把"这个基因有已上市药"错标到它每一个
            # 实验药头上——那是**把实验药说成上市药**，方向错了。
            "drug_approved": {str(d): bool(b) for d, b in
                              zip(s["_drug"], s["_approved"])},
            "drug_topical": {str(d): bool(b) for d, b in
                             zip(s["_drug"], s["_topical"])},
        })
    tab = pd.DataFrame(rows).sort_values(
        ["is_approved_target", "n_approved", "n_drugs"], ascending=False)

    return {
        "table": tab,
        "n_genes": len(genes_in),
        "n_genes_matched": int((tab["n_drugs"] > 0).sum()),
        "n_genes_unmatched": int((tab["n_drugs"] == 0).sum()),
        "n_rows_dedup_removed": int(n_raw - len(D)),
        "status_values_seen": status_seen,
        "route_values_seen": route_seen,
        "approved_values_used": sorted(approved_set),
        "topical_values_used": sorted(topical_set),
        "engine": "lib.druggable.map_genes",
    }


def druggable_report(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """可成药性汇总 —— **必报指标在这里**。

    :return: dict
        - ``pct_druggable``: 有任一药物记录的基因占比
        - ``pct_approved_drug_target``: **已上市药靶占比**（摘要末句那个数字）
        - ``pct_approved_topical``: 其中**可眼局部给药**的占比（干预窗口口径）
        - ``n_*``: 各层计数
    """
    t = mapping["table"]
    n = int(len(t))
    n_drug = int(t["is_druggable"].sum())
    n_app = int(t["is_approved_target"].sum())
    n_top = int((t["is_approved_target"] & t["has_topical"]).sum())
    pct = (lambda x: round(100.0 * x / n, 1) if n else float("nan"))
    return {
        "n_genes": n,
        "n_druggable": n_drug,
        "n_approved_target": n_app,
        "n_approved_topical": n_top,
        "n_approved_systemic_only": int(n_app - n_top),
        "pct_druggable": pct(n_drug),
        "pct_approved_drug_target": pct(n_app),
        "pct_approved_topical": pct(n_top),
        "caliber": ("分母是**输入的效应基因集**（不是全基因组）；"
                    "'可成药' = 药库里有一条记录，'已上市药靶' = 至少一条 "
                    "status 命中 approved 词表；给药途径分层单独计数"),
        "warning": ("若 pct 为 0，先看 mapping['status_values_seen'] —— "
                    "很可能是 approved 词表与药库写法不一致，而不是真的没药"),
        "engine": "lib.druggable.druggable_report",
    }


def intervention_window(mapping: Dict[str, Any], onset,
                        gene_col: str = "gene",
                        onset_gene_col: str = "gene",
                        onset_age_col: str = "onset_age",
                        cell_type_col: Optional[str] = "cell_type",
                        lead_years: float = 0.0) -> Dict[str, Any]:
    """药物 × 细胞类型 × 年龄 —— 干预窗口表（压轴图的表格底稿）。

    :param onset: 失守点表（`lib.age_onset.screen_age_effects` 的 ``table``，
        或任意含 基因列 + 失守年龄列 的 DataFrame）
    :param lead_years: 建议**提前**多少年干预（默认 0 = 就在失守点之前）
    :return: dict，含 ``table``（逐 基因×药物 行）与 ``n_*`` 计数

    口径（写死）：干预窗口 = **失守点之前** ``lead_years`` 年起。
    ``onset_age`` 为空（趋势不显著 / 未检出）的基因**不给窗口**，
    标 ``window="unknown"``，不硬编一个年龄。
    """
    O = pd.DataFrame(onset)
    _require(O, [onset_gene_col, onset_age_col], "失守点表")
    O = O.copy()
    O["_gene"] = O[onset_gene_col].astype(str).str.strip().str.upper()

    M = mapping["table"].copy()
    M["_gene"] = M["gene"].astype(str).str.strip().str.upper()

    # 一个基因可能对应多种药 → 展开成 (基因 × 药) 行
    rows = []
    onset_by_gene = {r["_gene"]: r for _, r in O.iterrows()}
    for _, m in M.iterrows():
        o = onset_by_gene.get(m["_gene"])
        age = o[onset_age_col] if o is not None else None
        try:
            age_f = float(age) if age is not None and np.isfinite(float(age)) else None
        except (TypeError, ValueError):
            age_f = None
        ct = (o[cell_type_col] if (o is not None and cell_type_col
                                   and cell_type_col in O.columns) else None)
        for d in (m["drugs"] or []):
            if age_f is None:
                window = "unknown"
            else:
                lo = max(0.0, age_f - float(lead_years))
                window = f"{lo:.0f}–{age_f:.0f}"
            rows.append({
                "gene": m["gene"],
                "drug": d,
                # 药物级，不是基因级（见 map_genes 里的注释）
                "approved": bool(m["drug_approved"].get(d, False)),
                "topical": bool(m["drug_topical"].get(d, False)),
                "cell_type": ct,
                "onset_age": age_f,
                "window": window,
            })
    tab = pd.DataFrame(rows)
    return {
        "table": tab,
        "n_rows": int(len(tab)),
        "n_with_window": int((tab["window"] != "unknown").sum()),
        "n_unknown_window": int((tab["window"] == "unknown").sum()),
        "lead_years": float(lead_years),
        "caliber": ("窗口 = 失守点前 lead_years 年起；失守点缺失的基因不给窗口"
                    "（硬编一个年龄等于把噪声当发现）"),
        "engine": "lib.druggable.intervention_window",
    }


def positive_control(mapping: Dict[str, Any], known_targets: Sequence[str]) -> Dict[str, Any]:
    """阳性对照：已知药靶必须被映射命中（验证⑥的"阳性对照门"）。

    :param known_targets: 已知一定是药靶的基因（如白内障领域公认的靶点）
    :return: dict，含 ``recall`` 与 ``missed``。
        **recall < 1 就要停主分析先修管线**，不能带着漏检继续往下算。
    """
    t = mapping["table"]
    have = {str(g).upper() for g in t.loc[t["is_druggable"], "gene"]}
    want = [str(g) for g in known_targets]
    if not want:
        raise ValueError("known_targets 为空；阳性对照失去意义")
    hit = [g for g in want if g.upper() in have]
    missed = [g for g in want if g.upper() not in have]
    return {
        "n_known": len(want),
        "n_hit": len(hit),
        "n_missed": len(missed),
        "recall": (len(hit) / len(want)) if want else float("nan"),
        "missed": missed,
        "verdict": ("PASS" if not missed else
                    "FAIL —— 阳性对照未复现，**停主分析先修管线**"
                    "（词表没对上 / 药库缺源 / 命名不一致，必居其一）"),
        "engine": "lib.druggable.positive_control",
    }
