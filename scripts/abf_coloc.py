#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""abf_coloc.py —— 完整 ABF 共定位（Tier0，纯标准库）

依据 Wakefield 近似贝叶斯因子 + Giambartolomei 2014 的单因果假设分解。
公式经三处权威来源核对（coloc 源码 R/claudia.R、coloc 官方文档、Giambartolomei 2014 教程），
**不是凭记忆写的**：

  per-SNP：
    z = beta / se
    V = se²
    W = (sd_prior · sdY)²         # 定量性状 sd_prior=0.15；病例对照 0.20
    r = W / (V + W)               # 收缩因子
    lABF = 0.5 · ( log(1−r) + r·z² )

  区域级（log 空间，logsumexp）：
    l1   = logsumexp(lABF1)
    l2   = logsumexp(lABF2)
    lsum = logsumexp(lABF1 + lABF2)
    lH0  = 0
    lH1  = log(p1)  + l1
    lH2  = log(p2)  + l2
    lH3  = log(p1·p2) + logdiffexp(l1+l2, lsum)     # Σ_{j≠k} ABF1_j·ABF2_k
    lH4  = log(p12) + lsum                          # Σ_j ABF1_j·ABF2_j
    PP.Hk = softmax([lH0..lH4])

PROTOCOL 硬约束（dhp / ptgds 共有）：
  * **必须用完整未筛选的区域统计量**，不得只取显著子集（只取显著子集会严重高估 H4）
  * 主先验 p1=p2=1e-4、p12=1e-5；p12 ∈ {1e-6, 5e-6} 做敏感性
  * sdY 主值 + 0.5 / 1.5 敏感性
  * 单因果假设必须显式声明；PP.H4 > 0.8 常作主判据
  * 无法获取完整区域时**不得声称共定位**

用法：
    python abf_coloc.py --trait1 eqtl_region.tsv --trait2 gwas_region.tsv \
        --p12-list 1e-6,5e-6,1e-5 --sdy-list 0.5,1,1.5 --json --out logs/coloc.json
退出码：0 成功 / 2 输入错误 / 3 口径缺失
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NEG_INF = float("-inf")

ALIAS = {
    "snp": ["snp", "rsid", "variant", "v", "id", "markername"],
    "beta": ["beta", "b", "effect", "slope", "beta1", "beta2"],
    "se": ["se", "sebeta", "stderr", "standard_error", "varbeta"],
}


def resolve(header):
    low = {}
    for i, h in enumerate(header):
        low[h.lower()] = i
        low[h.lower().lstrip("#")] = i
    out = {}
    for k, cands in ALIAS.items():
        for c in cands:
            if c.lower() in low:
                out[k] = low[c.lower()]
                break
    return out


def logsumexp(xs):
    """log(Σ exp(x))，数值稳定。空或全 −inf 返回 −inf。"""
    xs = [x for x in xs if x != NEG_INF]
    if not xs:
        return NEG_INF
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def logdiffexp(a, b):
    """log(exp(a) − exp(b))，要求 a > b（否则返回 −inf）。"""
    if a <= b:
        return NEG_INF
    return a + math.log1p(-math.exp(b - a))


def labf(beta: float, se: float, w: float) -> float:
    """Wakefield 近似贝叶斯因子的对数值（单个 SNP 对单个性状）。"""
    if se is None or se <= 0:
        return NEG_INF
    v = se * se
    z = beta / se
    r = w / (v + w)
    if r <= 0 or r >= 1:
        return NEG_INF
    return 0.5 * (math.log(1.0 - r) + r * z * z)


def load_trait(path: str) -> dict:
    """读取单性状区域统计：{snp: (beta, se)}。"""
    out = {}
    with C.open_text(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = resolve(header)
        if "snp" not in col or "beta" not in col or "se" not in col:
            raise ValueError(f"需要 snp/beta/se 列；实际表头={header}")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            try:
                out[f[col["snp"]]] = (float(f[col["beta"]]), float(f[col["se"]]))
            except (ValueError, IndexError):
                continue
    return out


def run_coloc(a: dict, b: dict, snps: list, p1: float, p2: float, p12: float,
              w: float):
    """在给定 SNP 交集上跑一次 ABF 共定位，返回五个后验与逐 SNP 支持。"""
    l1_list, l2_list, both = [], [], []
    per_snp = []
    for s in snps:
        b1, se1 = a[s]
        b2, se2 = b[s]
        x1 = labf(b1, se1, w)
        x2 = labf(b2, se2, w)
        l1_list.append(x1)
        l2_list.append(x2)
        both.append(x1 + x2)
        per_snp.append({"snp": s, "lABF1": x1, "lABF2": x2})

    l1 = logsumexp(l1_list)
    l2 = logsumexp(l2_list)
    lsum = logsumexp(both)

    lH0 = 0.0
    lH1 = math.log(p1) + l1 if p1 > 0 and l1 != NEG_INF else NEG_INF
    lH2 = math.log(p2) + l2 if p2 > 0 and l2 != NEG_INF else NEG_INF
    # H3：两个不同因果变异 → Σ_{j≠k} = Σ_j,k − Σ_{j=k}
    lH3 = (math.log(p1) + math.log(p2) + logdiffexp(l1 + l2, lsum)
           if p1 > 0 and p2 > 0 and l1 != NEG_INF and l2 != NEG_INF else NEG_INF)
    lH4 = math.log(p12) + lsum if p12 > 0 and lsum != NEG_INF else NEG_INF

    Ls = [lH0, lH1, lH2, lH3, lH4]
    denom = logsumexp(Ls)
    pp = [math.exp(x - denom) if x != NEG_INF else 0.0 for x in Ls]

    # 逐 SNP 的 PP.H4（条件于 H4）
    snp_pp = []
    if lsum != NEG_INF:
        for item in per_snp:
            snp_pp.append({"snp": item["snp"],
                           "lABF1": round(item["lABF1"], 4) if item["lABF1"] != NEG_INF else None,
                           "lABF2": round(item["lABF2"], 4) if item["lABF2"] != NEG_INF else None,
                           "SNP.PP.H4": round(math.exp(item["lABF1"] + item["lABF2"] - lsum), 6)})
    snp_pp.sort(key=lambda d: d["SNP.PP.H4"], reverse=True)

    return {
        "PP.H0": pp[0], "PP.H1": pp[1], "PP.H2": pp[2], "PP.H3": pp[3], "PP.H4": pp[4],
        "log_support": {"lH0": lH0, "lH1": lH1, "lH2": lH2, "lH3": lH3, "lH4": lH4},
        "best_snp": snp_pp[0] if snp_pp else None,
        "top_snps": snp_pp[:5],
        "verdict": ("强共定位" if pp[4] >= 0.8 else
                    "中等证据" if pp[4] >= 0.5 else
                    "无共定位（信号可能由不同变异驱动）"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="完整 ABF 共定位（H0–H4）")
    ap.add_argument("--trait1", required=True, help="性状1 区域统计（snp/beta/se）")
    ap.add_argument("--trait2", required=True)
    ap.add_argument("--p1", type=float, default=1e-4)
    ap.add_argument("--p2", type=float, default=1e-4)
    ap.add_argument("--p12", type=float, default=1e-5)
    ap.add_argument("--p12-list", default=None,
                    help="p12 敏感性列表，逗号分隔，如 1e-6,5e-6,1e-5")
    ap.add_argument("--sd-y", type=float, default=1.0)
    ap.add_argument("--sdy-list", default=None,
                    help="sdY 敏感性列表，如 0.5,1,1.5")
    ap.add_argument("--type", choices=["quant", "cc"], default="quant")
    ap.add_argument("--sd-prior", type=float, default=None,
                    help="效应先验 sd；默认 定量 0.15 / 病例对照 0.20")
    ap.add_argument("--min-snps", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in (args.trait1, args.trait2):
        if not Path(p).exists():
            return C.fail(f"文件不存在: {p}", code=C.EXIT_INPUT, as_json=args.json)

    sd_prior = args.sd_prior if args.sd_prior is not None else (0.15 if args.type == "quant" else 0.20)

    try:
        a = load_trait(args.trait1)
        b = load_trait(args.trait2)
    except (OSError, ValueError) as e:
        return C.fail(f"读取失败：{e}", code=C.EXIT_INPUT, as_json=args.json)

    snps = sorted(set(a) & set(b))
    if len(snps) < args.min_snps:
        # 完整区域不足 → 不得声称共定位（PROTOCOL 硬约束）
        return C.fail(
            f"两性状共同 SNP 仅 {len(snps)} 个（< {args.min_snps}）。"
            f"共定位需完整未筛选的区域统计量；**不得只取显著子集**，也不得在此情形下声称共定位。",
            code=C.EXIT_INPUT, as_json=args.json,
            n_common_snps=len(snps))

    # 主分析
    w_main = (sd_prior * args.sd_y) ** 2
    main = run_coloc(a, b, snps, args.p1, args.p2, args.p12, w_main)

    # 敏感性矩阵
    p12_list = ([float(x) for x in args.p12_list.split(",")]
                if args.p12_list else [args.p12])
    sdy_list = ([float(x) for x in args.sdy_list.split(",")]
                if args.sdy_list else [args.sd_y])
    sensitivity = []
    for p12 in p12_list:
        for sdy in sdy_list:
            w = (sd_prior * sdy) ** 2
            r = run_coloc(a, b, snps, args.p1, args.p2, p12, w)
            sensitivity.append({"p12": p12, "sdY": sdy,
                                "PP.H3": r["PP.H3"], "PP.H4": r["PP.H4"],
                                "verdict": r["verdict"]})

    report = {
        "ok": True,
        "summary": f"共同 SNP {len(snps)} 个；PP.H4={main['PP.H4']:.4f}（{main['verdict']}）"
                   f"｜先验 p1=p2={args.p1:g}, p12={args.p12:g}, sdY={args.sd_y:g}",
        "caliber": {
            "method": "Wakefield ABF + Giambartolomei 2014 单因果假设分解",
            "type": args.type,
            "sd_prior": sd_prior,
            "sdY_main": args.sd_y,
            "W_prior_variance": w_main,
            "priors": {"p1": args.p1, "p2": args.p2, "p12": args.p12},
            "n_common_snps": len(snps),
            "single_causal_assumption": True,
            "no_ld_multisignal": True,
            "note": ("单因果假设 + 无 LD 多信号精细定位，是明确局限；"
                     "若区域存在多信号须改用 coloc.susie。PP.H4>0.8 常作主判据"),
        },
        "main": main,
        "sensitivity_matrix": sensitivity,
    }
    if args.out:
        C.write_json(args.out, report)

    C.log(f"共同 SNP {len(snps)}；PP.H4={main['PP.H4']:.4f} ({main['verdict']})")
    C.log(f"  PP.H0={main['PP.H0']:.4f} H1={main['PP.H1']:.4f} "
          f"H2={main['PP.H2']:.4f} H3={main['PP.H3']:.4f}")
    for s in sensitivity:
        C.log(f"  敏感性 p12={s['p12']:g} sdY={s['sdY']:g} → PP.H4={s['PP.H4']:.4f}")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
