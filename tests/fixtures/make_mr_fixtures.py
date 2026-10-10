#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_mr_fixtures.py —— 药物靶点 MR 链的合成数据（确定性，纯标准库）

产物：
  eqtl.tsv             模拟 cis-eQTL 汇总统计（含一个强工具、一个弱工具、一个回文 SNP）
  genes.tsv            基因边界
  mr_input.tsv         Wald ratio MR 输入（工具已选好）
  trait1_region.tsv    共定位：性状1 区域统计（真共享信号）
  trait2_region.tsv    共定位：性状2 区域统计（与性状1 共享因果变异）
  trait1_null.tsv      共定位：与性状1 无关的区域（阴性对照）
  trait2_null.tsv
  tests.tsv            检验家族分层输入
"""

from __future__ import annotations

import random
from pathlib import Path

SEED = 20261010


def _rand_region(rng, n, causal_idx, causal_beta, base_chrom="chr1", base_pos=1_000_000):
    """造一个区域：n 个 SNP，其中 causal_idx 处为真信号，其余为噪声。"""
    rows = []
    for i in range(n):
        pos = base_pos + i * 500
        if i == causal_idx:
            beta = causal_beta
            se = 0.04
        else:
            beta = rng.gauss(0, 0.02)
            se = 0.05
        rows.append((f"rs{base_pos + i}", base_chrom, pos, beta, se))
    return rows


def build(outdir: str) -> dict:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # ---- cis-eQTL：PTGDS 强工具 / 弱工具 / 回文 SNP ----
    with open(d / "eqtl.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("gene\ttissue\tchrom\tpos\tpval\tbeta\tse\tref\talt\taf\n")
        # PTGDS Whole_Blood：强工具（P=1e-30, beta=0.5, se=0.04 → F=156）
        fh.write("PTGDS\tWhole_Blood\tchr9\t136900000\t1e-30\t0.5\t0.04\tA\tG\t0.32\n")
        fh.write("PTGDS\tWhole_Blood\tchr9\t136900500\t0.001\t0.05\t0.06\tC\tT\t0.20\n")
        # 回文 SNP，靶位点最强但必须被排除
        fh.write("PTGDS\tWhole_Blood\tchr9\t136901000\t1e-40\t0.9\t0.05\tA\tT\t0.45\n")
        # PTGDS Artery_Tibial：弱工具（P=1e-9 通过，但 F 低）
        fh.write("PTGDS\tArtery_Tibial\tchr9\t136900000\t1e-9\t0.10\t0.05\tA\tG\t0.32\n")
        # CACNA1C：另一基因（窗口外，应被排除）
        fh.write("CACNA1C\tArtery_Tibial\tchr12\t2100000\t1e-20\t0.4\t0.04\tG\tT\t0.28\n")
        # PTGDS 窗口外的变异（不应入选）
        fh.write("PTGDS\tWhole_Blood\tchr9\t139000000\t1e-20\t0.4\t0.04\tC\tA\t0.30\n")
    with open(d / "genes.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("gene\tchrom\tstart\tend\n")
        fh.write("PTGDS\tchr9\t136895000\t136905000\n")
        fh.write("CACNA1C\tchr12\t2100000\t2200000\n")

    # ---- Wald ratio 输入 ----
    with open(d / "mr_input.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("snp\tgene\ttissue\tbeta_exposure\tse_exposure\tbeta_outcome\tse_outcome\n")
        # 强工具：F=(0.5/0.04)²=156，Wald=0.25/0.5=0.5
        fh.write("rs136900000\tPTGDS\tWhole_Blood\t0.5\t0.04\t0.25\t0.08\n")
        # 弱工具：F=(0.10/0.05)²=4 < 10
        fh.write("rs136900000\tPTGDS\tArtery_Tibial\t0.10\t0.05\t0.05\t0.08\n")
        # 暴露效应为 0 → 不可估
        fh.write("rs0\tPTGDS\tX\t0.0\t0.05\t0.10\t0.08\n")
        # 阴性：效应方向相反且不显著
        fh.write("rsNeg\tPTGDS\tWhole_Blood\t0.4\t0.05\t-0.01\t0.09\n")

    # ---- 共定位：真共享信号 ----
    n = 40
    t1 = _rand_region(rng, n, causal_idx=20, causal_beta=0.45)
    r2 = random.Random(SEED + 1)
    # 性状2：同一因果变异位置，独立噪声
    t2 = []
    for i, (snp, ch, pos, beta, se) in enumerate(t1):
        if i == 20:
            beta2, se2 = 0.30, 0.04      # 同一因果变异 → 应高 PP.H4
        else:
            beta2, se2 = r2.gauss(0, 0.02), 0.05
        t2.append((snp, ch, pos, beta2, se2))
    # 阴性对照：性状2 信号在同一区域但落在**不同**位置（causal_idx=35）
    t2n = []
    r3 = random.Random(SEED + 2)
    for i, (snp, ch, pos, beta, se) in enumerate(t1):
        if i == 35:
            beta2, se2 = 0.30, 0.04      # 不同因果变异 → 应低 PP.H4 / 高 PP.H3
        else:
            beta2, se2 = r3.gauss(0, 0.02), 0.05
        t2n.append((snp, ch, pos, beta2, se2))

    def write_region(name, rows):
        with open(d / name, "w", encoding="utf-8", newline="") as fh:
            fh.write("snp\tbeta\tse\n")
            for snp, ch, pos, beta, se in rows:
                fh.write(f"{snp}\t{beta:.6f}\t{se:.6f}\n")

    write_region("trait1_region.tsv", t1)
    write_region("trait2_region.tsv", t2)      # 共享
    write_region("trait2_null.tsv", t2n)       # 不共享

    # ---- 外部复现检查 ----
    with open(d / "disc.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("id\tbeta\tse\tp\tancestry\tn_case\tn_control\n")
        fh.write("PTGDS\t0.25\t0.08\t0.002\tEUR\t1000\t409535\n")
        fh.write("CACNA1C\t0.10\t0.05\t0.05\tEUR\t1000\t409535\n")
    with open(d / "rep.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("id\tbeta\tse\tp\tancestry\tn_case\tn_control\n")
        # PTGDS：同向且显著，但对照数与发现期完全相同 → 应报样本重叠
        fh.write("PTGDS\t0.20\t0.09\t0.02\tEUR\t800\t409535\n")
        # CACNA1C：方向翻转 + ancestry 不同 → 应报两项
        fh.write("CACNA1C\t-0.08\t0.06\t0.18\tEAS\t900\t300000\n")

    # ---- 检验家族 ----
    with open(d / "tests.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("test_id\tp\tfamily\n")
        fh.write("PTGDS_blood\t0.0029\tmain\n")
        fh.write("CACNA1C_artery\t0.031\tmain\n")
        fh.write("PTGDS_aorta\t0.012\tsecondary\n")
        fh.write("PTGDS_coronary\t0.041\tsecondary\n")
        fh.write("PTGDS_wholeblood\t0.008\tsecondary\n")
        fh.write("PTGDS_explore1\t0.55\texploratory\n")
        fh.write("PTGDS_explore2\t0.73\texploratory\n")

    return {"dir": str(d), "n_region_snps": n}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(build(sys.argv[1] if len(sys.argv) > 1 else "./mr_fixtures"),
                     ensure_ascii=False, indent=2))
