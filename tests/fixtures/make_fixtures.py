#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_fixtures.py —— 生成确定性合成数据（无第三方依赖）

产物（写入指定目录）：
  gwas_full.tsv.gz   模拟 GWAS 汇总统计（#chrom/pos/ref/alt/pval/beta/sebeta/af_alt）
  snp_filter.tsv     模拟精细定位子集（region/chromosome/position/allele1/allele2/p/beta/maf）
  endpoints.tsv      两个端点（对照数相同 → 触发 S4③ 样本重叠）
  genes.tsv          基因注释（chrom/start/end/gene，含孟德尔白内障基因）
  regions.tsv        待注释区间

固定随机种子 → 数值可复现。
"""

from __future__ import annotations

import gzip
import random
from pathlib import Path

SEED = 20261010
HDR_FULL = ["#chrom", "pos", "ref", "alt", "pval", "beta", "sebeta", "af_alt"]

# 固定的显著位点（用于阳性对照与位点计数测试）
SIG = [
    ("1", 1_000_000, "T", "C", 1e-10, 0.31),
    ("1", 1_000_400, "A", "G", 1e-9, 0.28),     # 与上一位点同 1 Mb 窗口
    ("2", 2_000_000, "C", "T", 5e-9, 0.22),
    ("15", 73_931_230, "G", "A", 1e-12, 0.45),  # 共病位点（LOXL1 位置示意）
    ("3", 50_000, "G", "T", 3e-8, 0.19),
]
# 需要 MAF 折算验证的变异（ALT 为常见等位，af_alt>0.5）
MAF_TRAP = ("7", 1_234_567, "A", "T", 2e-7, 0.90)   # maf 应为 0.10


def make_gwas_rows(n_random: int = 4000):
    rnd = random.Random(SEED)
    rows = []
    for chrom, pos, ref, alt, p, beta in SIG:
        rows.append([chrom, pos, ref, alt, p, beta, 0.05, 0.30])
    c, pos, ref, alt, p, beta = MAF_TRAP
    rows.append([c, pos, ref, alt, p, beta, 0.05, 0.90])
    for i in range(n_random):
        chrom = str(rnd.randint(1, 22))
        pos = rnd.randint(10_000, 200_000_000)
        ra, al = rnd.choice([("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")])
        p = rnd.random()          # 绝大多数不显著
        beta = round(rnd.uniform(-0.3, 0.3), 4)
        af = round(rnd.uniform(0.01, 0.99), 4)
        rows.append([chrom, pos, ra, al, p, beta, 0.05, af])
    return rows


def make_sumstats_rows(n_random: int = 3000):
    """专供原生层（biostat-vcf sumstats）与 Python 参考实现对拍的汇总统计。

    与 `make_gwas_rows` 的区别：这份**刻意注入**每一类质控异常，
    使「异常计数」也有确定期望值，而不只是对拍 λGC 一个数字。
    列名用管线规范词汇 `af_alt`（原生层别名表已对齐），并且不含 gzip，
    便于同时产出 .tsv 与 .tsv.gz 两个变体验证透明解码。
    """
    rnd = random.Random(SEED + 1)
    header = ["chr", "pos", "id", "ref", "alt", "af_alt", "beta", "se", "pval"]
    rows = []

    def push(chrom, pos, af, beta, se, p, i):
        rows.append([chrom, pos, f"rs{i}", "A", "G", af, beta, se, p])

    # 正常行：AF 覆盖五个分档，β/se 给出受控的 χ²
    plan = [(0.005, 1), (0.03, 4), (0.10, 9), (0.30, 4), (0.45, 1)]
    i = 0
    for _ in range(400):
        for af, se in plan:
            push(rnd.randint(1, 22), rnd.randint(10_000, 200_000_000),
                 af, round(rnd.uniform(-0.2, 0.2), 4), se, round(rnd.random(), 6), i)
            i += 1
    n_normal = len(rows)

    # ---- 注入异常（每一项都能被计数，期望值确定） ----
    push("1", 900_000_001, 0.3, 0.1, 0.05, "NaN", i); i += 1   # 非有限 p
    push("1", 900_000_002, 0.3, 0.1, 0.05, 1.7, i); i += 1     # p 越界
    push("1", 900_000_003, 0.3, 0.1, 0.0, 0.01, i); i += 1     # se=0
    push("1", 900_000_004, 0.3, 0.1, -0.5, 0.01, i); i += 1    # se<0
    push("1", 900_000_005, 1.7, 0.1, 0.05, 0.01, i); i += 1    # af 越界 → 无分档
    push("1", 900_000_006, 0.92, 0.1, 0.05, 0.01, i); i += 1   # af_alt>0.5 → 折算 maf=0.08
    push("2", 700_000, 0.3, 0.1, 0.05, 0.01, i); i += 1        # 与下一行重复键
    push("2", 700_000, 0.3, 0.1, 0.05, 0.01, i); i += 1

    expected = {
        "rows_read": len(rows),
        # 可用行 = 2000 正常行 + 折叠后落档的 af_alt=0.92 + 重复对的两行
        "rows_used": n_normal + 3,
        "nonfinite_p": 1,
        "p_out_of_range": 1,
        "nonfinite_beta": 0,
        "nonpositive_se": 2,
        "missing_af": 1,
        "duplicate_keys": 1,
    }
    return header, rows, expected, n_normal


def build(outdir: str) -> dict:
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)

    # 1) 全量 GWAS
    rows = make_gwas_rows()
    with gzip.open(d / "gwas_full.tsv.gz", "wt", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(HDR_FULL) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # 2) 精细定位子集（allele1/allele2 与 ref/alt **同序**；maf 为次等位频率）
    hdr_sub = ["region", "v", "chromosome", "position", "allele1", "allele2",
               "p", "beta", "maf", "cs_size", "cs_avg_r2", "cs_min_r2"]
    sub_rows = []
    for i, (chrom, pos, ref, alt, p, beta) in enumerate(SIG):
        maf = min(0.30, 0.70)
        sub_rows.append([f"region_{i+1}", f"v{i+1}", f"chr{chrom}", pos, ref, alt,
                         p, beta, maf, 25, 0.92, 0.85])
    # MAF 折算验证行：maf(次等位)=0.10，与 af_alt=0.90 折算后一致
    c, pos, ref, alt, p, beta = MAF_TRAP
    sub_rows.append(["region_maf", "vmaf", f"chr{c}", pos, ref, alt, p, beta,
                     0.10, 30, 0.90, 0.80])
    # 一个低纯度大可信集（应被 good_cs 剔除）
    sub_rows.append(["region_bad", "vbad", "chr4", 9_999_999, "A", "C", 1e-6, 0.1,
                     0.2, 21398, 0.05, 0.01])
    with open(d / "snp_filter.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(hdr_sub) + "\n")
        for r in sub_rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    # 3) 端点表（对照数相同 → 样本重叠）
    with open(d / "endpoints.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("endpoint\tn_control\n")
        fh.write("CATARACTSENILE\t409535\n")
        fh.write("CATARACT_OTHER\t409535\n")   # 完全相同 → 疑似共享对照

    # 4) 基因注释（覆盖 chr1:1Mb 与 chr15:73.9Mb）
    with open(d / "genes.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("chrom\tstart\tend\tgene\n")
        fh.write("chr1\t999000\t1002000\tSOX2\n")
        fh.write("chr1\t1003000\t1006000\tLIM2\n")
        fh.write("chr2\t1998000\t2003000\tCRYGC\n")
        fh.write("chr15\t73930000\t73935000\tLOXL1\n")
        fh.write("chr3\t49000\t52000\tGJA8\n")

    # 5) 区间表
    with open(d / "regions.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("chrom\tstart\tend\n")
        fh.write("chr1\t950000\t1050000\n")
        fh.write("chr15\t73900000\t74000000\n")

    # 6) 原生层对拍用的汇总统计（含注入异常；同时产出 .tsv 与 .tsv.gz）
    ss_hdr, ss_rows, ss_expected, _ = make_sumstats_rows()
    ss_text = "\t".join(ss_hdr) + "\n" + "".join(
        "\t".join(str(x) for x in r) + "\n" for r in ss_rows)
    with open(d / "sumstats_qc.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write(ss_text)
    with gzip.open(d / "sumstats_qc.tsv.gz", "wt", encoding="utf-8", newline="") as fh:
        fh.write(ss_text)
    # 管线真实表头形态：`#chrom` + `##` 元信息行（PLINK/bcftools 惯例）
    ss_hash_hdr = ["#chrom", *ss_hdr[1:]]
    ss_hash_text = ("##generated by make_fixtures.py\n"
                    + "\t".join(ss_hash_hdr) + "\n"
                    + "".join("\t".join(str(x) for x in r) + "\n" for r in ss_rows))
    with open(d / "sumstats_hash.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write(ss_hash_text)

    return {
        "dir": str(d),
        "gwas_full": str(d / "gwas_full.tsv.gz"),
        "snp_filter": str(d / "snp_filter.tsv"),
        "endpoints": str(d / "endpoints.tsv"),
        "genes": str(d / "genes.tsv"),
        "regions": str(d / "regions.tsv"),
        "sumstats": str(d / "sumstats_qc.tsv"),
        "sumstats_gz": str(d / "sumstats_qc.tsv.gz"),
        "sumstats_hash": str(d / "sumstats_hash.tsv"),
        "sumstats_expected": ss_expected,
        "n_rows": len(rows),
    }


if __name__ == "__main__":
    import json
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "./fixtures_out"
    print(json.dumps(build(target), ensure_ascii=False, indent=2))
