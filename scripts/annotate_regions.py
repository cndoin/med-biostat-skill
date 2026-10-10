#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""annotate_regions.py —— S3① 权威区间基因注释（红线 R7）

实证背景：用 `gene_most_severe`（最近基因）单基因注释，导致 4 个含孟德尔
白内障基因的区域全部漏掉（SOX2 / LIM2 / CRYGC+CRYGD / MAF），阳性对照被误报为 0。
**区间注释优先，最近基因字段不能当「区域基因」用。**

两种注释源（可并用）：
  * 本地：--annotation <gene_tsv|gtf>，列 chrom/start/end/gene（GTF 自动解析 gene 行）
  * 在线：--online 走 mygene.info 区间查询（需联网；本机实测可达）

用法：
    python annotate_regions.py --regions loci.tsv --annotation genes.tsv --json
    python annotate_regions.py --regions loci.tsv --online --json
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

MYGENE_URL = "https://mygene.info/v3/query"


def load_annotation(path: str):
    """读基因注释表 → {chrom: (starts_sorted, ends_sorted, genes)}。支持 GTF/GFF。"""
    p = Path(path)
    starts = {}
    ivs = {}  # chrom -> list[(start, end, gene)]
    if p.suffix.lower() in (".gtf", ".gff", ".gff3", ".gz"):
        with C.open_text(path) as fh:
            for line in fh:
                if not line or line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9 or f[2].lower() != "gene":
                    continue
                m = re.search(r'gene_name\s+"?([^";]+)"?', f[8]) or \
                    re.search(r'gene_id\s+"?([^";]+)"?', f[8])
                gene = m.group(1) if m else "NA"
                ivs.setdefault(C.normalize_chrom(f[0]), []).append(
                    (int(f[3]), int(f[4]), gene))
    else:
        with C.open_text(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")
            low = {h.lower(): i for i, h in enumerate(header)}
            ci, si, ei = low.get("chrom"), low.get("start"), low.get("end")
            gi = low.get("gene", low.get("gene_name", low.get("symbol")))
            for line in fh:
                f = line.rstrip("\n").split("\t")
                try:
                    ivs.setdefault(C.normalize_chrom(f[ci]), []).append(
                        (int(f[si]), int(f[ei]), f[gi]))
                except (ValueError, IndexError):
                    continue
    out = {}
    for chrom, lst in ivs.items():
        lst.sort()
        out[chrom] = (lst, [x[0] for x in lst])
    return out


def query_local(anno, chrom, start, end):
    if chrom not in anno:
        return []
    lst, starts = anno[chrom]
    # 找所有 start <= end 的区间，再筛其 end >= start
    hi = bisect.bisect_right(starts, end)
    hits = []
    for i in range(hi):
        s, e, g = lst[i]
        if e >= start:
            hits.append(g)
    return sorted(set(hits))


def query_online(chrom, start, end, timeout=30):
    """mygene.info 区间查询。返回基因 symbol 列表。"""
    c = chrom.replace("chr", "")
    url = f"{MYGENE_URL}?q=chr{c}:{start}-{end}&species=human&fields=symbol&size=1000"
    try:
        with urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return sorted({h.get("symbol") for h in data.get("hits", []) if h.get("symbol")})
    except (URLError, OSError, ValueError) as e:
        C.log(f"[WARN] 在线注释失败（{c}:{start}-{end}）：{e}")
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="区间基因注释（区间优先，非最近基因）")
    ap.add_argument("--regions", required=True, help="区间表：chrom/start/end（TSV，含表头）")
    ap.add_argument("--annotation", default=None, help="本地基因注释表（TSV/GTF）")
    ap.add_argument("--online", action="store_true", help="使用 mygene.info 在线查询")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if not args.annotation and not args.online:
        return C.fail("必须提供 --annotation 或 --online（不提供注释源＝口径缺失）",
                      code=C.EXIT_CALIBER, as_json=args.json)

    anno = load_annotation(args.annotation) if args.annotation else {}

    regions = []
    with C.open_text(args.regions) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        low = {h.lower(): i for i, h in enumerate(header)}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            try:
                regions.append({
                    "chrom": C.normalize_chrom(f[low["chrom"]]),
                    "start": int(f[low["start"]]),
                    "end": int(f[low["end"]]),
                })
            except (ValueError, IndexError, KeyError):
                continue

    results = []
    for r in regions:
        genes = query_local(anno, r["chrom"], r["start"], r["end"]) if anno else []
        if args.online and not genes:
            genes = query_online(r["chrom"], r["start"], r["end"])
        results.append({**r, "n_genes": len(genes), "genes": genes})

    report = {
        "ok": True,
        "summary": f"{len(results)} 个区间，共注释到 "
                   f"{sum(x['n_genes'] for x in results)} 个基因",
        "caliber": {
            "method": "区间注释（interval overlap）",
            "source": ("local:" + str(args.annotation)) if args.annotation else "mygene.info",
            "online": args.online,
            "note": "禁止用「最近基因」字段替代区域基因（P11 / R7）",
        },
        "regions": results,
    }
    if args.out:
        C.write_json(args.out, report)
    for x in results:
        C.log(f"  {x['chrom']}:{x['start']}-{x['end']} → {x['n_genes']} 基因")
    return C.emit(report, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
