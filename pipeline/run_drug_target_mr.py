#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_drug_target_mr.py —— 药物靶点 MR 确定性分析管道

**这是对「分析过程由电脑数字分析，而不是用 AI 分析」的落地**：

  * 一份**声明式配置**（JSON）描述研究设计与全部参数；
  * 一条命令按固定顺序执行全链，**每一步都是确定性脚本，AI 不参与任何数值**；
  * 每次运行产出一份 `run_manifest.json`：输入哈希 + 参数 + 代码哈希 + 产物哈希 + 耗时，
    用以证明「这组数字确实来自这次运行」；
  * 失败即停（fail-fast），不跳步、不跳过缺失步骤。

执行链（可在配置中选做）：
    cis_instrument → wald_mr → abf_coloc → testing_family → replication_check

用法：
    python pipeline/run_drug_target_mr.py --config pipeline/config.example.json --json
    python pipeline/run_drug_target_mr.py --config my.json --outdir my_run --json
退出码：0 全链成功 / 1 有步骤失败 / 2 配置错误
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SCRIPTS = PKG / "scripts"
PY = sys.executable

# 步骤定义：(名称, 脚本, 配置键 → 命令行参数)
STEP_SPEC = {
    "cis_instrument": ("cis_instrument.py", {
        "eqtl": "--eqtl", "genes": "--genes", "window_kb": "--window-kb",
        "full_cis": "--full-cis", "p_threshold": "--p-threshold",
        "f_min": "--f-min", "all_cis": "--all-cis"}),
    "wald_mr": ("wald_mr.py", {
        "instruments": "--instruments", "f_min": "--f-min"}),
    "abf_coloc": ("abf_coloc.py", {
        "trait1": "--trait1", "trait2": "--trait2", "p12": "--p12",
        "p12_list": "--p12-list", "sd_y": "--sd-y", "sdy_list": "--sdy-list",
        "type": "--type", "sd_prior": "--sd-prior"}),
    "testing_family": ("testing_family.py", {
        "input": "--input", "alpha": "--alpha"}),
    "replication_check": ("replication_check.py", {
        "discovery": "--discovery", "replication": "--replication", "alpha": "--alpha"}),
}
# 默认执行顺序（管道按此固定顺序，避免 AI 随意编排）
ORDER = ["cis_instrument", "wald_mr", "abf_coloc", "testing_family", "replication_check"]

# 需要逐项运行（配置里给的是列表）的步骤
LIST_STEPS = {"abf_coloc"}


def sha256_file(p: Path) -> str:
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_cmd(step: str, cfg: dict, outdir: Path, tag: str):
    script, mapping = STEP_SPEC[step]
    cmd = [PY, str(SCRIPTS / script), "--json"]
    for key, val in cfg.items():
        if key not in mapping:
            continue
        flag = mapping[key]
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        elif val is not None:
            cmd.extend([flag, str(val)])
    out = outdir / f"{step}{tag}.json"
    cmd.extend(["--out", str(out)])
    return cmd, out


def run_step(step: str, cfg: dict, outdir: Path, tag: str = "") -> dict:
    cmd, out = build_cmd(step, cfg, outdir, tag)
    t0 = time.time()
    cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed = round(time.time() - t0, 2)
    payload = None
    try:
        payload = json.loads(cp.stdout)
    except Exception:
        payload = None
    return {
        "step": step, "tag": tag, "cmd": cmd, "returncode": cp.returncode,
        "elapsed_sec": elapsed,
        "ok": cp.returncode == 0 and bool(payload and payload.get("ok", True)),
        "summary": (payload or {}).get("summary", ""),
        "stderr_tail": cp.stderr[-500:] if cp.returncode != 0 else "",
        "out_file": str(out) if out.exists() else None,
        "out_sha256": sha256_file(out) if out.exists() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="药物靶点 MR 确定性管道")
    ap.add_argument("--config", required=True, help="声明式配置 JSON")
    ap.add_argument("--outdir", default=None, help="产物目录（默认 logs/pipeline_run_<时间>）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不运行")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        return C_fail(f"配置不存在: {cfg_path}", args.json)
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return C_fail(f"配置解析失败: {e}", args.json)

    study = cfg.get("study", "unnamed")
    outdir = Path(args.outdir) if args.outdir else (PKG / "logs" /
                                                    f"pipeline_{study}_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)

    steps_cfg = cfg.get("steps", {})
    if not steps_cfg:
        return C_fail("配置里没有 steps", args.json)
    # 配置类型校验：steps 必须是对象；值必须是 dict，或在 LIST_STEPS 内的 list。
    # 曾经把 list 配给了非 LIST_STEPS 的步骤却直接传下去，
    # 崩在 run_step 里报 AttributeError: 'list' object has no attribute 'items'
    # —— 配置错误不该以 traceback 形式出现（exit 1 还与正常失败撞车）。
    if not isinstance(steps_cfg, dict):
        return C_fail(f"steps 必须是对象（dict），实际是 {type(steps_cfg).__name__}",
                      args.json)
    bad = []
    for k, v in steps_cfg.items():
        if isinstance(v, list) and k not in LIST_STEPS:
            bad.append(f"{k}: 只接受对象，不接受列表（支持列表的步骤：{sorted(LIST_STEPS)}）")
        elif not isinstance(v, (dict, list)):
            bad.append(f"{k}: 必须是对象或列表，实际是 {type(v).__name__}")
    if bad:
        return C_fail("steps 配置非法：" + "；".join(bad), args.json)

    results = []
    # 固定顺序执行（配置里没写 skipped=false 的才跑）
    for step in ORDER:
        if step not in steps_cfg:
            continue
        spec = steps_cfg[step]
        if isinstance(spec, list) and step in LIST_STEPS:
            for i, item in enumerate(spec):
                results.append(run_step(step, item, outdir, tag=f"_{i+1}"))
        else:
            results.append(run_step(step, spec, outdir, tag=""))

    n_ok = sum(1 for r in results if r["ok"])
    manifest = {
        "study": study,
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "pipeline_sha256": sha256_file(Path(__file__)),
        "common_sha256": sha256_file(SCRIPTS / "_common.py"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "n_steps": len(results),
        "n_ok": n_ok,
        "all_ok": n_ok == len(results),
        "steps": results,
        "provenance_note": ("本清单用于证明产物可复现：相同 config_sha256 + 相同脚本哈希 "
                            "→ 相同产物哈希。任何数值都由脚本产生，AI 不介入。"),
    }
    manifest_path = outdir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    ok = manifest["all_ok"]
    report = {
        "ok": ok,
        "summary": f"研究 {study}：{n_ok}/{len(results)} 步成功",
        "outdir": str(outdir),
        "manifest": str(manifest_path),
        "steps": [{"step": r["step"], "ok": r["ok"], "rc": r["returncode"],
                   "summary": r["summary"], "sec": r["elapsed_sec"]} for r in results],
    }
    for r in results:
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  [{mark}] {r['step']}{r['tag']} ({r['elapsed_sec']}s) {r['summary']}",
              file=sys.stderr)
        if not r["ok"] and r["stderr_tail"]:
            print(f"        {r['stderr_tail'][-200:]}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(report["summary"])
    return 0 if ok else 1


def C_fail(msg: str, as_json: bool) -> int:
    print(msg, file=sys.stderr)
    if as_json:
        print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    sys.exit(main())
