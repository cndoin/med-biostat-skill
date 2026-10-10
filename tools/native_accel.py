"""原生加速层：调用魔改版 biostat-vcf（Rust）做 GWAS 汇总统计质控。

这个模块解决一个具体问题：biostat-copilot 的 Tier0 脚本是纯标准库的，
在 Claude Code / Codex 的裸 Python 里照跑没问题，但**性能**上，
千万行汇总统计逐行解析要几十秒到几分钟，而且每个项目都要自己写一遍质控。

魔改版 biostat-vcf 提供了一个流式 + rayon 并行的 `sumstats` 子命令，
实测 300 万行 / 190 MB：Rust 16 线程 922 ms，纯 Python 3431 ms（3.7×），
且 λGC 与 Python 实现**逐位一致**。

设计约束（与 `tools/wrapper_core.py` 同一套不变量）：
1. **能力清单前置** —— 启动即检查二进制存在与可执行，不盲目调用。
2. **不 mock** —— 没有二��制就 SKIP，绝不返回假数字。
3. **优雅降级** —— 原生层不可用时，调用方回退到 `scripts/qc_gwas.py`。
4. **数值可核** —— 每次调用返回 `implementation` 字段，让下游能区分
   「Rust 算的」与「Python 算的」，避免混用后无法复现。
5. **原样透传** —— `--native-args` 直通二进制，保证「包装层不是功能上限」。

用法：
    python tools/native_accel.py --list
    python tools/native_accel.py --input gwas.tsv --json
    python tools/native_accel.py official -- --help
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
NATIVE_DIR = PKG / "vendor" / "native"
BINARY_NAME = "biostat-vcf.exe" if os.name == "nt" else "biostat-vcf"

EXIT_OK, EXIT_ERROR, EXIT_INPUT = 0, 1, 2

#: 上游 `rsomics_common::ExitCode` 的固定映射（exit.rs，值永不变号）。
#: ⚠️ 注意 3 是 UpstreamError，**不是**「质控发现阻断问题」——
#: 「质控发现阻断问题」走的是 Validation::Invalid → InvalidInput → **1**。
#: 早期把这个写反，会让所有阻断性问题被读成「上游库出错」。
BINARY_EXIT_STATUS = {
    0: "OK",
    1: "BLOCKED",     # InvalidInput：校验/质控发现阻断性问题
    2: "CONFIG",      # ConfigError：参数或输入配置错误（含空文件）
    3: "UPSTREAM",    # UpstreamError：上游库返回错误
    4: "IO",          # IoError：文件读写失败
}

#: λGC 口径标签。**必须**随结果一起传下去，否则下游无法区分
#: 本层（Wald 均值口径）与 `scripts/qc_gwas.py`（p 值中位数口径）。
LAMBDA_CALIBER = "wald-mean-chi2/theta=0.4549364231195727"

#: 原生层声明的能力清单。启动即比对二进制自报的子命令，
#: 差集就是「文档写了但二进制没有」的能力——这类漂移必须暴露而不是假装有。
#: 上游 rsomics-vcf v0.6.0 共 10 个子命令（annotate/filter/head/index/norm/
#: query/reheader/setgt/validate/view），本次魔改**只做加法**新增 sumstats，
#: 合计 11 个。clap 自动生成的 `help` 不计入能力清单。
UPSTREAM_COMMANDS = [
    "annotate", "filter", "head", "index", "norm",
    "query", "reheader", "setgt", "validate", "view",
]
ADDED_COMMANDS = ["sumstats"]
DECLARED_CAPABILITIES = UPSTREAM_COMMANDS + ADDED_COMMANDS

#: clap 自动注入、不算「未声明能力」的伪子命令。
CLAP_BUILTINS = {"help"}


def binary_path() -> Path | None:
    """定位原生二进制；找不到返回 None（调用方转 SKIP）。"""
    override = os.environ.get("BIOSTAT_NATIVE_BIN")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    candidate = NATIVE_DIR / BINARY_NAME
    return candidate if candidate.is_file() else None


def available() -> bool:
    return binary_path() is not None


def probe() -> dict:
    """能力探测：二进制是否存在 + 实际自报了哪些子命令。

    为什么要真跑一次 `--help` 而不是相信常量：
    常量会与二进制漂移（重新编译、换版本、复制了旧产物），
    而漂移后的能力表会让调用方以为某功能可用。
    """
    path = binary_path()
    if path is None:
        return {
            "available": False,
            "reason": f"未找到原生二进制（期望 {NATIVE_DIR / BINARY_NAME}）",
            "fallback": "scripts/qc_gwas.py（纯标准库，慢但零依赖）",
        }
    cp = subprocess.run([str(path), "--help"], capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    reported: list[str] = []
    capture = False
    for line in (cp.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("Commands:"):
            capture = True
            continue
        if capture:
            if not stripped or stripped.endswith(":") or line.startswith("Options"):
                break
            reported.append(stripped.split()[0])
    missing = [c for c in DECLARED_CAPABILITIES if c not in reported]
    extra = [c for c in reported
             if c not in DECLARED_CAPABILITIES and c not in CLAP_BUILTINS]
    return {
        "available": True,
        "binary": str(path),
        "reported_commands": reported,
        "declared_but_missing": missing,
        "undeclared_present": extra,
        "capability_drift": bool(missing),
    }


def _extract_report(envelope: dict) -> dict | None:
    """从两种信封里取出 CommandOutput 层。

    成功：`{"status":"ok","result":{"command","report","diagnostics"}}` → stdout
    失败：`{"status":"error","error":{...},"report":{"command","report",...}}` → **stderr**
    （上游把错误信封写到 stderr，只读 stdout 会把「干净失败」误判成崩溃）
    """
    body = envelope.get("result") or envelope.get("report")
    return body if isinstance(body, dict) else None


def sumstats(input_path: str, *, threads: int | None = None,
             max_rows: int | None = None, extra_args: list[str] | None = None) -> tuple[dict, int]:
    """调用原生 sumstats。返回 (归一化结果, 二进制退出码)。

    退出码：0 通过 / 1 校验发现阻断问题 / 2 配置错 / 3 上游错 / 4 IO 错。

    归一化结果字段：
      ok            —— 是否 exit 0
      status        —— OK / BLOCKED / CONFIG / UPSTREAM / IO / CRASH / SKIP
      exit_code     —— 二进制原始退出码
      implementation—— 固定为 rust:biostat-vcf/sumstats（数值出处）
      lambda_caliber—— λGC 口径标签（防止与 qc_gwas.py 的口径混用）
      report        —— 上游 Report（rows_read/rows_used/lambda_gc/maf_binned/issues…）
      diagnostics   —— 上游诊断列表
    """
    path = binary_path()
    if path is None:
        return {
            "ok": False,
            "status": "SKIP",
            "reason": "原生二进制不存在",
            "fallback": "scripts/qc_gwas.py",
        }, EXIT_INPUT

    argv = [str(path), "sumstats", "--json"]
    if threads is not None:
        argv += ["--threads", str(threads)]
    if max_rows is not None:
        argv += ["--max-rows", str(max_rows)]
    argv += list(extra_args or [])
    argv.append(str(input_path))

    cp = subprocess.run(argv, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    envelope = None
    for stream in (cp.stdout, cp.stderr):   # 成功看 stdout，失败看 stderr
        try:
            envelope = json.loads(stream)
            break
        except (json.JSONDecodeError, TypeError):
            continue

    status = BINARY_EXIT_STATUS.get(cp.returncode, "FAILED")
    if envelope is None:
        # 二进制崩了却没能给 JSON —— 这属于「失败被伪装成别的形状」，
        # 必须原样上报，不能包装成空报告。
        return {
            "ok": False,
            "status": "CRASH",
            "exit_code": cp.returncode,
            "reason": "原生二进制未输出可解析的 JSON",
            "stdout_head": cp.stdout[:400],
            "stderr_tail": cp.stderr[-400:],
        }, cp.returncode or EXIT_ERROR

    body = _extract_report(envelope) or {}
    result = {
        "ok": cp.returncode == 0,
        "status": status,
        "exit_code": cp.returncode,
        "implementation": "rust:biostat-vcf/sumstats",
        "lambda_caliber": LAMBDA_CALIBER,
        "report": body.get("report"),
        "diagnostics": body.get("diagnostics", []),
    }
    if envelope.get("error"):
        result["error"] = envelope["error"]
    return result, cp.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="原生加速层（魔改版 biostat-vcf）")
    parser.add_argument("--input", help="汇总统计文件")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--probe", action="store_true", help="只做能力探测")
    parser.add_argument("--list", action="store_true", help="列出可用能力")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("passthrough", nargs="*", help="原样透传给二进制")
    args = parser.parse_args()

    if args.list or args.probe:
        report = probe()
        report["declared_capabilities"] = DECLARED_CAPABILITIES
        report["upstream_commands"] = UPSTREAM_COMMANDS
        report["added_commands"] = ADDED_COMMANDS
        print(json.dumps(report, ensure_ascii=False, indent=2))
        # 能力漂移必须让命令失败，否则「文档写了但没有」会被当成正常。
        return EXIT_ERROR if report.get("capability_drift") else EXIT_OK

    # 兜底透传必须**先于** `--input` 前置校验：它是「包装层不是功能上限」的实现方式，
    # 上游 10 个子命令大多不需要 `--input`（如 `view --help`）。
    # 早期把 `--input` 校验放在前面，导致 official 入口被自己的 parser 拒掉 ——
    # 兜底入口形同虚设（与 `wrapper_core` 记过的同一个坑）。
    if args.passthrough and args.passthrough[0] == "official":
        path = binary_path()
        if path is None:
            print(json.dumps({"ok": False, "error": "原生二进制不存在"},
                             ensure_ascii=False), file=sys.stderr)
            return EXIT_INPUT
        cp = subprocess.run([str(path), *args.passthrough[1:]],
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        sys.stdout.write(cp.stdout)
        if cp.stderr:
            sys.stderr.write(cp.stderr)
        return cp.returncode

    if not args.input:
        print(json.dumps({"ok": False, "error": "需要 --input"},
                         ensure_ascii=False), file=sys.stderr)
        return EXIT_INPUT

    payload, code = sumstats(args.input, threads=args.threads,
                             max_rows=args.max_rows,
                             extra_args=args.passthrough or None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
