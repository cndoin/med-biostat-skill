# VENDORED-FILE —— 请勿直接编辑；改源文件后重跑 tools/vendor_sync.py
# 来源: medomics（自建）
# 原路径: medomics/core/audit.py
# 许可: 自建/内部
# 说明: 溯源审计（manifest 落盘）
# 源 sha256: 34d8fe5d0e1c44e5
# VENDORED-FILE-END
"""溯源与运行审计。

借鉴 BioMaster 的"可审计产物"思想：每一次分析都落盘一份 manifest，
记录输入指纹、参数、软件版本、耗时与输出指纹。任何人拿到 run 目录，
都能回答"这个数字是怎么来的"。
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contract import fingerprint


@dataclass
class Step:
    name: str
    started_at: str
    params: Dict[str, Any] = field(default_factory=dict)
    input_fingerprints: Dict[str, str] = field(default_factory=dict)
    output_fingerprints: Dict[str, str] = field(default_factory=dict)
    duration_sec: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "params": self.params,
            "input_fingerprints": self.input_fingerprints,
            "output_fingerprints": self.output_fingerprints,
            "duration_sec": round(self.duration_sec, 4),
            "notes": list(self.notes),
        }


class Run:
    """一次分析运行的审计容器。"""

    def __init__(self, name: str, outdir: str | Path):
        self.name = name
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.steps: List[Step] = []
        self._t0 = time.time()
        self.meta = {
            "run": name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }

    def step(self, name: str, params: Optional[dict] = None,
             inputs: Optional[dict] = None) -> Step:
        s = Step(
            name=name,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            params=dict(params or {}),
            input_fingerprints={k: fingerprint(v) for k, v in (inputs or {}).items()},
        )
        self.steps.append(s)
        s._t = time.time()
        return s

    def finish(self, step: Step, outputs: Optional[dict] = None,
               notes: Optional[List[str]] = None) -> Step:
        step.output_fingerprints = {k: fingerprint(v) for k, v in (outputs or {}).items()}
        step.duration_sec = time.time() - getattr(step, "_t", time.time())
        if notes:
            step.notes.extend(notes)
        return step

    def save(self) -> Path:
        payload = {
            "meta": self.meta,
            "total_sec": round(time.time() - self._t0, 3),
            "steps": [s.to_dict() for s in self.steps],
        }
        path = self.outdir / "manifest.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def save_table(self, df, filename: str) -> Path:
        path = self.outdir / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path


def guard(condition: bool, message: str) -> None:
    """运行期硬闸门：不满足即中止，而不是带着问题继续产出结果。"""
    if not condition:
        raise RuntimeError("运行期闸门未通过：%s" % message)
