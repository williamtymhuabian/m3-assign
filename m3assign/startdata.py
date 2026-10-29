"""只保留 reason=="start" 行的 shard 视图。

初始承诺实验用：一条 episode 只取第一帧，100k 快照里约 9.4% 是 start 行。
逐 shard 读取后立刻丢弃，避免把 13 GB 地图常驻主存。
"""
from __future__ import annotations

import json

import numpy as np
from safetensors import safe_open

from .data import SHARD_DIR

FIELDS = (
    "geo",
    "heat",
    "pose",
    "peak",
    "travel",
    "teacher",
    "teacher_core",
    "contract",
    "valid_k",
    "valid_m",
)
INT16 = ("teacher", "teacher_core", "contract")


class StartStore:
    """与 data.Store 同属性名的最小视图，cpu_tensors() 可直接使用。"""

    def __init__(self, ids: list[int], reason: str = "start"):
        chunks: dict[str, list[np.ndarray]] = {k: [] for k in FIELDS}
        metas: list[dict] = []
        base = 0
        for sid in ids:
            sp = SHARD_DIR / f"shard_{int(sid):05d}.safetensors"
            mp = SHARD_DIR / f"shard_{int(sid):05d}.meta.jsonl"
            meta = [json.loads(line) for line in open(mp, encoding="utf-8")]
            sel = np.asarray(
                [i for i, m in enumerate(meta) if m.get("reason") == reason], dtype=np.int64
            )
            if sel.size == 0:
                continue
            with safe_open(str(sp), framework="numpy") as fh:
                for k in FIELDS:
                    chunks[k].append(fh.get_tensor(k)[sel])
            for i in sel:
                m = dict(meta[int(i)])
                m["row"] = base
                base += 1
                metas.append(m)
        for k in FIELDS:
            arr = np.concatenate(chunks[k]) if chunks[k] else np.zeros((0,), dtype=np.float32)
            self.__dict__[k] = arr.astype(np.int16) if k in INT16 else arr
        self.n = base
        self.meta = metas
