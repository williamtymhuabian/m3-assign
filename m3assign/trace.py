"""复现训练采样轨迹：把 train.py 的随机顺序重放成 step -> 样本行。

train.py 只用 Python random 打乱 shard 组顺序、用 numpy 打乱组内行序，
其余随机性（翻转增强）走 torch RNG。因此给定 state.json 里的 seed/参数，
可以离线精确还原每个 step 采到的 512 行，用于看板的“训练当前帧”。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .data import SHARD_DIR, shard_ids

_ROW_COUNT: dict[int, int] = {}
_META: dict[int, list[dict]] = {}
_OFFSETS: dict[int, int] | None = None
_TRACE: dict[tuple[str, float], list[dict]] = {}


def shard_rows(sid: int) -> int:
    if sid not in _ROW_COUNT:
        with open(SHARD_DIR / f"shard_{sid:05d}.meta.jsonl", encoding="utf-8") as fh:
            _ROW_COUNT[sid] = sum(1 for _ in fh)
    return _ROW_COUNT[sid]


def shard_meta(sid: int) -> list[dict]:
    if sid not in _META:
        rows = []
        with open(SHARD_DIR / f"shard_{sid:05d}.meta.jsonl", encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        _META[sid] = rows
    return _META[sid]


def offsets() -> dict[int, int]:
    global _OFFSETS
    if _OFFSETS is None:
        out, acc = {}, 0
        for sid in shard_ids():
            out[sid] = acc
            acc += shard_rows(sid)
        _OFFSETS = out
    return _OFFSETS


def locate(row: int) -> tuple[int, int]:
    o = offsets()
    for sid, start in o.items():
        if start <= row < start + shard_rows(sid):
            return sid, row - start
    raise KeyError(f"row {row} out of range")


def _global_rows(ids: list[int]) -> np.ndarray:
    o = offsets()
    return np.concatenate(
        [o[s] + np.arange(shard_rows(s), dtype=np.int64) for s in ids]
    )


def build_trace(args: dict) -> list[dict]:
    tr_ids = shard_ids()[: int(args["train_shards"])]
    batch = int(args["batch"])
    epochs = int(args["epochs"])
    seed = int(args["seed"])
    group_size = int(args.get("shard_group") or 0)

    random.seed(seed)
    np.random.seed(seed)
    steps: list[dict] = []
    if group_size:
        groups = [tr_ids[i : i + group_size] for i in range(0, len(tr_ids), group_size)]
        for epoch in range(1, epochs + 1):
            order = list(groups)
            random.shuffle(order)
            for g in order:
                glob = _global_rows(g)
                perm = np.random.permutation(glob.size)
                for s in range(0, glob.size, batch):
                    steps.append({"epoch": epoch, "rows": glob[perm[s : s + batch]]})
    else:
        glob = _global_rows(tr_ids)
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(glob.size)
            for s in range(0, glob.size, batch):
                steps.append({"epoch": epoch, "rows": glob[perm[s : s + batch]]})
    return steps


def trace_for(run_dir: Path) -> tuple[list[dict], dict]:
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    key = (str(run_dir), float(state.get("started_at") or 0))
    if key not in _TRACE:
        steps = build_trace(state["args"])
        if len(steps) != int(state["total_steps"]):
            raise ValueError(
                f"轨迹重放步数 {len(steps)} 与训练记录 {state['total_steps']} 不一致"
            )
        _TRACE.clear()
        _TRACE[key] = steps
    return _TRACE[key], state
