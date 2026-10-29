"""HM3D-M3 teacher_100k 数据访问层：shard 张量、episode 元数据、IIN 目标图索引。"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file as load_safetensors

# 数据根目录：默认是仓库所在工作区的上一级（本机 m3-assign 就在 VLN 下），
# 对外克隆时用 M3_ROOT 指向自己的 HM3D-M3 拷贝。
ROOT = Path(os.environ.get("M3_ROOT") or Path(__file__).resolve().parents[2]).expanduser()
DATA_ROOT = ROOT / "HM3D-M3"
TEACHER_ROOT = DATA_ROOT / "train/teacher_100k"
SHARD_DIR = TEACHER_ROOT / "shards"
EPISODES_JSON = (
    DATA_ROOT
    / "data/datasets/ma_objectnav_ovon/train/ma_train_16000_samefloor_mixed3_seed2026_2027.json"
)
IIN_DIR = (
    DATA_ROOT
    / "data/datasets/ma_objectnav_ovon/train/iin_images_train16000_mixed3_stretch512"
)
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

MAP_SIZE = 96
K_MAX = 4
M_MAX = 5
UNREACHABLE = 1e4


def shard_ids(limit: int | None = None, offset: int = 0) -> list[int]:
    paths = sorted(SHARD_DIR.glob("shard_*.safetensors"))
    ids = [int(p.stem.split("_")[1]) for p in paths]
    return ids[offset: limit + offset] if limit is not None else ids


def load_index() -> list[dict]:
    rows = []
    with open(TEACHER_ROOT / "index.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


_FLOAT = {"geo": np.float16, "heat": np.float16}


class Store:
    """一组 shard 的内存视图，行号是本对象内的局部索引。"""

    def __init__(self, ids: list[int]):
        chunks = []
        metas: list[dict] = []
        base = 0
        for sid in ids:
            arrs = load_safetensors(str(SHARD_DIR / f"shard_{sid:05d}.safetensors"))
            n = arrs["geo"].shape[0]
            arrs["teacher"] = arrs["teacher"].astype(np.int16)
            arrs["teacher_core"] = arrs["teacher_core"].astype(np.int16)
            arrs["contract"] = arrs["contract"].astype(np.int16)
            arrs["teacher_gt"] = arrs["teacher_gt"].astype(np.int16)
            chunks.append(arrs)
            with open(SHARD_DIR / f"shard_{sid:05d}.meta.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    meta = json.loads(line)
                    meta["row"] = base + meta["offset"]
                    metas.append(meta)
            base += n

        self.n = base
        self.geo = np.concatenate([c["geo"] for c in chunks])
        self.heat = np.concatenate([c["heat"] for c in chunks])
        self.pose = np.concatenate([c["pose"] for c in chunks])
        self.peak = np.concatenate([c["peak"] for c in chunks])
        self.travel = np.concatenate([c["travel"] for c in chunks])
        self.teacher = np.concatenate([c["teacher"] for c in chunks])
        self.teacher_core = np.concatenate([c["teacher_core"] for c in chunks])
        self.contract = np.concatenate([c["contract"] for c in chunks])
        self.valid_k = np.concatenate([c["valid_k"] for c in chunks])
        self.valid_m = np.concatenate([c["valid_m"] for c in chunks])
        self.meta = metas
        self.episode_rows: dict[str, list[int]] = {}
        for m in metas:
            self.episode_rows.setdefault(m["episode_id"], []).append(m["row"])

    def episodes(self) -> list[str]:
        return sorted(self.episode_rows)

    def rows_of(self, episode_id: str) -> np.ndarray:
        return np.asarray(sorted(self.episode_rows[episode_id]), dtype=np.int64)


def split_by_episode(n_shards: int, val_shards: int) -> tuple[list[int], list[int]]:
    all_ids = shard_ids()
    val = all_ids[-val_shards:] if val_shards else []
    train = all_ids[:n_shards]
    return train, val


def _write_gzip_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(obj, fh, ensure_ascii=False)
    tmp.replace(path)


def episode_meta_lite() -> dict[str, dict]:
    """16k episodes 的瘦身视图（指令/目标/难度），首次解析后缓存。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "episodes_lite.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    with open(EPISODES_JSON, encoding="utf-8") as fh:
        raw = json.load(fh)

    lite: dict[str, dict] = {}
    for e in raw["episodes"]:
        goals = []
        for g in e["goals_set"]:
            goals.append(
                {
                    "index": int(g["goal_index"]),
                    "category": g["object_category"],
                    "modality": g["modality"],
                    "object_name": g.get("object_name") or "",
                    "tn_text": (g.get("tn_text") or "")[:600],
                    "position": g["position"],
                    "n_distractors": len(g.get("distractors") or []),
                }
            )
        agents = []
        for a in e["agents"]:
            agents.append(
                {
                    "agent_id": a.get("agent_id"),
                    "start_position": a.get("start_position"),
                    "start_rotation": a.get("start_rotation"),
                }
            )
        routing = e.get("routing") or {}
        lite[e["episode_id"]] = {
            "episode_id": e["episode_id"],
            "scene_key": e["scene_key"],
            "scene_id": e["scene_id"],
            "num_agents": e["num_agents"],
            "success_distance": e["success_distance"],
            "budget_geo": e.get("budget_geo"),
            "stratum": e.get("stratum"),
            "difficulty": e.get("difficulty"),
            "oracle_makespan": routing.get("oracle_makespan"),
            "geometric_makespan": routing.get("geometric_makespan"),
            "goals": goals,
            "agents": agents,
        }
    _write_gzip_json(cache, lite)
    return lite


def iin_map() -> dict[str, str]:
    """(episode_id, goal_index) -> IIN 图片文件名。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "iin_index.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    with open(IIN_DIR / "manifest.json", encoding="utf-8") as fh:
        man = json.load(fh)
    out = {}
    for it in man["items"]:
        out[f"{it['episode_id']}#{it['goal_index']}"] = Path(it["image"]).name
    _write_gzip_json(cache, out)
    return out


def geodesic_costs() -> dict[str, np.ndarray]:
    """episode_id -> 打包好的真测地数据 (4,5) sg / (5,5) gg / oracle / 长度。

    只从 episodes JSON 的 routing 里取学长生成器算好的 habitat navmesh 测地矩阵——
    shard 里的 travel 是 BFS 步数，与真测地 Spearman 只有 0.05，不能当决策代价用。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "geodesic.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {
            "ids": z["ids"],
            "sg": z["sg"],
            "gg": z["gg"],
            "ok": z["ok"],
            "oracle": z["oracle"],
            "oracle_ms": z["oracle_ms"],
        }

    with open(EPISODES_JSON, encoding="utf-8") as fh:
        raw = json.load(fh)
    # 测地不可达在 JSON 里是 null；sg 与路由矩阵量纲一致，统一用 inf 表示走不到
    ids, sg, gg, oracle, oms, ok = [], [], [], [], [], []
    for e in raw["episodes"]:
        r = e.get("routing") or {}
        s = r.get("start_goal_geo")
        g = r.get("goal_goal_geo")
        if not s or not g:
            continue
        s = np.asarray(s, dtype=np.float32)
        g = np.asarray(g, dtype=np.float32)
        if s.shape[0] > K_MAX or s.shape[1] > M_MAX:
            continue
        a = np.full((K_MAX, M_MAX), np.inf, np.float32)
        b = np.full((M_MAX, M_MAX), np.inf, np.float32)
        a[: s.shape[0], : s.shape[1]] = np.where(np.isfinite(s), s, np.inf)
        b[: g.shape[0], : g.shape[0]] = np.where(np.isfinite(g), g, np.inf)
        o = np.asarray(r.get("oracle_assignment") or [], dtype=np.int16)
        oc = np.full(M_MAX, -1, np.int16)
        oc[: min(o.size, M_MAX)] = o[:M_MAX]
        ms = float(r.get("oracle_makespan") or np.nan)
        ids.append(e["episode_id"])
        sg.append(a)
        gg.append(b)
        oracle.append(oc)
        oms.append(ms)
        ok.append(np.isfinite(a).any() and np.isfinite(ms))
    out = {
        "ids": np.asarray(ids),
        "sg": np.stack(sg),
        "gg": np.stack(gg),
        "ok": np.asarray(ok, dtype=bool),
        "oracle": np.stack(oracle),
        "oracle_ms": np.asarray(oms, dtype=np.float32),
    }
    np.savez_compressed(cache, **out)
    return out

