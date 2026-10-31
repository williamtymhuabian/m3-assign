"""把 goals_set 的世界坐标投到 episode 的 BEV 地图坐标系。

shard 里的 travel（BFS 步数）与真测地 Spearman 只有 0.05，peak 热图峰值的位置也几乎
不含目标几何（中位 rho ≈ -0.09）。但起点帧的 pose 是精确的地图坐标，而 episode JSON
里已知每个 agent 的世界起点，两者可以拟合出 world→map 的相似变换（每 episode 一个），
再把 goals_set[j].position 投进去——投出来的目标间距离与真测地 Spearman 中位 0.886。

产物 cache/goal_map.npz：
  ids  (E,)        episode_id
  pos  (E,M_MAX,2) 目标地图坐标（缺失/拟合失败为 NaN）
  ok   (E,)        该 episode 的变换是否可用
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from safetensors import safe_open

from .data import CACHE_DIR, EPISODES_JSON, K_MAX, M_MAX, SHARD_DIR, shard_ids


MAP_SCALE = 1.0 / 12.0  # 实测 145 个场景的拟合 scale 全部是 1/12（地图为世界坐标等比缩放）


def _fit_rt(src: np.ndarray, dst: np.ndarray, allow_reflection: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """固定 scale=MAP_SCALE，只拟合旋转+平移（稳定，k=2 也不会把 scale 炸开）。"""
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cs, cd = src - mu_s, dst - mu_d
    cov = cd.T @ cs / n
    U, _S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if not allow_reflection:
        D[1, 1] = np.sign(np.linalg.det(U) * np.linalg.det(Vt)) or 1.0
    R = U @ D @ Vt
    t = mu_d - MAP_SCALE * R @ mu_s
    res = float(np.median(np.linalg.norm(MAP_SCALE * (cs @ R.T) + mu_d - dst, axis=1)))
    return R, t, res


def fit_similarity(world: np.ndarray, map_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """拟合 world→map（两种朝向取残差小的），返回 (MAP_SCALE*R, t, 残差中位)。"""
    best = None
    for refl in (False, True):
        R, t, res = _fit_rt(world, map_xy, refl)
        if best is None or res < best[2]:
            best = (MAP_SCALE * R, t, res)
    return best


def _start_poses() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """episode_id -> (起点帧 pose (K,2) 地图坐标, 对应的 agent 槽位)。

    valid_k 不是前缀掩码（槽位可能有洞），必须按 flatnonzero 取，否则把补齐行当 agent。
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in shard_ids():
        mpath = SHARD_DIR / f"shard_{sid:05d}.meta.jsonl"
        first: dict[str, int] = {}
        with open(mpath, encoding="utf-8") as fh:
            for line in fh:
                m = json.loads(line)
                eid = m["episode_id"]
                if eid not in first or m["event_id"] < first[eid][0]:
                    first[eid] = (m["event_id"], m["offset"])
        if not first:
            continue
        with safe_open(str(SHARD_DIR / f"shard_{sid:05d}.safetensors"), framework="numpy") as f:
            pose = f.get_tensor("pose")
            vk = f.get_tensor("valid_k")
        for eid, (_, off) in first.items():
            slots = np.flatnonzero(vk[off])
            out[eid] = (pose[off][slots].astype(np.float64), slots)
    return out


def goal_map_positions() -> dict[str, np.ndarray]:
    """episode_id -> (M_MAX,2) 目标地图坐标；拟合失败的 episode 不出现。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "goal_map.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {str(e): p for e, p, ok in zip(z["ids"], z["pos"], z["ok"]) if ok}

    with open(EPISODES_JSON, encoding="utf-8") as fh:
        raw = json.load(fh)
    ep_goals: dict[str, np.ndarray] = {}
    ep_agents: dict[str, np.ndarray] = {}
    for e in raw["episodes"]:
        pos = np.full((M_MAX, 2), np.nan)
        for g in e["goals_set"]:
            i = int(g["goal_index"])
            if i < M_MAX:
                p = g["position"]
                pos[i] = (p[0], p[2])  # Habitat 世界坐标的水平分量是 x / z
        st = np.full((K_MAX, 2), np.nan)
        for a in e["agents"][:K_MAX]:
            p = a["start_position"]
            st[len(np.flatnonzero(~np.isnan(st[:, 0])))] = (p[0], p[2])
        ep_goals[e["episode_id"]] = pos
        ep_agents[e["episode_id"]] = st

    poses = _start_poses()
    ids, allpos, ok, resid = [], [], [], []
    for eid, (pose, _slots) in poses.items():
        if eid not in ep_goals:
            continue
        st = ep_agents[eid]
        k = min(len(st), len(pose))
        gp = ep_goals[eid]
        good = np.isfinite(gp[:, 0])
        # 起点几乎重合时旋转不可辨识（会把 scale 拟合炸开），直接跳过
        spread = float(np.linalg.norm(st[:k, None] - st[None, :k], axis=-1).max()) if k else 0.0
        if k < 2 or good.sum() < 1 or spread < 0.3:
            continue
        A, t, res = fit_similarity(st[:k], pose[:k])
        if res >= 0.02 and k <= 4:
            # 少数 episode 的 agent 顺序与槽位不一致：用排列找到能拟合成相似变换的配对
            best = None
            for pm in itertools.permutations(range(k)):
                a2, t2, r2 = fit_similarity(st[list(pm)], pose[:k])
                if best is None or r2 < best[2]:
                    best = (a2, t2, r2)
            if best[2] < res:
                A, t, res = best
        proj = np.full((M_MAX, 2), np.nan)
        proj[good] = gp[good] @ A.T + t
        ids.append(eid)
        allpos.append(proj)
        ok.append(res < 0.02)
        resid.append(res)

    pos = np.stack(allpos).astype(np.float32)
    ids_a = np.asarray(ids)
    ok_a = np.asarray(ok, dtype=bool)
    np.savez_compressed(cache, ids=ids_a, pos=pos, ok=ok_a)
    print(
        f"[mapgeo] {len(ids)} episodes, 拟合残差中位 {np.median(resid):.5f}, "
        f"可用 {ok_a.mean():.3f}",
        flush=True,
    )
    return {str(e): p for e, p, o in zip(ids_a, pos, ok_a) if o}
