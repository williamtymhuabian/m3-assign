"""诊断：内置测地矩阵到底是用什么参数/什么代表点 bake 出来的。

在 m3 环境跑：EGL 环境变量同 setup/run_verify_navmesh.sh。
对每个不匹配的 episode，比较三种假设：
  A) 排序后的距离集合一致 -> 只是配对/顺序问题
  B) 换 reach_points 里的别的代表点能对上 -> 代表点选择问题
  C) 换 navmesh 烘焙参数能对上 -> agent radius/height 问题
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import habitat_sim

ROOT = Path(os.environ.get("M3_ROOT") or Path(__file__).resolve().parents[2])
DATA = ROOT / "HM3D-M3/data"
VAL = DATA / "datasets/ma_objectnav_ovon/val_unseen/ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json"
SCENES = DATA / "scene_datasets/hm3d-1.0/hm3d_v0.2/val"
CONFIG = DATA / "scene_datasets/hm3d-1.0/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
INF = float("inf")

PARAMS = [(0.1, 0.88), (0.05, 0.88), (0.1, 1.5), (0.05, 1.5), (0.25, 1.5), (0.3, 1.3)]


def maybe_json(x):
    return json.loads(x) if isinstance(x, str) else x


def make_sim(glb, radius, height):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(glb)
    sim_cfg.scene_dataset_config_file = str(CONFIG)
    sim_cfg.create_renderer = False
    sim_cfg.requires_textures = False
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height, agent_cfg.radius = height, radius
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius, ns.agent_height = radius, height
    ns.agent_max_climb, ns.agent_max_slope = 0.20, 45.0
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim


def geo_of(pf, a, b):
    try:
        pa, pb = pf.snap_point(np.asarray(a, np.float64)), pf.snap_point(np.asarray(b, np.float64))
    except RuntimeError:
        return INF
    p = habitat_sim.ShortestPath()
    p.requested_start, p.requested_end = pa, pb
    return p.geodesic_distance if pf.find_path(p) else INF


def main() -> int:
    eps = json.load(open(VAL, encoding="utf-8"))["episodes"]
    by_scene = {}
    for e in eps:
        by_scene.setdefault(Path(e["scene_id"]).parent.name, []).append(e)
    keys = sys.argv[1:] or ["00853-5cdEh9F2hJL", "00824-Dd4bFSTQ8gi"]
    for key in keys:
        glb = sorted((SCENES / key).glob("*.basis.glb"))
        if not glb:
            print("no glb", key)
            continue
        e = by_scene[key][0]
        ref = np.asarray(e["routing"]["goal_goal_geo"], np.float64)
        goals = [maybe_json(g["reach_points"]) for g in e["goals_set"]]
        reps = [maybe_json([g["rep_point"]])[0] for g in e["goals_set"]]
        M = len(goals)
        print(f"\n=== {key} {e['episode_id']} M={M} 每目标代表点数={[len(g) for g in goals]}")
        print(f"ref finite={np.isfinite(ref).sum()}/{ref.size}")
        for radius, height in PARAMS:
            sim = make_sim(glb[0], radius, height)
            pf = sim.pathfinder
            got = np.array([[geo_of(pf, p, q) for q in reps] for p in reps])
            ok = np.isfinite(ref) & np.isfinite(got)
            d = np.abs(ref[ok] - got[ok])
            first = [g[0] for g in goals]
            got1 = np.array([[geo_of(pf, p, q) for q in first] for p in first])
            ok1 = np.isfinite(ref) & np.isfinite(got1)
            d1 = np.abs(ref[ok1] - got1[ok1])
            sref = np.sort(ref[np.isfinite(ref)])
            sgot = np.sort(got[np.isfinite(got)])
            n = min(len(sref), len(sgot))
            ds = np.abs(sref[:n] - sgot[:n]) if n else np.array([np.nan])
            # 每个格子取所有代表点组合里的最小误差
            best = np.full_like(ref, INF)
            for i in range(M):
                for j in range(M):
                    for pi in goals[i]:
                        for pj in goals[j]:
                            v = geo_of(pf, pi, pj)
                            if np.isfinite(v) and np.isfinite(ref[i, j]):
                                best[i, j] = min(best[i, j], abs(v - ref[i, j]))
            fb = np.isfinite(best)
            sim.close()
            print(
                f"  r={radius} h={height}: rep_point mean={d.mean():.3f} p95={np.percentile(d,95):.3f} finite={ok.sum()}/{ref.size} | "
                f"reach[0] mean={d1.mean():.3f} | 排序集合 mean={ds.mean():.3f} | "
                f"任一reach best mean={np.mean(best[fb]):.3f} p95={np.percentile(best[fb],95):.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
