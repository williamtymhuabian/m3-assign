"""用 habitat-sim 复算 val episode 的测地距离，核对 episode 内置几何矩阵的可信度。

跑在 m3 环境（py3.8 + habitat-sim 0.1.7）：
    setup/run_verify_navmesh.sh [场景数]
输出 cache/navmesh_check.json。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .data import CACHE_DIR, DATA_ROOT

DATA = DATA_ROOT / "data"
VAL = DATA / "datasets/ma_objectnav_ovon/val_unseen/ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json"
SCENES = DATA / "scene_datasets/hm3d-1.0/hm3d_v0.2/val"
CONFIG = DATA / "scene_datasets/hm3d-1.0/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
OUT = CACHE_DIR / "navmesh_check.json"
INF = float("inf")


def maybe_json(x):
    return json.loads(x) if isinstance(x, str) else x


def main() -> int:
    n_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    import habitat_sim

    eps = json.load(open(VAL, encoding="utf-8"))["episodes"]
    by_scene: dict[str, list[dict]] = {}
    for e in eps:
        key = Path(e["scene_id"]).parent.name  # 00849-a8BtkwhxdRV
        by_scene.setdefault(key, []).append(e)
    picked = [k for k in sorted(by_scene) if (SCENES / k).is_dir()][:n_scenes]
    print(f"[verify] 场景 {len(picked)} 个", flush=True)

    rows = []
    for key in picked:
        glb = sorted((SCENES / key).glob("*.basis.glb"))
        if not glb:
            continue
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(glb[0])
        sim_cfg.scene_dataset_config_file = str(CONFIG)
        sim_cfg.create_renderer = False
        sim_cfg.requires_textures = False
        sim_cfg.enable_physics = False
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        # 内置矩阵是 habitat 默认 agent 剖面（radius 0.1 / height 1.5）下 bake 的，
        # 用 setup/diag_navmesh.py 实测：0.1/1.5 误差 0.000 m，0.1/0.88 会差到 2 m。
        agent_cfg.height, agent_cfg.radius = 1.5, 0.1
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        # 官方 navmesh 是 TESM v2，0.1.7 读不进来；按默认剖面在内存里重算
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius, ns.agent_height = 0.1, 1.5
        ns.agent_max_climb, ns.agent_max_slope = 0.20, 45.0
        sim.recompute_navmesh(sim.pathfinder, ns)
        pf = sim.pathfinder

        def geo(a, b):
            try:
                pa, pb = pf.snap_point(np.asarray(a, np.float64)), pf.snap_point(np.asarray(b, np.float64))
            except RuntimeError:
                return INF
            p = habitat_sim.ShortestPath()
            p.requested_start, p.requested_end = pa, pb
            return p.geodesic_distance if pf.find_path(p) else INF

        for e in by_scene[key][:2]:
            starts = [np.asarray(a["start_position"], np.float64) for a in e["agents"]]
            pts = [maybe_json(g["rep_point"]) for g in e["goals_set"]]
            sg = np.asarray(e["routing"]["start_goal_geo"], np.float64)
            gg = np.asarray(e["routing"]["goal_goal_geo"], np.float64)
            mine_sg = np.array([[geo(s, p) for p in pts] for s in starts])
            mine_gg = np.array([[geo(p, q) for q in pts] for p in pts])
            for name, ref, got in (("start_goal", sg, mine_sg), ("goal_goal", gg, mine_gg)):
                ok = np.isfinite(ref) & np.isfinite(got)
                if not ok.any():
                    continue
                d = np.abs(ref[ok] - got[ok])
                rows.append(
                    {
                        "episode_id": e["episode_id"],
                        "scene": key,
                        "matrix": name,
                        "n": int(ok.sum()),
                        "mean_abs_err_m": round(float(d.mean()), 3),
                        "p95_abs_err_m": round(float(np.percentile(d, 95)), 3),
                        "corr": round(float(np.corrcoef(ref[ok], got[ok])[0, 1]), 4),
                    }
                )
            print(f"[verify] {e['episode_id']} rows={len(rows)}", flush=True)
        sim.close()

    per_scene: dict[str, float] = {}
    for r in rows:
        per_scene[r["scene"]] = max(per_scene.get(r["scene"], 0.0), r["mean_abs_err_m"])
    agg = {
        "scenes": len(picked),
        "clean_scenes": sum(1 for v in per_scene.values() if v < 0.05),
        "samples": len(rows),
        "mean_abs_err_m": round(float(np.mean([r["mean_abs_err_m"] for r in rows])), 3) if rows else None,
        "worst_p95_m": round(float(np.max([r["p95_abs_err_m"] for r in rows])), 3) if rows else None,
        "corr": round(float(np.mean([r["corr"] for r in rows])), 4) if rows else None,
        "agent": "radius 0.1 / height 1.5",
        "generated_at": time.time(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"agg": agg, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[verify] {agg} -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
