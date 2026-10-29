"""habitat-sim 冒烟测试：加载一个 HM3D v0.2 val 场景，渲染并查询 navmesh 最短路。

用法（环境变量在 run_smoke.sh 里设置）：
    python setup/smoke_habitat.py [scene_dir_name]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import habitat_sim
import numpy as np

DATA = Path(os.environ.get("M3_ROOT") or Path(__file__).resolve().parents[2]) / "HM3D-M3" / "data"
HM3D = DATA / "scene_datasets/hm3d-1.0/hm3d_v0.2"
SCENES = HM3D / "val"
CONFIG = HM3D / "hm3d_annotated_basis.scene_dataset_config.json"
EPISODES = DATA / "datasets/ma_objectnav_ovon/val_unseen/ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json"


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "00800-TEEsavR23oF"
    glb = sorted((SCENES / name).glob("*.basis.glb"))
    if not glb:
        print(f"[smoke] 场景缺失: {SCENES / name}")
        return 1
    print(f"[smoke] scene={glb[0].name} config={CONFIG.exists()}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(glb[0])
    sim_cfg.scene_dataset_config_file = str(CONFIG)
    sim_cfg.create_renderer = True
    sim_cfg.requires_textures = True
    sim_cfg.enable_physics = False
    sim_cfg.random_seed = 11

    rgb = habitat_sim.SensorSpec()
    rgb.uuid, rgb.sensor_type = "rgb", habitat_sim.SensorType.COLOR
    rgb.resolution, rgb.position, rgb.channels = [256, 256], [0.0, 0.88, 0.0], 4
    depth = habitat_sim.SensorSpec()
    depth.uuid, depth.sensor_type = "depth", habitat_sim.SensorType.DEPTH
    depth.resolution, depth.position, depth.channels = [256, 256], [0.0, 0.88, 0.0], 1

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    # HM3D-M3 的 navmesh 是 TESM v2，habitat-sim 0.1.7 读不了（is_loaded=False）。
    # 按 benchmark/rebake_navmesh.py 的参数在内存里重算，等价于官方 rebake 的结果，
    # 且不去改动已下载的场景文件。
    agent_cfg.height = 0.88
    agent_cfg.radius = 0.1
    agent_cfg.sensor_specifications = [rgb, depth]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius, ns.agent_height = 0.1, 0.88
    ns.agent_max_climb, ns.agent_max_slope = 0.20, 45.0
    sim.recompute_navmesh(sim.pathfinder, ns)
    print(f"[smoke] navmesh loaded={sim.pathfinder.is_loaded} area={sim.pathfinder.navigable_area:.1f} m^2")

    eps = json.load(open(EPISODES, encoding="utf-8"))["episodes"]
    eps = [e for e in eps if name.split("-", 1)[1] in e["scene_id"]]
    if not eps:
        print("[smoke] 该场景没有 mini100 episode，仅验证渲染")
    if eps:
        e = eps[0]
        ag0 = np.array(e["agents"][0]["start_position"], dtype=np.float64)
        gp = np.array(json.loads(e["goals_set"][0]["reach_points"]) if isinstance(e["goals_set"][0]["reach_points"], str) else e["goals_set"][0]["reach_points"][0], dtype=np.float64)
        a = sim.pathfinder.snap_point(ag0)
        g = sim.pathfinder.snap_point(gp)
        st = habitat_sim.AgentState()
        st.position = a
        st.rotation = np.asarray(e["agents"][0]["start_rotation"], dtype=np.float64)
        sim.get_agent(0).set_state(st)
        obs = sim.get_sensor_observations()
        print(f"[smoke] rgb={obs['rgb'].shape} mean={obs['rgb'].mean():.1f} depth_max={np.nanmax(obs['depth']):.2f}")
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = a, g
        ok = sim.pathfinder.find_path(path)
        print(f"[smoke] find_path ok={ok} points={len(path.points)} distance={path.geodesic_distance:.2f} m")
    sim.close()
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
