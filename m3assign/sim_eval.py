"""真实几何下的分配策略评测。

episode 里带着 P1 生成器用 habitat navmesh 算好的测地矩阵（start_goal_geo /
goal_goal_geo），这里把"分配方案"折成官方 benchmark.metrics 的
SR / MSPL / SPL_sum / makespan，让 AssignNet 与 teacher、几何基线同台比较。

用法：
    python -m m3assign.sim_eval --episodes 400
输出 cache/sim_eval.json（看板 /api/sim 读取）。
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .data import DATA_ROOT, K_MAX, M_MAX, ROOT, UNREACHABLE, Store, shard_ids
from .model import build
from .mapgeo import goal_map_positions
from .train import cpu_tensors, mgeo_tensors, take

SIM_CODE = ROOT / "hm3d-m3-code" / "sim-code"
sys.path.insert(0, str(SIM_CODE))
from benchmark.metrics import team_metrics  # noqa: E402
from benchmark.routing import balanced_greedy, geometric_assignment  # noqa: E402

TRAIN_JSON = DATA_ROOT / "data/datasets/ma_objectnav_ovon/train/ma_train_16000_samefloor_mixed3_seed2026_2027.json"
INF = float("inf")


def load_routing(episode_ids: set[str]) -> dict[str, dict]:
    """episode_id -> 官方几何矩阵与预算，只保留需要的那批。"""
    with open(TRAIN_JSON, encoding="utf-8") as fh:
        raw = json.load(fh)
    out = {}
    for e in raw["episodes"]:
        eid = e["episode_id"]
        if eid not in episode_ids:
            continue
        r = e["routing"]
        out[eid] = {
            "episode": e,
            "sg": np.asarray(r["start_goal_geo"], dtype=np.float64),
            "gg": np.asarray(r["goal_goal_geo"], dtype=np.float64),
            "oracle": list(r["oracle_assignment"]),
            "d_star": float(r["oracle_makespan"]),
            "budget": float(e["budget_geo"]),
            "K": int(e["num_agents"]),
            "M": len(e["goals_set"]),
        }
    return out


def nn_order(sg_a: np.ndarray, gg: np.ndarray, grp: list[int]) -> list[int]:
    """从起点出发的贪心最近邻顺序（与 benchmark.routing 的路线一致）。"""
    rest, order, at = list(grp), [], None
    while rest:
        j = min(rest, key=lambda x: sg_a[x]) if at is None else min(rest, key=lambda x: gg[at, x])
        order.append(j)
        rest.remove(j)
        at = j
    return order


def route_len(sg_a: np.ndarray, gg: np.ndarray, order: list[int]) -> tuple[float, list[int]]:
    """按给定顺序走的长度；测地不可达（跨 island）处截断，之后视作没找到。"""
    total, prev, found = 0.0, None, []
    for j in order:
        leg = sg_a[j] if prev is None else gg[prev, j]
        if not np.isfinite(leg):
            break
        total += float(leg)
        found.append(j)
        prev = j
    return total, found


def score_routes(info: dict, routes: list[list[int]]) -> dict:
    """routes[a] = agent a 依次要去的目标 -> 官方 team_metrics + 预算判定。"""
    sg, gg = info["sg"], info["gg"]
    per_agent = []
    for a, order in enumerate(routes):
        length, found = route_len(sg[a], gg, order)
        per_agent.append({"path_length": length, "goals_found": found})
    res = team_metrics(info["episode"], {"per_agent": per_agent})
    lengths = [p["path_length"] for p in per_agent]
    res["makespan"] = max(lengths) if lengths else INF
    res["regret"] = (res["makespan"] / info["d_star"]) if np.isfinite(res["makespan"]) else INF
    res["within_budget"] = int(res["success"] and res["makespan"] <= info["budget"])
    return res


def routes_from_assignment(info: dict, assign: list[int]) -> list[list[int]]:
    """静态 goal->agent 映射 -> 每个 agent 的 NN 顺序路线。"""
    groups = [[] for _ in range(info["K"])]
    for j, a in enumerate(assign):
        if 0 <= a < info["K"]:
            groups[a].append(j)
    return [nn_order(info["sg"][a], info["gg"], groups[a]) for a in range(info["K"])]


def auction_routes(probs: np.ndarray, valid_k: np.ndarray, K: int, M: int) -> list[list[int]]:
    """按模型分数从高到低认领目标，每个 agent 限额 ceil(M/K)。

    这样分配覆盖全部目标，考察的是模型的排序质量，而不是它一帧内敢不敢开口。
    """
    cap = int(np.ceil(M / max(1, K)))
    routes = [[] for _ in range(K)]
    taken: set[int] = set()
    pairs = sorted(
        ((float(probs[a, j]), a, j) for a in range(K) for j in range(M) if valid_k[a]),
        reverse=True,
    )
    for _, a, j in pairs:
        if j not in taken and len(routes[a]) < cap:
            routes[a].append(j)
            taken.add(j)
    return routes


def one_shot_cover(info: dict, pred_row: np.ndarray, valid_k: np.ndarray) -> float:
    """模型一帧内 argmax 能覆盖多少目标（撞车和漏目标是真实失败模式）。"""
    claimed = {int(pred_row[a]) for a in range(len(pred_row)) if valid_k[a]}
    return len([j for j in claimed if j < info["M"]]) / info["M"]


def teacher_routes(store: Store, rows: list[int], K: int, M: int, sg: np.ndarray) -> list[list[int]]:
    """teacher 在整条 rollout 里依次下达的 teacher_core 就是它真正执行的计划。

    快照是每 25 步采一次， rollout 尾部可能没覆盖到的目标在这里按最近 agent 补齐。
    """
    routes = [[] for _ in range(K)]
    for r in sorted(rows):
        for a, g in enumerate(store.teacher_core[r][:K]):
            g = int(g)
            if 0 <= g < M and g not in routes[a]:
                routes[a].append(g)
    seen = {j for r in routes for j in r}
    for j in range(M):
        if j in seen:
            continue
        a = int(np.argmin(np.where(np.isfinite(sg[:, j]), sg[:, j], INF)))
        routes[a].append(j)
        seen.add(j)
    return routes  # 顺序就是 teacher 的实际下令顺序，不再重排


def fill_leftovers(info: dict, assign: list[int]) -> list[int]:
    """一对一分配没覆盖到的目标，补给最近的 agent（评测要求分配完整可比）。"""
    sg = info["sg"]
    out = list(assign)
    for j, a in enumerate(out):
        if a < 0 and np.isfinite(sg[:, j]).any():
            out[j] = int(np.argmin(np.where(np.isfinite(sg[:, j]), sg[:, j], INF)))
    return out


def hungarian_assign(sg: np.ndarray, K: int, M: int) -> list[int]:
    cost = np.where(np.isfinite(sg), sg, 1e6)
    assign = [-1] * M
    pairs = list(zip(*linear_sum_assignment(cost.T if K >= M else cost)))
    for i, j in pairs:
        if K >= M:
            assign[i] = int(j)
        else:
            assign[j] = int(i)
    return assign


def hungarian_on_costs(
    cost: np.ndarray, valid_k: np.ndarray, valid_m: np.ndarray, M: int
) -> list[int]:
    """代价矩阵 (K_MAX, M_MAX) -> goal->agent 一对一分配。

    行按 valid_k、列按 valid_m 取，补齐的 agent 不能领目标、已解决的目标也不能再被分配。
    cost 越大越差；非有限值按 1e6 处理。
    """
    actors = np.flatnonzero(valid_k)
    goals = np.flatnonzero(valid_m)
    goals = goals[goals < M]  # 起点帧上 valid_m 恰好是前 M 个目标；防御性截断补齐列
    assign = [-1] * M
    if actors.size == 0 or goals.size == 0:
        return assign
    c = np.where(np.isfinite(cost), cost, 1e6)[np.ix_(actors, goals)]
    for i, j in zip(*linear_sum_assignment(c)):
        assign[int(goals[j])] = int(actors[i])
    return assign


def model_hungarian_assign(
    score: np.ndarray, valid_k: np.ndarray, valid_m: np.ndarray, M: int
) -> list[int]:
    """模型打分 -> 一对一匈牙利分配。

    auction 解码考的是排序，和几何匈牙利基线不是同一个解码器，数字不可比；
    这里换成与 hungarian 基线完全相同的解码器，只让"代价来源"不同（测地 vs 模型）。
    """
    return hungarian_on_costs(-np.asarray(score, dtype=np.float64), valid_k, valid_m, M)


def minmax_on_costs(
    cost: np.ndarray,
    gg: np.ndarray,
    valid_k: np.ndarray,
    valid_m: np.ndarray,
    M: int,
    cap: int | None = None,
) -> list[int]:
    """min-makespan 枚举解码：MSPL 是 max 指标，min-sum 是错的目标。

    cost 只含第一跳（agent→目标），第二跳用 gg。K^M ≤ 1024，直接枚举。
    cap 限制单个 agent 的目标数（上限取 max(cap, ceil(目标数/K))，保证可行）。
    """
    actors = np.flatnonzero(valid_k)
    goals = [j for j in range(M) if valid_m[j]]
    if actors.size == 0 or not goals:
        return [-1] * M
    K = len(actors)
    if cap is not None:
        cap = max(cap, -(-len(goals) // K))
    best, best_assign = INF, None
    for combo in itertools.product(range(K), repeat=len(goals)):
        if cap is not None and max(combo.count(a) for a in range(K)) > cap:
            continue
        groups = [[] for _ in range(K)]
        for j, a in zip(goals, combo):
            groups[a].append(j)
        ms = 0.0
        for a in range(K):
            order = nn_order(cost[actors[a]], gg, groups[a]) if groups[a] else []
            tot, prev, bad = 0.0, None, False
            for j in order:
                leg = cost[actors[a], j] if prev is None else gg[prev, j]
                if not np.isfinite(leg):
                    bad = True
                    break
                tot += leg
                prev = j
            if bad:
                ms = INF
                break
            ms = max(ms, tot)
        if ms < best:
            best, best_assign = ms, combo
    out = [-1] * M
    for j, a in zip(goals, best_assign):
        out[j] = int(actors[a])
    return out


def space_costs(info: dict) -> tuple[np.ndarray, np.ndarray]:
    """部署可得几何：任务给定的目标世界坐标 + 起点位姿 → 平面欧氏（sg, gg）。"""
    ep = info["episode"]
    starts = np.asarray([a["start_position"] for a in ep["agents"]], dtype=np.float64)
    goals = np.asarray([g["position"] for g in ep["goals_set"]], dtype=np.float64)
    sg = np.linalg.norm(starts[:, None, :] - goals[None, :, :], axis=-1)
    gg = np.linalg.norm(goals[:, None, :] - goals[None, :, :], axis=-1)
    return sg, gg


def nearest_greedy_assign(sg: np.ndarray, K: int) -> list[int]:
    M = sg.shape[1]
    assign = [-1] * M
    free = set(range(K))
    order = list(np.argsort(np.where(np.isfinite(sg), sg, INF).min(axis=0)))
    for j in order:
        cand = [k for k in free if np.isfinite(sg[k, j])]
        if cand:
            a = min(cand, key=lambda k: sg[k, j])
            assign[int(j)] = a
            free.discard(a)
    for j in order:  # 目标多于 agent：让最近的 agent 顺手多跑一个
        if assign[int(j)] < 0 and np.isfinite(sg[:, j]).any():
            assign[int(j)] = int(np.argmin(np.where(np.isfinite(sg[:, j]), sg[:, j], INF)))
    return assign


def assign_of(routes: list[list[int]], M: int) -> list[int]:
    """路线列表反推 goal->agent，用于计算与 oracle 分配的一致率。"""
    out = [-1] * M
    for a, r in enumerate(routes):
        for j in r:
            out[j] = a
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--detail", type=int, default=40, help="每策略落盘的逐 episode 行数（配对 A/B 用大值）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = Path(__file__).resolve().parent.parent / "runs" / "latest"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    ckpt = args.ckpt or (run_dir / "best.pt")
    out_path = args.out or (Path(__file__).resolve().parent.parent / "cache" / "sim_eval.json")

    val_shards = state["val_shards"]
    print(f"[sim_eval] 用评测分片 {val_shards}（训练未见过的数据）", flush=True)
    store = Store(shard_ids(limit=len(val_shards), offset=min(val_shards)))

    by_ep: dict[str, list[int]] = {}
    first_row: dict[str, int] = {}
    for m in store.meta:
        eid = m["episode_id"]
        by_ep.setdefault(eid, []).append(m["row"])
        ev = m.get("event_id", 0)
        if eid not in first_row or ev < first_row[eid][0]:
            first_row[eid] = (ev, m["row"])
    first_row = {e: v[1] for e, v in first_row.items()}

    rng = random.Random(args.seed)
    eids = sorted(first_row)
    rng.shuffle(eids)
    infos = load_routing(set(eids))
    eids = [e for e in eids if e in infos][: args.episodes]
    print(f"[sim_eval] 可评测 episode：{len(eids)}", flush=True)
    if not eids:
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = build(blob.get("meta")).to(device).eval()
    model.load_state_dict(blob["state_dict"])
    ckpt_step = (blob.get("meta") or {}).get("step")
    tensors = cpu_tensors(store)
    gmap = goal_map_positions()
    if ((blob.get("meta") or {}).get("args") or {}).get("mgeo"):
        tensors.update(mgeo_tensors(store))

    # 起点快照（event_id 最小的那行）上的模型分数
    probs_by_ep, pred_by_ep, vk_by_ep, vm_by_ep, raw_by_ep, travel_by_ep, eucl_by_ep, map_by_ep = {}, {}, {}, {}, {}, {}, {}, {}
    idx = np.asarray([first_row[e] for e in eids], dtype=np.int64)
    for s in range(0, len(idx), 256):
        chunk = idx[s : s + 256]
        batch = take(tensors, chunk, device)
        with torch.no_grad():
            logits = model(batch)
            probs = logits.softmax(-1).cpu().numpy()
        raw = logits.detach().float().cpu().numpy()
        vk = batch["valid_k"].cpu().numpy()
        vm = batch["valid_m"].cpu().numpy()
        tv = batch["travel"].detach().float().cpu().numpy()
        ps, pk = batch["pose"].cpu().numpy(), batch["peak"][..., :2].cpu().numpy()
        eu = np.linalg.norm(pk[:, None, :, :] - ps[:, :, None, :], axis=-1)  # (B,K,M) 归一化坐标欧氏
        for bi, eid in enumerate(eids[s : s + len(chunk)]):
            probs_by_ep[eid] = probs[bi]
            pred_by_ep[eid] = probs[bi].argmax(-1)
            raw_by_ep[eid] = raw[bi]
            vk_by_ep[eid] = vk[bi]
            vm_by_ep[eid] = vm[bi]
            travel_by_ep[eid] = tv[bi]
            eucl_by_ep[eid] = eu[bi]
            gpos = gmap.get(eid)
            if gpos is None or not np.isfinite(gpos[:, 0]).any():
                map_by_ep[eid] = np.full((K_MAX, M_MAX), 1e6, np.float64)
            else:
                md = np.linalg.norm(ps[bi][:, None, :] - gpos[None, :, :], axis=-1)
                md = np.where(np.isfinite(gpos[None, :, 0]), md, 1e6)
                pad = np.full((K_MAX, M_MAX), 1e6)
                pad[: md.shape[0], : md.shape[1]] = md[:K_MAX, :M_MAX]
                map_by_ep[eid] = pad

    policies = ("oracle", "balanced", "geometric", "hungarian", "hungarian_bfs", "hungarian_eucl", "hungarian_map", "minmax_eucl", "minmax_bal", "nearest", "teacher", "assignnet", "assignnet_hung", "random")
    per_ep: dict[str, list[dict]] = {p: [] for p in policies}
    cover = []
    # 信息差诊断：travel 是地图 BFS 步数、sg 是测地米数，量纲不同，只能比"排序/决策"是否一致
    bfs_agree, bfs_rho = [], []
    t0 = time.time()
    for pos, eid in enumerate(sorted(eids)):
        info = infos[eid]
        sg, gg, K, M = info["sg"], info["gg"], info["K"], info["M"]
        # 部署可得：目标世界坐标（任务给定）+ 里程计位姿 → 平面欧氏；第二跳同样用欧氏
        sg_e, gg_e = space_costs(info)
        sg_e = np.where(np.isfinite(sg_e), sg_e, 1e6)
        gg_e = np.where(np.isfinite(gg_e), gg_e, 1e6)
        routes = {
            "oracle": routes_from_assignment(info, info["oracle"]),
            "balanced": routes_from_assignment(info, balanced_greedy(sg, gg)[1]),
            "geometric": routes_from_assignment(info, geometric_assignment(sg)),
            "hungarian": routes_from_assignment(info, fill_leftovers(info, hungarian_assign(sg, K, M))),
            "hungarian_bfs": routes_from_assignment(
                info,
                fill_leftovers(info, hungarian_on_costs(travel_by_ep[eid], vk_by_ep[eid], vm_by_ep[eid], M)),
            ),
            "hungarian_eucl": routes_from_assignment(
                info,
                fill_leftovers(info, hungarian_on_costs(eucl_by_ep[eid], vk_by_ep[eid], vm_by_ep[eid], M)),
            ),
            "hungarian_map": routes_from_assignment(
                info,
                fill_leftovers(info, hungarian_on_costs(map_by_ep[eid], vk_by_ep[eid], vm_by_ep[eid], M)),
            ),
            "minmax_eucl": routes_from_assignment(
                info,
                fill_leftovers(
                    info,
                    minmax_on_costs(sg_e, gg_e, vk_by_ep[eid], vm_by_ep[eid], M),
                ),
            ),
            "minmax_bal": routes_from_assignment(
                info,
                fill_leftovers(
                    info,
                    minmax_on_costs(
                        sg_e, gg_e, vk_by_ep[eid], vm_by_ep[eid], M, cap=-(-M // K)
                    ),
                ),
            ),
            "nearest": routes_from_assignment(info, nearest_greedy_assign(sg, K)),
            "teacher": teacher_routes(store, by_ep[eid], K, M, sg),
            "assignnet": auction_routes(probs_by_ep[eid], vk_by_ep[eid], K, M),
            "assignnet_hung": routes_from_assignment(
                info,
                fill_leftovers(
                    info,
                    model_hungarian_assign(raw_by_ep[eid], vk_by_ep[eid], vm_by_ep[eid], M),
                ),
            ),
            "random": routes_from_assignment(info, [rng.randrange(K) for _ in range(M)]),
        }
        cover.append(one_shot_cover(info, pred_by_ep[eid], vk_by_ep[eid]))
        for p, rt in routes.items():
            m = score_routes(info, rt)
            m["episode_id"] = eid
            m["agree_oracle"] = float(np.mean([x == y for x, y in zip(assign_of(rt, M), info["oracle"])]))
            per_ep[p].append(m)
        if (pos + 1) % 100 == 0:
            print(f"[sim_eval] {pos + 1}/{len(eids)} ep, {time.time() - t0:.0f}s", flush=True)

    # 信息差 vs 算法差：BFS 代价与 GT 测地是否给出同一个分配
    from scipy.stats import spearmanr

    for eid in sorted(eids):
        info = infos[eid]
        sg, K, M = info["sg"], info["K"], info["M"]
        vk, vm = vk_by_ep[eid], vm_by_ep[eid]
        act, goal = np.flatnonzero(vk), np.flatnonzero(vm)
        goal = goal[goal < M]  # sg 只有 M 列，valid_m 落到补齐列时不可比对
        tv = travel_by_ep[eid][np.ix_(act, goal)].astype(np.float64)
        sgv = sg[np.ix_(act, goal)].astype(np.float64)
        ok = np.isfinite(sgv) & (tv < UNREACHABLE)
        if ok.sum() < 3:
            continue
        a = hungarian_on_costs(travel_by_ep[eid], vk, vm, M)
        b = hungarian_assign(sg, K, M)
        both = [j for j in range(M) if a[j] >= 0 and b[j] >= 0]
        if both:
            bfs_agree.append(float(np.mean([a[j] == b[j] for j in both])))
        # 单一 agent 或单一目标时相关系数无定义，跳过而不是记 nan
        if len(act) > 1 and len(goal) > 1 and sgv[ok].std() > 0 and tv[ok].std() > 0:
            bfs_rho.append(float(spearmanr(sgv[ok], tv[ok]).statistic))

    print(
        f"[info_gap] BFS 与 GT 测地的分配一致率={np.mean(bfs_agree):.3f} "
        f"(逐 ep 标准差 {np.std(bfs_agree):.3f}, n={len(bfs_agree)}), "
        f"Spearman={np.nanmean(bfs_rho):.3f} (n={len(bfs_rho)})",
        flush=True,
    )

    agg = {}
    for p, lst in per_ep.items():
        agg[p] = {
            "n": len(lst),
            "SR": round(float(np.mean([m["success"] for m in lst])), 4),
            "MSPL": round(float(np.mean([m["mspl"] for m in lst])), 4),
            "SPL_sum": round(float(np.mean([m["spl_sum"] for m in lst])), 4),
            "SR_within_budget": round(float(np.mean([m["within_budget"] for m in lst])), 4),
            "progress": round(float(np.mean([m["progress"] for m in lst])), 4),
            "makespan": round(float(np.median([m["makespan"] for m in lst])), 3),
            "regret": round(float(np.median([m["regret"] for m in lst])), 3),
            "gini": round(float(np.mean([m["fairness_gini"] for m in lst])), 4),
            "agree_oracle": round(float(np.mean([m["agree_oracle"] for m in lst])), 4),
        }
    agg["assignnet"]["one_shot_goal_cover"] = round(float(np.mean(cover)), 4)

    payload = {
        "generated_at": time.time(),
        "ckpt": str(ckpt),
        "ckpt_step": ckpt_step,
        "val_shards": val_shards,
        "episodes": len(eids),
        "note": "几何取自 episode 内置的 habitat navmesh 测地矩阵（setup/run_verify_navmesh.sh 已用 habitat 复算核对：36 场景/144 组矩阵平均误差 5 mm、相关系数 1.0）；路线长度按分配顺序累加",
        "metrics": agg,
        "episodes_detail": [
            {
                "episode_id": m["episode_id"],
                "policy": p,
                "SR": int(m["success"]),
                "MSPL": round(float(m["mspl"]), 4),
                "makespan": round(float(m["makespan"]), 3),
                "regret": round(float(m["regret"]), 3),
                "within_budget": m["within_budget"],
                "agree_oracle": round(m["agree_oracle"], 3),
            }
            for p in ("oracle", "teacher", "assignnet", "hungarian", "hungarian_bfs", "hungarian_eucl", "hungarian_map", "minmax_eucl", "minmax_bal", "assignnet_hung")
            for m in per_ep[p][: args.detail]
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[sim_eval] 写入 {out_path}", flush=True)
    for p in policies:
        print(f"  {p:12s} {agg[p]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
