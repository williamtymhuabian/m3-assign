"""事件序列上的承诺评测：模型自己维持 contract，而不是读教师的 contract。

sim_eval 只在起点帧解一次分配；这里沿 teacher rollout 的事件序列走，每一步把
"我们上一帧的承诺"喂回模型，比较：
  - per_frame：每帧都重新解，丢掉记忆（contract 全 -1）
  - memory：contract = 我们上一帧的分配
  - memory + 切换余量：新目标分数要高出 λ 才换，否则保持
指标：与 oracle 分配的一致率（只看双方都分配的格子）、与 teacher 当前分配的
一致率、每事件切换率、以及"旧目标仍有效却换掉"的自由 churn 占比。

用法：
    python -m m3assign.commit_eval --ckpt runs/.../best.pt --episodes 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Store, shard_ids
from .model import build
from .train import cpu_tensors, mgeo_tensors, take

OUT = Path(__file__).resolve().parent.parent / "cache" / "commit_eval.json"


def load_oracle():
    from .sim_eval import load_routing

    # 只借 routing 里的 oracle_assignment；load_routing 需要 episode_id 集合
    return load_routing


def decode(logits: np.ndarray, valid_k: np.ndarray, valid_m: np.ndarray, M: int,
           prev: np.ndarray, margin) -> np.ndarray:
    """goal->agent 一对一分配；margin>0 时对离开旧目标加惩罚（可为每 agent 一个值）。"""
    from scipy.optimize import linear_sum_assignment

    actors = np.flatnonzero(valid_k)
    goals = np.flatnonzero(valid_m)
    goals = goals[goals < M]
    assign = np.full(M, -1, dtype=np.int64)
    if actors.size == 0 or goals.size == 0:
        return assign
    cost = -logits[np.ix_(actors, goals)].astype(np.float64)
    cost = np.where(np.isfinite(cost), cost, 1e6)
    mvec = np.asarray(margin, dtype=np.float64)
    if mvec.ndim == 0:
        mvec = np.full(4, float(mvec))
    if (mvec > 0).any():
        for i, a in enumerate(actors):
            old = prev[a]
            if old >= 0 and mvec[a] > 0:
                cost[i, goals == old] -= mvec[a]  # 保留旧目标更便宜
    r, c = linear_sum_assignment(cost)
    for i, j in zip(r, c):
        assign[int(goals[j])] = int(actors[i])
    return assign


def assign_agents(assign: np.ndarray) -> np.ndarray:
    """goal->agent 反推 agent->goal（一个 agent 只留第一个目标）。"""
    out = np.full(4, -1, dtype=np.int64)
    for g, a in enumerate(assign):
        if 0 <= a < out.size and out[a] < 0:
            out[a] = g
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--margins", type=str, default="0,0.5,1,2",
                    help="切换余量 λ 列表；0 即每帧重解")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--half", type=int, default=-1,
                    help="-1=全部；0/1=打乱后按奇偶分半，调参与检验分离")
    ap.add_argument("--mode", type=str, default="all", choices=("all", "fresh"),
                    help="all=per_frame+fresh+memory；fresh=只跑新鲜视角+余量")
    ap.add_argument("--rel-margins", type=str, default="",
                    help="尺度无关余量 α 列表：保留旧目标要求分数高出 α×该 agent 的打分标准差")
    ap.add_argument("--hold-margins", type=str, default="",
                    help="事件触发余量 λ 列表：periodic 快照冻结承诺，found/start 才用新鲜证据+λ")
    args = ap.parse_args()

    ids = shard_ids()
    val = ids[-4:]
    store = Store(val)
    by_ep: dict[str, list[int]] = {}
    for m in store.meta:
        by_ep.setdefault(m["episode_id"], []).append(m["row"])
    for rows in by_ep.values():
        rows.sort(key=lambda r: int(store.meta[r]["event_id"]))

    load_routing = load_oracle()
    infos = load_routing(set(by_ep))

    rng = np.random.RandomState(args.seed)
    eids = sorted(e for e in by_ep if e in infos)
    rng.shuffle(eids)
    eids = eids[: args.episodes]
    if args.half >= 0:
        eids = eids[args.half :: 2]
    print(f"[commit_eval] episodes={len(eids)} rows={sum(len(by_ep[e]) for e in eids)}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build(blob.get("meta")).to(dev).eval()
    model.load_state_dict(blob["state_dict"])
    tens = cpu_tensors(store)
    if ((blob.get("meta") or {}).get("args") or {}).get("mgeo"):
        tens.update(mgeo_tensors(store))
    has_contract = bool((blob.get("meta") or {}).get("args", {}).get("contract", 0))

    margins = [float(x) for x in args.margins.split(",") if x]
    rel = [float(x) for x in args.rel_margins.split(",") if x]
    hold = [float(x) for x in args.hold_margins.split(",") if x]
    variants = ["per_frame"] + [f"fresh@{m:g}" for m in margins if m > 0]
    variants += [f"rel@{a:g}" for a in rel]
    variants += [f"hold@{m:g}" for m in hold]
    if args.mode == "all":
        variants += [f"memory@{m:g}" for m in margins]
    stat = {v: dict(cells=0, oracle_hit=0, teacher_hit=0, switches=0, free=0,
                    forced=0, agents=0, events=0, cost_sum=0.0, cost_n=0,
                    corrective=0, harmful=0, costs=[], reverts=0,
                    oracle_corrective=0, oracle_harmful=0, oracle_net=0) for v in variants}
    ref_cost, teacher_cost = [], []
    gains_corr: list[float] = []
    gains_harm: list[float] = []
    # 参考线：教师当帧 / 复读教师上一帧 与 oracle 的一致率（同一批 event 上）
    refs = dict(t_hit=0, t_cells=0, c_hit=0, c_cells=0, c_switch=0)

    # 每条 episode 独立维护自己的承诺；逐事件前向（B=1）
    for eid in eids:
        info = infos[eid]
        K, M = info["K"], info["M"]
        oracle = np.asarray(info["oracle"], dtype=np.int64)
        prev = {v: np.full(4, -1, dtype=np.int64) for v in variants}
        prev2 = {v: np.full(4, -1, dtype=np.int64) for v in variants}
        tprev = np.full(4, -1, dtype=np.int64)
        for row in by_ep[eid]:
            meta = store.meta[row]
            vk = store.valid_k[row].astype(bool)
            vm = store.valid_m[row].astype(bool)
            teacher = store.teacher[row].astype(np.int64)
            travel = store.travel[row].astype(np.float64)
            idx = np.asarray([row])

            for a in range(K):
                if not vk[a]:
                    continue
                g = int(teacher[a])
                if 0 <= g < M:
                    refs["t_hit"] += int(oracle[g] == a)
                    refs["t_cells"] += 1
                old = int(tprev[a])
                if 0 <= old < M and vm[old]:
                    refs["c_hit"] += int(oracle[old] == a)
                    refs["c_cells"] += 1
                    refs["c_switch"] += int(old != g)

            def max_travel(targets: np.ndarray) -> float:
                vals, bad = [], False
                for a in range(K):
                    g = int(targets[a])
                    if vk[a] and 0 <= g < M and vm[g]:
                        if travel[a, g] >= 1e4:
                            bad = True
                        else:
                            vals.append(float(travel[a, g]))
                if bad:
                    return float("nan")
                return max(vals) if vals else float("nan")

            # 参考：BFS 匈牙利（与模型同解码器，只换代价来源）
            ref = assign_agents(decode(-travel, vk, vm, M, np.full(4, -1), 0.0))
            rc = max_travel(ref)
            if np.isfinite(rc):
                ref_cost.append(rc)
            tc = max_travel(teacher)
            if np.isfinite(tc):
                teacher_cost.append(tc)

            for v in variants:
                base = take(tens, idx, dev)
                if has_contract:
                    if v == "per_frame" or v.startswith(("fresh@", "rel@", "hold@")):
                        con = np.full(4, -1, dtype=np.int64)
                    else:
                        con = prev[v].copy()
                    # 旧目标已经被找到：不再算有效承诺
                    for a in range(4):
                        if 0 <= con[a] < M and not vm[con[a]]:
                            con[a] = -1
                    base["contract"] = torch.from_numpy(con.astype(np.int64)).to(dev)[None]
                with torch.no_grad():
                    logits = model(base)[0].float().cpu().numpy()
                if v == "per_frame":
                    margin = 0.0
                elif v.startswith("hold@"):
                    # 事件触发：periodic 快照不重评（目标仍有效的 agent 一律保持），
                    # 只有 found/start 这种真有新信息的时刻才按余量重新考虑
                    margin = 1e9 if meta.get("reason") == "periodic" else float(v.split("@")[1])
                elif v.startswith("rel@"):
                    # 尺度无关：每个 agent 的余量 = α × 它这一帧在有效目标上的打分标准差
                    acts = np.flatnonzero(vk)
                    gls = np.flatnonzero(vm)
                    gls = gls[gls < M]
                    margin = np.zeros(4)
                    if gls.size > 1:
                        margin[acts] = float(v.split("@")[1]) * logits[np.ix_(acts, gls)].std(axis=1)
                else:
                    margin = float(v.split("@")[1])
                cur = decode(logits, vk, vm, M, prev[v], margin)
                now = assign_agents(cur)
                cost_now = max_travel(now)
                if np.isfinite(cost_now):
                    stat[v]["cost_sum"] += cost_now
                    stat[v]["cost_n"] += 1
                    stat[v]["costs"].append(cost_now)
                # 只统计本帧有效的 agent
                for a in range(K):
                    if not vk[a]:
                        continue
                    stat[v]["agents"] += 1
                    old = prev[v][a]
                    new = now[a]
                    hit = old >= 0 and old < M and vm[old]
                    if old != new:
                        stat[v]["switches"] += 1
                        if prev2[v][a] == new:
                            stat[v]["reverts"] += 1  # 换回两帧前的目标：振荡
                        if hit:
                            stat[v]["free"] += 1
                            old_cost = max_travel(prev[v])
                            if np.isfinite(old_cost) and np.isfinite(cost_now):
                                if cost_now < old_cost - 1e-6:
                                    stat[v]["corrective"] += 1
                                elif cost_now > old_cost + 1e-6:
                                    stat[v]["harmful"] += 1
                            before = int(oracle[old] == a)
                            after = int(0 <= new < M and oracle[new] == a)
                            stat[v]["oracle_net"] += after - before
                            if v == "per_frame" and 0 <= new < M:
                                gain = float(logits[a, new] - logits[a, old])
                                if after > before:
                                    gains_corr.append(gain)
                                elif after < before:
                                    gains_harm.append(gain)
                            if after > before:
                                stat[v]["oracle_corrective"] += 1
                            elif after < before:
                                stat[v]["oracle_harmful"] += 1
                        else:
                            stat[v]["forced"] += 1
                    if new >= 0:
                        stat[v]["oracle_hit"] += int(oracle[new] == a)
                        stat[v]["teacher_hit"] += int(teacher[a] == new)
                        stat[v]["cells"] += 1
                prev2[v] = prev[v].copy()
                prev[v] = now
            stat[v]["events"] += 1
            tprev = teacher.copy()

    out = {}
    for v, s in stat.items():
        n = max(1, s["agents"])
        out[v] = {
            "oracle_agree": round(s["oracle_hit"] / max(1, s["cells"]), 4),
            "teacher_agree": round(s["teacher_hit"] / max(1, s["cells"]), 4),
            "switch_per_agent_event": round(s["switches"] / n, 4),
            "free_churn_share": round(s["free"] / max(1, s["switches"]), 4),
            "revert_share": round(s["reverts"] / max(1, s["switches"]), 4),
            "corrective": s["corrective"],
            "harmful": s["harmful"],
            "oracle_corrective": s["oracle_corrective"],
            "oracle_harmful": s["oracle_harmful"],
            "oracle_net": s["oracle_net"],
            "median_max_travel": round(float(np.median(s["costs"])), 3) if s["costs"] else None,
            "mean_max_travel": round(float(np.mean(s["costs"])), 3) if s["costs"] else None,
            "cells": s["cells"],
        }
    payload = {
        "ckpt": str(args.ckpt),
        "episodes": len(eids),
        "ref_median_max_travel": round(float(np.median(ref_cost)), 3) if ref_cost else None,
        "teacher_median_max_travel": round(float(np.median(teacher_cost)), 3) if teacher_cost else None,
        "switch_gain": {
            "corrective_n": len(gains_corr),
            "corrective_median": round(float(np.median(gains_corr)), 4) if gains_corr else None,
            "harmful_n": len(gains_harm),
            "harmful_median": round(float(np.median(gains_harm)), 4) if gains_harm else None,
        },
        "refs": {
            "teacher_oracle_agree": round(refs["t_hit"] / max(1, refs["t_cells"]), 4),
            "copy_prev_oracle_agree": round(refs["c_hit"] / max(1, refs["c_cells"]), 4),
            "copy_prev_switch_rate": round(refs["c_switch"] / max(1, refs["c_cells"]), 4),
        },
        "variants": out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
