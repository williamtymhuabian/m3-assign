"""起点帧的反事实 headroom 分解：16pp 的缺口到底能不能被"更好的预测"吃掉。

同一个 400-episode val 集、同一套 NN 路线、同一个 d*，只换三件事：
  1. 代价来源：模型打分 / 真测地
  2. 分配目标：min-sum（匈牙利）vs min-makespan（枚举）
  3. 路线顺序：最近邻 vs 精确最优（子集 DP）

若 (真测地 + min-makespan + 最优顺序) 复现 oracle，则缺口全是"决策层"的，
不是"信息层"的——世界模型买不到；反之说明要预测的东西确实存在。

用法：
    python -m m3assign.headroom --episodes 400
输出 cache/headroom.json。
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np

from .data import Store, shard_ids
from .sim_eval import (fill_leftovers, hungarian_assign, load_routing, nn_order, route_len,
                       space_costs)

CACHE = Path(__file__).resolve().parent.parent / "cache"
INF = float("inf")


def makespan_nn(sg: np.ndarray, gg: np.ndarray, assign: list[int], K: int) -> float:
    groups = [[] for _ in range(K)]
    for j, a in enumerate(assign):
        if 0 <= a < K:
            groups[a].append(j)
    out = []
    for a in range(K):
        order = nn_order(sg[a], gg, groups[a])
        total, found = route_len(sg[a], gg, order)
        out.append(total if len(found) == len(order) else INF)
    return max(out)


def _subset_dp(sg_a: np.ndarray, gg: np.ndarray, M: int) -> np.ndarray:
    """dp[mask] = 从起点遍历 mask 的最短长度（终点不固定）。"""
    dp = np.full(1 << M, INF)
    dp[0] = 0.0
    for mask in range(1 << M):
        if not np.isfinite(dp[mask]):
            continue
        if mask == 0:
            legs = sg_a[:M]
        else:
            legs = np.full(M, INF)
            for j in range(M):
                if mask >> j & 1:
                    legs = np.minimum(legs, gg[j, :M])
        for k in range(M):
            if mask >> k & 1 or not np.isfinite(legs[k]):
                continue
            nxt = dp[mask] + legs[k]
            if nxt < dp[mask | 1 << k]:
                dp[mask | 1 << k] = nxt
    return dp


def group_masks(assign: list[int], K: int, M: int) -> list[int]:
    masks = [0] * K
    for j, a in enumerate(assign):
        if 0 <= a < K:
            masks[a] |= 1 << j
    return masks


def enum_best(sg, gg, K, M, dps=None):
    """枚举所有 K^M 分配，返回 (best_ms, best_assign, ms_array)。

    dps 非空时用精确最优访问顺序（每 agent 的 2^M 子集表已预先算好）。
    """
    vals = np.empty(K**M)
    best, best_assign = INF, None
    for i, assign in enumerate(itertools.product(range(K), repeat=M)):
        if dps is None:
            ms = makespan_nn(sg, gg, list(assign), K)
        else:
            ms = max(dps[a][m] for a, m in enumerate(group_masks(list(assign), K, M)))
        vals[i] = ms
        if ms < best:
            best, best_assign = ms, assign
    return best, best_assign, vals


def mspl(d_star: float, ms: float) -> float:
    if not np.isfinite(ms) or ms <= 0 or not d_star:
        return 0.0
    return d_star / max(d_star, ms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-shards", type=int, default=4)
    ap.add_argument("--out", type=Path, default=CACHE / "headroom.json")
    args = ap.parse_args()

    store = Store(shard_ids(limit=args.val_shards, offset=98 - args.val_shards))
    first_row = {}
    for m in store.meta:
        eid, ev = m["episode_id"], m.get("event_id", 0)
        if eid not in first_row or ev < first_row[eid][0]:
            first_row[eid] = (ev, m["row"])
    eids = sorted(first_row)
    random.Random(args.seed).shuffle(eids)
    infos = load_routing(set(eids))
    eids = [e for e in eids if e in infos][: args.episodes]

    model_ms = {}
    for f in ("sim_eval_pair_mgeo.json", "sim_eval_pair_full94.json"):
        p = CACHE / f
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for row in d.get("episodes_detail", []):
            if row["policy"] == "assignnet_hung":
                model_ms.setdefault((f, row["episode_id"]), row["makespan"])
    print(f"[headroom] {len(eids)} ep，模型候选来源 {sorted({k[0] for k in model_ms})}", flush=True)

    rows = []
    t0 = time.time()
    for i, eid in enumerate(eids):
        info = infos[eid]
        sg, gg, K, M, d_star = info["sg"], info["gg"], info["K"], info["M"], info["d_star"]

        dps = [_subset_dp(sg[a], gg, M) for a in range(K)]
        ms_min_sum = makespan_nn(sg, gg, fill_leftovers(info, hungarian_assign(sg, K, M)), K)
        ms_best_nn, a_nn, vals = enum_best(sg, gg, K, M)
        ms_best_opt, a_opt, _ = enum_best(sg, gg, K, M, dps=dps)
        ms_oracle = makespan_nn(sg, gg, list(info["oracle"]), K)

        # 部署可得代价（平面欧氏）下的两种目标，用真测地评
        sg_e, gg_e = space_costs(info)
        dps_e = [_subset_dp(sg_e[a], gg_e, M) for a in range(K)]
        _, a_eucl_ms, _ = enum_best(sg_e, gg_e, K, M, dps=dps_e)
        ms_eucl_ms = makespan_nn(sg, gg, list(a_eucl_ms), K)
        from .sim_eval import hungarian_on_costs
        assign_eucl_sum = fill_leftovers(
            info, hungarian_on_costs(sg_e, np.ones(K, bool), np.ones(M, bool), M)
        )
        ms_eucl_sum = makespan_nn(sg, gg, assign_eucl_sum, K)

        # 更强的 min-sum 基线：目标换成"总路线长度"（含 goal->goal），仍按真测地
        def best_sum_assignment(dps_):
            best_s, best_a = INF, None
            for assign in itertools.product(range(K), repeat=M):
                masks = group_masks(list(assign), K, M)
                tot = sum(dps_[a][m] for a, m in enumerate(masks))
                if tot < best_s:
                    best_s, best_a = tot, assign
            return list(best_a)

        a_geo_sum = best_sum_assignment(dps)
        ms_geo_sumroute = makespan_nn(sg, gg, a_geo_sum, K)
        a_e_sum = best_sum_assignment(dps_e)
        ms_e_sumroute = makespan_nn(sg, gg, a_e_sum, K)

        finite = np.sort(vals[np.isfinite(vals)])
        gap12 = float(finite[1] - finite[0]) if finite.size > 1 else 0.0

        row = {
            "episode_id": eid,
            "K": K,
            "M": M,
            "d_star": float(d_star),
            "oracle_assign_ms": ms_oracle,
            "min_sum_ms": ms_min_sum,
            "best_nn_ms": ms_best_nn,
            "best_opt_ms": ms_best_opt,
            "eucl_minmax_ms": ms_eucl_ms,
            "geo_sumroute_ms": ms_geo_sumroute,
            "eucl_sumroute_ms": ms_e_sumroute,
            "eucl_minsum_ms": ms_eucl_sum,
            "top2_gap_ms": gap12,
            "median_ms": float(np.median(vals)),
            "p90_ms": float(np.percentile(vals, 90)),
            "worst_ms": float(np.max(vals)),
            "assign_min_sum": fill_leftovers(info, hungarian_assign(sg, K, M)),
            "assign_best_nn": list(map(int, a_nn)),
            "assign_best_opt": list(map(int, a_opt)),
            "mspl_oracle": mspl(d_star, ms_oracle),
            "mspl_min_sum": mspl(d_star, ms_min_sum),
            "mspl_best_nn": mspl(d_star, ms_best_nn),
            "mspl_best_opt": mspl(d_star, ms_best_opt),
            "mspl_eucl_minmax": mspl(d_star, ms_eucl_ms),
            "mspl_eucl_minsum": mspl(d_star, ms_eucl_sum),
            "mspl_geo_sumroute": mspl(d_star, ms_geo_sumroute),
            "mspl_eucl_sumroute": mspl(d_star, ms_e_sumroute),
        }
        for f in ("sim_eval_pair_mgeo.json", "sim_eval_pair_full94.json"):
            if (f, eid) in model_ms:
                row[f"model_ms:{f}"] = model_ms[(f, eid)]
                row[f"mspl_model:{f}"] = mspl(d_star, model_ms[(f, eid)])
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"[headroom] {i + 1}/{len(eids)}, {time.time() - t0:.0f}s", flush=True)

    def agg(key: str) -> dict | None:
        vals = [r[key] for r in rows if key in r]
        if not vals:
            return None
        v = np.asarray(vals, dtype=float)
        return {"mean": round(float(np.mean(v)), 4), "median": round(float(np.median(v)), 4),
                "n": int(v.size)}

    policies = [k for k in rows[0] if k.startswith("mspl_")]
    summary = {k: agg(k) for k in policies}
    summary["model_full94"] = agg("mspl_model:sim_eval_pair_full94.json")
    summary["model_mgeo"] = agg("mspl_model:sim_eval_pair_mgeo.json")

    match = np.asarray([abs(r["oracle_assign_ms"] - r["d_star"]) / max(r["d_star"], 1e-9) for r in rows])
    print("\n[headroom] oracle 分配能否由枚举复现（相对 d* 的误差）: "
          f"中位 {np.median(match):.4f}，p90 {np.percentile(match, 90):.4f}", flush=True)
    for k in sorted(summary):
        if summary[k]:
            print(f"  {k:22s} mean {summary[k]['mean']:.4f}  median {summary[k]['median']:.4f}", flush=True)

    payload = {"generated_at": time.time(), "episodes": len(rows), "summary": summary,
               "oracle_repro_err": {"median": float(np.median(match)),
                                    "p90": float(np.percentile(match, 90))},
               "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[headroom] 写入 {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
