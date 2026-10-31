"""oracle 教师在事件之间到底改不改主意？（contract / teacher_core 逐事件动力学）

只读需要的 int64 张量（safe_open + get_slice），不搬 13 GB 的地图。
研究问题：
  1. 教师在下一个事件上保持上一承诺的比例（persistence）；
  2. 一个"只重复上一次承诺"的无记忆基线能拿到多少 teacher 准确率——
     这是任何带承诺代价的重分配策略的下界参考；
  3. 改动是不是由"目标被找到"触发的，还是纯粹的抖动（churn）。

用法：
    python -m m3assign.churn                 # 全部 98 个 shard
    python -m m3assign.churn --limit 20
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np
from safetensors import safe_open

from .data import CACHE_DIR, SHARD_DIR, shard_ids

OUT = CACHE_DIR / "churn.json.gz"
NO_COMMIT = -1
REASONS = ("start", "periodic", "found")


def load_rows(ids: list[int]) -> list[dict]:
    """每个 shard 只取标签与小特征，摊平成行记录。"""
    rows = []
    for k, sid in enumerate(ids):
        sp = SHARD_DIR / f"shard_{int(sid):05d}.safetensors"
        with safe_open(str(sp), framework="numpy") as fh:
            t = {name: fh.get_tensor(name) for name in
                 ("teacher", "teacher_core", "contract", "valid_k", "valid_m", "travel", "pose", "peak")}
        meta = [json.loads(l) for l in open(SHARD_DIR / f"shard_{int(sid):05d}.meta.jsonl",
                                           encoding="utf-8")]
        vk = t["valid_k"].astype(bool)
        vm = t["valid_m"].astype(bool)
        for i, m in enumerate(meta):
            K = int(m["K"])
            rows.append({
                "episode_id": m["episode_id"], "event_id": int(m["event_id"]),
                "reason": m["reason"], "K": K, "M": int(m["M"]), "scene": m["scene"],
                "core": t["teacher_core"][i][:K].astype(np.int64),
                "teacher": t["teacher"][i][:K].astype(np.int64),
                "contract": t["contract"][i][:K].astype(np.int64),
                "vk": vk[i][:K], "vm": vm[i][:m["M"]],
                "travel": t["travel"][i][:K, :m["M"]].astype(np.float32),
                "pose": t["pose"][i][:K].astype(np.float32),
                "peak": t["peak"][i][:m["M"], :2].astype(np.float32),
            })
        if (k + 1) % 10 == 0:
            print(f"[churn] loaded {k + 1}/{len(ids)} shards, {len(rows)} rows", flush=True)
        del t
    return rows


def group_by_episode(rows: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["episode_id"]].append(r)
    for lst in by.values():
        lst.sort(key=lambda r: r["event_id"])
    return by


def _blank() -> dict:
    return dict(cells=0, changed=0, forced=0, free=0, keep=0, miss=0,
                copy_hit=0, copy_tot=0, gmatch=0, greedy_tot=0, trans=0,
                keep_prev=0, keep_prev_hit=0, keep_self=0, keep_self_hit=0,
                b_prev_any=0, b_prev_hit=0, b_self_any=0, b_self_hit=0,
                first_free_hit=0)


def _fin(s: dict) -> dict:
    return {
        "event_transitions": s["trans"],
        "teacher_core_cells": s["cells"],
        "change_rate": round(s["changed"] / max(1, s["cells"]), 4),
        "change_forced_by_goal_resolved": s["forced"],
        "change_free_churn": s["free"],
        "free_churn_share": round(s["free"] / max(1, s["changed"]), 4),
        "persistence_keep_rate": round(s["keep"] / max(1, s["keep"] + s["miss"]), 4),
        "copy_previous_core_accuracy": round(s["copy_hit"] / max(1, s["copy_tot"]), 4),
        "greedy_min_travel_accuracy": round(s["gmatch"] / max(1, s["greedy_tot"]), 4),
        # 只看"承诺在本帧仍是待解决目标"的格子：这就是把 contract 喂给模型能吃到的信息。
        # 两个来源分开算，混在一起会把分母翻倍。
        "prev_core_still_valid_cells": s["keep_prev"],
        "oracle_keeps_prev_core_rate": round(s["keep_prev_hit"] / max(1, s["keep_prev"]), 4),
        "own_contract_still_valid_cells": s["keep_self"],
        "oracle_keeps_own_contract_rate": round(s["keep_self_hit"] / max(1, s["keep_self"]), 4),
        "baseline_carry_previous_core_acc": round(s["b_prev_hit"] / max(1, s["b_prev_any"]), 4),
        "baseline_use_own_contract_acc": round(s["b_self_hit"] / max(1, s["b_self_any"]), 4),
        "baseline_first_free_goal_acc": round(s["first_free_hit"] / max(1, s["b_prev_any"]), 4),
    }


def analyse(by: dict[str, list[dict]]) -> dict:
    """按事件间隔分桶：100K 池只是快照的子样本，相邻行未必是相邻事件。"""
    all_s, adj_s = _blank(), _blank()
    gaps: Counter = Counter()
    seq_len, eps_multi = [], 0
    for eid, lst in by.items():
        seq_len.append(len(lst))
        if len(lst) < 2:
            continue
        eps_multi += 1
        for prev, cur in zip(lst[:-1], lst[1:]):
            gap = cur["event_id"] - prev["event_id"]
            gaps[min(int(gap), 9)] += 1
            K = cur["K"]
            for tgt in ([all_s] + ([adj_s] if gap == 1 else [])):
                tgt["trans"] += 1
                for a in range(K):
                    p, c = int(prev["contract"][a]), int(cur["core"][a])
                    tp = int(prev["core"][a])
                    tgt["cells"] += 1
                    if c != tp:
                        tgt["changed"] += 1
                        if tp < 0 or tp >= cur["M"] or not bool(cur["vm"][tp]):
                            tgt["forced"] += 1
                        else:
                            tgt["free"] += 1
                    if p != NO_COMMIT and c != NO_COMMIT:
                        tgt["keep" if p == c else "miss"] += 1
                if K:
                    tgt["copy_hit"] += int(np.sum(cur["core"] == prev["core"]))
                    tgt["copy_tot"] += K
                    tt = np.where(cur["vm"][:cur["M"]], cur["travel"], np.inf)
                    for a in range(K):
                        tgt["greedy_tot"] += 1
                        if np.isfinite(tt[a]).any() and int(np.argmin(tt[a])) == int(cur["core"][a]):
                            tgt["gmatch"] += 1
                    # 两个可部署基线：用上一行的承诺 / 用本行自带的 contract
                    M = cur["M"]
                    vmg = cur["vm"][:M]

                    def first_free(taken):
                        for j in range(M):
                            if vmg[j] and j not in taken:
                                return j
                        return -1

                    for tag, src in (("prev", prev["core"]), ("self", cur["contract"])):
                        taken: set[int] = set()
                        for a in range(K):
                            tgt["b_" + tag + "_any"] += 1
                            p = int(src[a])
                            pred = p if (0 <= p < M and bool(vmg[p]) and p not in taken) else first_free(taken)
                            taken.add(pred)
                            if pred >= 0:
                                tgt["b_" + tag + "_hit"] += int(pred == int(cur["core"][a]))
                            if 0 <= p < M and bool(vmg[p]):
                                tgt[f"keep_{tag}"] += 1
                                tgt[f"keep_{tag}_hit"] += int(p == int(cur["core"][a]))
                    taken = set()
                    for a in range(K):
                        pred = first_free(taken)
                        taken.add(pred)
                        if pred == int(cur["core"][a]):
                            tgt["first_free_hit"] += 1
    return {
        "episodes": len(by), "episodes_with_2plus_rows": eps_multi,
        "rows_per_episode_median": int(np.median(seq_len)) if seq_len else 0,
        "rows_per_episode_p90": int(np.percentile(seq_len, 90)) if seq_len else 0,
        "event_gap_histogram": {str(k): int(v) for k, v in sorted(gaps.items())},
        "any_gap": _fin(all_s),
        "adjacent_events_only": _fin(adj_s),
    }


def label_baselines(by: dict[str, list[dict]]) -> dict:
    """与 model.metrics 同分母的单帧基线：只用这一行自带的字段能不能猜中 teacher 标签。

    分母 = valid_k 且 teacher 标签是有效目标（labels_of 的口径），
    所以这里的数字可以和 AssignNet 的 teacher_acc 直接比。
    """
    tot = 0
    hit = Counter()
    hist = Counter()
    for lst in by.values():
        for r in lst:
            K, M = r["K"], r["M"]
            y = r["teacher"].astype(np.int64)
            vm = r["vm"][:M]
            ok = (y >= 0) & (y < M) & vm[np.clip(y, 0, M - 1)]
            if not ok.any():
                continue
            tot += int(ok.sum())
            # 1) 直接复读这一行自带的承诺
            c = r["contract"].astype(np.int64)
            keep_ok = (c >= 0) & (c < M) & vm[np.clip(c, 0, M - 1)]
            for a in np.flatnonzero(ok):
                if keep_ok[a] and c[a] == y[a]:
                    hit["contract"] += 1
                if 0 <= y[a] < M and r["core"][a] == y[a]:
                    hit["core"] += 1
            # 2) 贪心：travel 最小（忽略撞车）
            tt = np.where(vm[None, :], r["travel"], np.inf)
            g = np.argmin(tt, axis=1) if tt.size else np.full(K, -1)
            for a in np.flatnonzero(ok):
                if np.isfinite(tt[a]).any() and int(g[a]) == int(y[a]):
                    hit["greedy_travel"] += 1
            # 3) 最近欧氏距离（把 peak 当成目标位置）
            if len(r["peak"]) and K:
                eu = np.linalg.norm(r["peak"][None, :, :] - r["pose"][:, None, :], axis=-1)
                eu = np.where(vm[None, :], eu, np.inf)
                ge = np.argmin(eu, axis=1)
                for a in np.flatnonzero(ok):
                    if np.isfinite(eu[a]).any() and int(ge[a]) == int(y[a]):
                        hit["greedy_euclid"] += 1
            # 4) 各目标出现的频率（用于最大类先验）
            for a in np.flatnonzero(ok):
                hist[int(y[a])] += 1
    out = {"cells": tot}
    for k in ("contract", "core", "greedy_travel", "greedy_euclid"):
        out[f"{k}_acc"] = round(hit[k] / max(1, tot), 4)
    out["goal_prior_acc"] = round(max(hist.values()) / max(1, tot), 4) if hist else None
    out["goal_hist"] = {str(k): int(v) for k, v in sorted(hist.items())}
    return out


def contract_semantics(by: dict[str, list[dict]]) -> dict:
    """contract 是"决策前的记忆"还是"决策后的复读"？后者等于标签泄漏。

    对每行每个在场 agent 比 contract 与：本帧 teacher、上一帧 teacher、上一帧 teacher_core。
    如果 contract 主要来自上一帧，它就是部署时可得的自身状态，可以用；
    如果它几乎等于本帧 teacher，那是同帧泄漏。
    """
    acc: dict[str, dict[str, list[int]]] = {}

    def bump(key: str, reason: str, hit: bool) -> None:
        d = acc.setdefault(key, {}).setdefault(reason, [0, 0])
        d[1] += 1
        d[0] += int(hit)

    for lst in by.values():
        prev = None
        for r in lst:
            K = r["K"]
            ok = r["vk"][:K] & (r["teacher"][:K] >= 0)
            for a in np.flatnonzero(ok):
                t = int(r["teacher"][a])
                c = int(r["contract"][a])
                bump("contract_eq_teacher", r["reason"], c == t)
                bump("contract_eq_teacher_core", r["reason"], c == int(r["core"][a]))
            if prev is not None:
                pok = prev["vk"][:K] & (prev["teacher"][:K] >= 0)
                for a in np.flatnonzero(pok):
                    c = int(r["contract"][a])
                    bump("contract_eq_prev_teacher", r["reason"], c == int(prev["teacher"][a]))
                    bump("contract_eq_prev_core", r["reason"], c == int(prev["core"][a]))
            prev = r

    def out(key: str) -> dict:
        per = acc.get(key, {})
        h = sum(v[0] for v in per.values())
        t = sum(v[1] for v in per.values())
        d = {"overall": round(h / max(1, t), 4), "cells": t}
        for reason in REASONS:
            if reason in per:
                d[reason] = round(per[reason][0] / max(1, per[reason][1]), 4)
        return d

    return {k: out(k) for k in ("contract_eq_teacher", "contract_eq_teacher_core",
                                 "contract_eq_prev_teacher", "contract_eq_prev_core")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 shard")
    ap.add_argument("--episodes", type=int, default=0, help="只分析前 N 个 episode")
    args = ap.parse_args()
    t0 = time.time()
    ids = shard_ids()
    if args.limit:
        ids = ids[: args.limit]
    rows = load_rows(ids)
    by = group_by_episode(rows)
    if args.episodes:
        keep = sorted(by)[: args.episodes]
        by = {k: by[k] for k in keep}
    rep = {"generated_at": time.time(), "shards": len(ids), "rows": len(rows)}
    rep.update(analyse(by))
    rep["single_frame_baselines"] = label_baselines(by)
    rep["contract_semantics"] = contract_semantics(by)
    import gzip
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"[churn] {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
