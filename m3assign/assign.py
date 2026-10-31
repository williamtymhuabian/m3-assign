"""分配解码与切换归因：不依赖 torch / 数据集，回放与离线评测共用一套语义。

margin 的语义是“离开旧目标要划算”：只在旧目标仍然有效时，给它减 λ 的代价。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

N_AGENTS = 4


def decode_margin(logits, valid_k, valid_m, M, prev, margin):
    """在合法 agent×goal 上做 -score 的一对一指派，返回长度 M 的 goal→agent（-1 为未分配）。

    `prev` 是每个 agent 上一步的目标；margin > 0 时，旧目标仍然有效的 agent
    给旧目标减去 margin 的代价，等价于要求新目标打分至少高出 margin 才切换。
    """
    actors = np.flatnonzero(valid_k)
    goals = np.flatnonzero(valid_m)
    goals = goals[goals < M]
    assign = np.full(M, -1, dtype=np.int64)
    if actors.size == 0 or goals.size == 0:
        return assign
    cost = -np.asarray(logits, dtype=np.float64)[np.ix_(actors, goals)]
    cost = np.where(np.isfinite(cost), cost, 1e6)
    if margin > 0:
        for i, a in enumerate(actors):
            old = prev[a]
            if old >= 0 and (goals == old).any():
                cost[i, goals == old] -= margin
    r, c = linear_sum_assignment(cost)
    for i, j in zip(r, c):
        assign[int(goals[j])] = int(actors[i])
    return assign


def assign_agents(assign, n_agents: int = N_AGENTS):
    """把 goal→agent 折成 agent→goal（同一 agent 只保留下标最小的目标）。"""
    out = np.full(n_agents, -1, dtype=np.int64)
    for g, a in enumerate(assign):
        if 0 <= a < out.size and out[a] < 0:
            out[a] = g
    return out


def attribute_switches(logits, prev, now, valid_k, valid_m, M, oracle):
    """逐个 agent 判 keep / corr / harm / neutral / forced / absent，并记录 gap。

    keep 的 gap = 最佳替代目标打分 − 当前目标打分（余量阈值距离）；
    切换的 gap = 新目标打分 − 旧目标打分。
    """
    rows = []
    for a in range(len(prev)):
        old, new = int(prev[a]), int(now[a])
        old_valid = 0 <= old < M and bool(valid_m[old])
        if not valid_k[a]:
            rows.append({"a": a, "old": old, "new": new, "act": "absent", "gap": 0.0})
            continue
        if old != new:
            if not old_valid:
                kind, gap = "forced", 0.0
            else:
                kind = "neutral"
                if new >= 0:
                    before = int(oracle[old] == a)
                    after = int(oracle[new] == a)
                    if after > before:
                        kind = "corr"
                    elif after < before:
                        kind = "harm"
                gap = float(logits[a, new] - logits[a, old]) if new >= 0 else 0.0
        else:
            kind = "keep"
            alt = [float(logits[a, j]) for j in range(min(M, logits.shape[1]))
                   if j != old and valid_m[j]]
            gap = max(alt) - float(logits[a, old]) if (alt and old_valid) else 0.0
        rows.append({"a": a, "old": old, "new": new, "act": kind, "gap": round(gap, 4)})
    return rows
