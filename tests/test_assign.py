import json
from pathlib import Path

import numpy as np
import pytest

from m3assign.assign import assign_agents, attribute_switches, decode_margin

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "commit_trace.sample.json"

ACTS = {"keep", "corr", "harm", "neutral", "forced", "absent"}


def _logits(rows):
    return np.asarray(rows, dtype=np.float64)


def test_margin_keeps_previous_goal_against_small_gain():
    # 每个 agent 换成另一目标的收益都是 0.2；λ=0.5 应该压住这次切换
    logits = _logits([[1.0, 1.2], [1.2, 1.0]])
    prev = np.array([0, 1])
    vk, vm = np.ones(2, bool), np.ones(2, bool)

    greedy = decode_margin(logits, vk, vm, M=2, prev=prev, margin=0.0)
    sticky = decode_margin(logits, vk, vm, M=2, prev=prev, margin=0.5)
    assert greedy.tolist() == [1, 0]
    assert sticky.tolist() == [0, 1]


def test_margin_does_not_apply_when_old_goal_invalid():
    # prev 指向的目标已不可见：不享受余量，结果应与 margin=0 完全一致
    logits = _logits([[1.0, 1.2], [0.4, 0.9]])
    prev = np.array([1, 1])
    vk = np.ones(2, bool)
    vm = np.array([True, False])
    with_margin = decode_margin(logits, vk, vm, M=2, prev=prev, margin=5.0)
    without = decode_margin(logits, vk, vm, M=2, prev=prev, margin=0.0)
    assert with_margin.tolist() == without.tolist() == [0, -1]


def test_invalid_agents_and_goals_are_excluded():
    logits = _logits([[5.0, 0.1, 9.9], [8.0, 8.0, 8.0]])
    vk = np.array([True, False])
    vm = np.array([True, True, False])
    assign = decode_margin(logits, vk, vm, M=3, prev=np.array([-1, -1]), margin=0.0)
    assert assign.tolist() == [0, -1, -1]  # goal0 唯一合法目标；无效 agent 与不可见目标都不参与


def test_ties_still_produce_complete_valid_assignment():
    logits = _logits([[1.0, 1.0], [1.0, 1.0]])
    vk, vm = np.ones(2, bool), np.ones(2, bool)
    for margin in (0.0, 0.3):
        assign = decode_margin(logits, vk, vm, M=2, prev=np.array([-1, -1]), margin=margin)
        assert sorted(assign.tolist()) == [0, 1]


def test_assign_agents_dedupes_by_goal_index():
    assert assign_agents(np.array([0, 0, 2])).tolist() == [0, -1, 2, -1]
    assert assign_agents(np.array([0, 3])).tolist() == [0, -1, -1, 1]
    assert assign_agents(np.array([-1, -1])).tolist() == [-1, -1, -1, -1]


def test_attribute_switches_labels_and_gaps():
    logits = _logits([
        [2.0, 0.5, 0.1],   # agent0：保持 goal0
        [0.1, 0.4, 1.0],   # agent1：1→2，走向自己的 oracle 目标
        [0.2, 0.9, 0.3],   # agent2：2→1，但两个都不是它的 oracle 目标
        [0.1, 0.2, 0.3],   # agent3：无效
    ])
    oracle = np.array([0, -1, 1])  # agent1 没有 oracle 目标，agent2 的 oracle 目标是 goal1
    prev = np.array([0, 1, 2, 0])
    now = np.array([0, 2, 1, 1])
    vk = np.array([True, True, True, False])
    vm = np.ones(3, bool)

    rows = attribute_switches(logits, prev, now, vk, vm, M=3, oracle=oracle)
    act = [r["act"] for r in rows]
    assert act == ["keep", "corr", "neutral", "absent"]
    assert rows[0]["gap"] == pytest.approx(0.5 - 2.0)  # 最佳替代 − 当前
    assert rows[1]["gap"] == pytest.approx(1.0 - 0.4)  # 新目标 − 旧目标
    assert rows[2]["gap"] == pytest.approx(0.9 - 0.3)
    assert rows[3]["gap"] == 0.0


def test_attribute_switches_flags_leaving_oracle_goal_as_harm():
    logits = _logits([[1.0, 0.2], [0.1, 1.0]])
    oracle = np.array([0, 1])
    rows = attribute_switches(
        logits, np.array([0, 1]), np.array([0, 0]),
        np.ones(2, bool), np.ones(2, bool), M=2, oracle=oracle,
    )
    assert [r["act"] for r in rows] == ["keep", "harm"]


def test_attribute_switches_forced_and_neutral():
    logits = _logits([[0.3, 0.2], [0.9, 0.1]])
    oracle = np.array([0, -1])
    prev = np.array([-1, 0])       # agent0 第一次选目标；agent1 离开一个不在 oracle 上的目标
    now = np.array([1, 1])
    vk = np.ones(2, bool)
    vm = np.ones(2, bool)
    rows = attribute_switches(logits, prev, now, vk, vm, M=2, oracle=oracle)
    assert [r["act"] for r in rows] == ["forced", "neutral"]
    assert rows[0]["gap"] == 0.0


@pytest.mark.skipif(not SAMPLE.exists(), reason="样例轨迹未随包提供")
def test_sample_trace_schema():
    blob = json.loads(SAMPLE.read_text(encoding="utf-8"))
    assert {"ckpt", "margin", "variants", "episodes"} <= set(blob)
    assert len(blob["variants"]) == 3
    assert any(v.startswith("fresh@") for v in blob["variants"])
    assert blob["episodes"], "样例至少包含一条 episode"

    seen_act = set()
    for ep in blob["episodes"]:
        K, M = ep["K"], ep["M"]
        assert 0 < K <= 4 and 0 < M <= 5
        assert len(ep["oracle"]) == M
        assert ep["events"]
        for ev in ep["events"]:
            assert ev["reason"] in {"start", "periodic", "found"}
            assert len(ev["agents"]) == K and len(ev["goals"]) == M
            for xy in ev["agents"] + ev["goals"]:
                assert len(xy) == 2 and all(0.0 <= v <= 96.0 for v in xy)
            assert len(ev["valid_k"]) == K and len(ev["valid_m"]) == M
            assert len(ev["teacher"]) == K and len(ev["oracle"]) == M
            assert len(ev["logits"]) == K and all(len(r) == M for r in ev["logits"])
            assert set(ev["vars"]) == set(blob["variants"])
            for rows in ev["vars"].values():
                assert len(rows) == K
                for r in rows:
                    assert r["act"] in ACTS
                    seen_act.add(r["act"])
    # 样例被挑成“每帧重解会乱切、加余量能救”的 episode，因此不全是初始帧
    assert "keep" in seen_act
