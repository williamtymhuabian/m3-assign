"""把承诺决策逐事件导成回放轨迹：BEV 地图 + 各 agent 目标 + 切换判定。

供看板/独立 HTML 回放使用。每个事件记录三种解码的结果：
  per_frame   每帧用新鲜打分重解（contract 全 -1）
  fresh@λ     新鲜打分 + 余量（离开旧目标要求打分高出 λ）
  memory@0    把上一帧自己的分配喂回 contract
切换按 oracle 分配判为 纠正 / 有害 / 中性。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .assign import assign_agents, attribute_switches, decode_margin
from .data import Store, shard_ids
from .model import build
from .train import cpu_tensors, take

OUT = Path(__file__).resolve().parent.parent / "cache" / "commit_trace.json"


def bev_png(geo: np.ndarray) -> str:
    """geo:(2,96,96) -> 96x96 PNG：R=占据/可见度，G=已探索。"""
    rgb = np.zeros((96, 96, 3), np.uint8)
    rgb[..., 0] = np.clip(geo[0] * 255, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(geo[1] * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_px(xy: np.ndarray, size: int = 96) -> list[float]:
    """归一化坐标 [-2,0] -> 地图像素坐标（与 grid_sample 的 +1 仿射一致）。"""
    return [round(float((xy[0] + 2.0) / 2.0 * size), 2), round(float((xy[1] + 2.0) / 2.0 * size), 2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=5, help="导出多少条 episode")
    ap.add_argument("--scan", type=int, default=120, help="候选 episode 数")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    store = Store(shard_ids()[-4:])
    by: dict[str, list[int]] = {}
    for m in store.meta:
        by.setdefault(m["episode_id"], []).append(m["row"])
    for rows in by.values():
        rows.sort(key=lambda r: int(store.meta[r]["event_id"]))

    from .sim_eval import load_routing

    infos = load_routing(set(by))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build(blob.get("meta")).to(dev).eval()
    model.load_state_dict(blob["state_dict"])
    tens = cpu_tensors(store)
    has_contract = bool((blob.get("meta") or {}).get("args", {}).get("contract", 0))

    variants = [("per_frame", 0.0, False), (f"fresh@{args.margin:g}", args.margin, False),
                ("memory@0", 0.0, True)]

    def run_episode(eid: str) -> dict:
        info = infos[eid]
        K, M = info["K"], info["M"]
        oracle = np.asarray(info["oracle"], dtype=np.int64)
        prev = {v[0]: np.full(4, -1, dtype=np.int64) for v in variants}
        events = []
        switches = {"corr": 0, "harm": 0}
        for ev, row in enumerate(by[eid]):
            meta = store.meta[row]
            vk = store.valid_k[row].astype(bool)
            vm = store.valid_m[row].astype(bool)
            teacher = store.teacher[row].astype(np.int64)
            idx = np.asarray([row])
            var_out = {}
            fresh_logits = None
            for name, lam, use_mem in variants:
                base = take(tens, idx, dev)
                if has_contract:
                    if use_mem:
                        con = prev[name].copy()
                    else:
                        con = np.full(4, -1, dtype=np.int64)
                    for a in range(4):
                        if 0 <= con[a] < M and not vm[con[a]]:
                            con[a] = -1
                    base["contract"] = torch.from_numpy(con.astype(np.int64)).to(dev)[None]
                with torch.no_grad():
                    logits = model(base)[0].float().cpu().numpy()
                if not use_mem:
                    # 与余量无关的打分：前端据此可对任意 λ 重新滚动解码。
                    # 非有限值（不可达/已解决目标）写成 -1e6，保证产物是合法 JSON。
                    fresh_logits = [
                        [round(float(x), 4) if math.isfinite(float(x)) else -1e6 for x in row[:M]]
                        for row in logits[:K]
                    ]
                cur = decode_margin(logits, vk, vm, M, prev[name], lam)
                now = assign_agents(cur)
                per_agent = attribute_switches(logits, prev[name], now, vk, vm, M, oracle)
                for row_out in per_agent:
                    if row_out["act"] == "corr":
                        switches["corr"] += 1
                    elif row_out["act"] == "harm":
                        switches["harm"] += 1
                var_out[name] = per_agent
                prev[name] = now
            pose = store.pose[row]
            peak = store.peak[row]
            events.append({
                "t": int(meta["event_id"]),
                "reason": meta.get("reason"),
                "map": bev_png(store.geo[row]),
                "agents": [to_px(pose[a, :2]) for a in range(K)],
                "goals": [to_px(peak[j, :2]) for j in range(M)],
                "conf": [round(float(peak[j, 2]), 3) for j in range(M)],
                "valid_k": [int(x) for x in vk[:K]],
                "valid_m": [int(x) for x in vm[:M]],
                "teacher": [int(x) for x in teacher[:K]],
                "oracle": [int(x) for x in oracle[:M]],
                "logits": fresh_logits,
                "vars": var_out,
            })
        return {
            "id": eid,
            "K": K,
            "M": M,
            "oracle": [int(x) for x in oracle[:M]],
            "events": events,
            "corr": switches.get("corr", 0),
            "harm": switches.get("harm", 0),
        }

    # 先粗扫，挑"每帧重解伤得多、新鲜+余量救回来"的 episode，回放最有信息量
    cand = []
    for eid in by:
        if eid not in infos:
            continue
        if len(by[eid]) < 4:
            continue
        info = infos[eid]
        if info["K"] < 2 or info["M"] < 3:
            continue
        ep = run_episode(eid)
        p = [r for ev in ep["events"] for r in ev["vars"]["per_frame"]]
        f = [r for ev in ep["events"] for r in ev["vars"][f"fresh@{args.margin:g}"]]
        ep["_score"] = (
            sum(r["act"] == "harm" for r in p)
            - sum(r["act"] == "harm" for r in f)
            + sum(r["act"] == "corr" for r in f)
            + 0.5 * sum(r["act"] == "corr" for r in p)
        )
        cand.append(ep)
        if len(cand) >= args.scan:
            break
    cand.sort(key=lambda e: e["_score"], reverse=True)
    picked = cand[: args.episodes]
    for e in picked:
        e.pop("_score", None)
    payload = {
        "ckpt": str(args.ckpt),
        "margin": args.margin,
        "variants": [v[0] for v in variants],
        "episodes": picked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"[commit_trace] {len(picked)} episodes, "
          f"{sum(len(e['events']) for e in picked)} events -> {args.out} "
          f"({args.out.stat().st_size/1024:.0f} KB)")
    for e in picked:
        print(f"  {e['id']} K={e['K']} M={e['M']} events={len(e['events'])} corr={e['corr']} harm={e['harm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
