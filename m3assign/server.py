"""实时看板后端：标准库 HTTP 服务，提供训练指标、episode 轨迹、目标热图与 IIN 参考图。"""
from __future__ import annotations

import argparse
import base64
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from safetensors import safe_open

from .data import (
    IIN_DIR,
    K_MAX,
    M_MAX,
    SHARD_DIR,
    Store,
    episode_meta_lite,
    iin_map,
    shard_ids,
)
from .mapgeo import goal_map_positions
from .model import AssignNet, build
from .trace import locate, shard_meta, trace_for

WEB = Path(__file__).resolve().parent / "web"
RUNS = Path(__file__).resolve().parent.parent / "runs"
SIM_EVAL = Path(__file__).resolve().parent.parent / "cache" / "sim_eval.json"
NAVMESH_CHECK = Path(__file__).resolve().parent.parent / "cache" / "navmesh_check.json"
AGENT_COLORS = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399"]
GOAL_COLORS = ["#ef4444", "#f97316", "#eab308", "#06b6d4", "#8b5cf6"]


def _finite(obj):
    """递归把 NaN/Inf 换成 null。

    无效 agent/目标的 peak 第三维和 travel 就是 NaN，而 Python 的 json.dumps 会原样写出
    `NaN` —— 那不是合法 JSON，浏览器 response.json() 会整段抛错，episode 直接白屏。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class Backend:
    def __init__(self, browse_shards: int, ckpt: str | None):
        ids = shard_ids(limit=browse_shards)
        print(f"[board] loading browse shards {ids[:3]}...{ids[-1]}", flush=True)
        self.store = Store(ids)
        print(f"[board] rows={self.store.n} episodes={len(self.store.episodes())}", flush=True)
        self.episodes_meta = episode_meta_lite()
        self.iin = iin_map()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AssignNet().to(self.device).eval()
        self.ckpt_step = None
        self.ckpt_path = Path(ckpt) if ckpt else None
        self.ckpt_mtime = 0.0
        self.maybe_load_ckpt()

    def maybe_load_ckpt(self) -> None:
        """训练每写一次 best.pt，看板就跟到最新权重。"""
        if not self.ckpt_path or not self.ckpt_path.exists():
            return
        mtime = self.ckpt_path.stat().st_mtime
        if mtime == self.ckpt_mtime:
            return
        try:
            blob = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            meta = blob.get("meta") or {}
            args = meta.get("args") or {}
            arch = (
                int(args.get("couple_iters", 0)),
                bool(args.get("contract", 0)),
                bool(args.get("mgeo", 0)),
            )
            if arch != (self.model.couple_iters, self.model.use_contract, self.model.use_mgeo):
                self.model = build(meta).to(self.device).eval()  # 训练换了头，看板跟着重建
            self.model.load_state_dict(blob["state_dict"])
        except (RuntimeError, KeyError) as exc:
            print(f"[board] ckpt 读取失败（稍后重试）：{exc}", flush=True)
            return
        self.ckpt_mtime = mtime
        self.ckpt_step = meta.get("step")
        print(f"[board] checkpoint {self.ckpt_path} step={self.ckpt_step}", flush=True)

    @torch.no_grad()
    def predict(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.maybe_load_ckpt()
        t = {
            "geo": torch.from_numpy(self.store.geo[rows]).to(self.device).float(),
            "heat": torch.from_numpy(self.store.heat[rows]).to(self.device).float(),
            "pose": torch.from_numpy(self.store.pose[rows]).to(self.device).float(),
            "peak": torch.from_numpy(self.store.peak[rows]).to(self.device).float(),
            "travel": torch.from_numpy(self.store.travel[rows]).to(self.device).float(),
            "valid_k": torch.from_numpy(self.store.valid_k[rows].astype(np.bool_)).to(self.device),
            "valid_m": torch.from_numpy(self.store.valid_m[rows].astype(np.bool_)).to(self.device),
        }
        if self.model.use_contract:
            t["contract"] = torch.from_numpy(self.store.contract[rows].astype(np.int64)).to(self.device)
        if self.model.use_mgeo:
            gm = goal_map_positions()
            mpos = np.zeros((len(rows), M_MAX, 2), np.float32)
            mvalid = np.zeros((len(rows), M_MAX), np.float32)
            for i, r in enumerate(rows):
                p = gm.get(self.store.meta[r]["episode_id"])
                if p is None:
                    continue
                ok = np.isfinite(p[:, 0])
                mpos[i] = np.where(ok[:, None], np.nan_to_num(p), 0.0)
                mvalid[i] = ok
            t["mpos"] = torch.from_numpy(mpos).to(self.device)
            t["mvalid"] = torch.from_numpy(mvalid).to(self.device)
        probs = self.model(t).softmax(dim=-1).cpu().numpy()
        return probs.argmax(-1), probs.max(-1)

    def episode_list(self, query: str, limit: int) -> list[dict]:
        out = []
        q = query.lower()
        for ep in self.store.episodes():
            meta = self.episodes_meta.get(ep)
            if not meta:
                continue
            if q and q not in ep.lower() and q not in meta["scene_key"].lower():
                continue
            goals = meta["goals"]
            out.append(
                {
                    "episode_id": ep,
                    "scene": meta["scene_key"],
                    "k": meta["num_agents"],
                    "m": len(goals),
                    "frames": len(self.store.episode_rows[ep]),
                    "categories": [g["category"] for g in goals],
                    "modalities": [g["modality"] for g in goals],
                    "oracle_makespan": meta.get("oracle_makespan"),
                }
            )
            if len(out) >= limit:
                break
        return out

    def episode_frames(self, ep: str) -> dict | None:
        rows = self.store.rows_of(ep)
        if rows.size == 0:
            return None
        meta = self.episodes_meta.get(ep, {})
        probs, conf = self.predict(rows)
        metas = {m["row"]: m for m in self.store.meta}
        frames = []
        for j, r in enumerate(rows):
            geo = self.store.geo[r]
            small = geo[:, ::2, ::2].transpose(1, 2, 0)
            small8 = np.clip(small * 255, 0, 255).astype(np.uint8)
            sm = metas.get(int(r), {})
            frames.append(
                {
                    "row": int(r),
                    "shard": sm.get("shard"),
                    "offset": sm.get("offset"),
                    "event_id": sm.get("event_id"),
                    "reason": sm.get("reason"),
                    "geo": base64.b64encode(small8.tobytes()).decode(),
                    "pose": np.round(self.store.pose[r], 4).tolist(),
                    "peak": np.round(self.store.peak[r], 4).tolist(),
                    "travel": np.round(np.minimum(self.store.travel[r], 9999), 2).tolist(),
                    "teacher": self.store.teacher[r].tolist(),
                    "teacher_core": self.store.teacher_core[r].tolist(),
                    "contract": self.store.contract[r].tolist(),
                    "valid_k": self.store.valid_k[r].astype(int).tolist(),
                    "valid_m": self.store.valid_m[r].astype(int).tolist(),
                    "pred": probs[j].tolist(),
                    "conf": np.round(conf[j], 3).tolist(),
                }
            )
        goals = []
        for g in meta.get("goals", []):
            fn = self.iin.get(f"{ep}#{g['index']}")
            goals.append(
                {
                    **g,
                    "iin_file": fn,
                    "iin_exists": bool(fn) and (IIN_DIR / fn).exists(),
                    "color": GOAL_COLORS[g["index"] % len(GOAL_COLORS)],
                }
            )
        return {
            "episode_id": ep,
            "meta": {k: v for k, v in meta.items() if k != "goals"},
            "goals": goals,
            "agents": [{"i": i, "color": AGENT_COLORS[i % 4]} for i in range(K_MAX)],
            "frames": frames,
            "ckpt_step": self.ckpt_step,
        }

    def train_step(self, step: int, slot: int, goal: int, run: str = "latest") -> dict:
        run_dir = RUNS / run
        if not (run_dir / "state.json").exists() and run_dir.is_symlink():
            run_dir = run_dir.resolve()
        steps, state = trace_for(run_dir)
        latest = int(state.get("step") or len(steps))
        step = max(1, min(int(step or 1), min(len(steps), latest)))
        info = steps[step - 1]
        rows = info["rows"]
        slot = int(slot or 0) % int(rows.size)
        row = int(rows[slot])
        sid, off = locate(row)
        meta = shard_meta(sid)[off]
        with safe_open(str(SHARD_DIR / f"shard_{sid:05d}.safetensors"), framework="np") as f:
            geo = f.get_slice("geo")[off].astype(np.float32)
            heat = f.get_slice("heat")[off].astype(np.float32)
            pose = np.asarray(f.get_slice("pose")[off], dtype=np.float32)
            peak = np.asarray(f.get_slice("peak")[off], dtype=np.float32)
            teacher = np.asarray(f.get_slice("teacher")[off])
            contract = np.asarray(f.get_slice("contract")[off])
            valid_k = np.asarray(f.get_slice("valid_k")[off]).astype(bool)
            valid_m = np.asarray(f.get_slice("valid_m")[off]).astype(bool)

        valid_goals = np.flatnonzero(valid_m)
        goal = int(goal or 0)
        if goal not in valid_goals:
            goal = int(valid_goals[0]) if valid_goals.size else 0
        ep_id = meta.get("episode_id")
        ep = self.episodes_meta.get(ep_id) or {}
        gmeta = {int(g.get("index", -1)): g for g in (ep.get("goals") or [])}
        # 只有 IIN 模态的目标在训练集里带真实图片；TN/ON 目标只有文字描述
        goals = [
            {
                "i": int(i),
                "mod": (gmeta.get(i) or {}).get("modality") or "?",
                "cat": (gmeta.get(i) or {}).get("category") or "",
                "img": f"{ep_id}#{i}" in self.iin,
                "valid": bool(valid_m[i]),
            }
            for i in range(valid_m.size)
        ]
        h = heat[goal]
        peak_h = float(np.nanmax(h))
        hq = np.clip(h / max(peak_h, 1e-3) * 255, 0, 255).astype(np.uint8)
        geo8 = np.clip(geo[:, ::2, ::2].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        return {
            "step": step,
            "total_steps": len(steps),
            "epoch": info.get("epoch"),
            "slot": slot,
            "batch": int(rows.size),
            "row": row,
            "shard": sid,
            "offset": off,
            "episode_id": meta.get("episode_id"),
            "scene": meta.get("scene"),
            "event_id": meta.get("event_id"),
            "reason": meta.get("reason"),
            "goal": goal,
            "goal_category": next((g["cat"] for g in goals if g["i"] == goal), ""),
            "goals": goals,
            "valid_m": valid_m.astype(int).tolist(),
            "geo": base64.b64encode(geo8.tobytes()).decode(),
            "heat": base64.b64encode(hq.tobytes()).decode(),
            "heat_peak": round(peak_h, 4),
            "pose": np.round(pose, 4).tolist(),
            "peak": np.round(peak, 4).tolist(),
            "teacher": teacher.tolist(),
            "contract": contract.tolist(),
            "valid_k": valid_k.astype(int).tolist(),
        }

    def heat(self, row: int, goal: int) -> dict | None:
        if row < 0 or row >= self.store.n or not (0 <= goal < M_MAX):
            return None
        h = self.store.heat[row, goal].astype(np.float32)
        peak = float(h.max())
        # 热图原始幅值常低于 0.5，按本帧峰值归一化后才看得清响应区域
        q = np.clip(h / max(peak, 1e-3) * 255, 0, 255).astype(np.uint8)
        return {
            "shape": list(h.shape),
            "peak": round(peak, 4),
            "data": base64.b64encode(q.tobytes()).decode(),
        }


Handler: Backend | None = None


class Req(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *a) -> None:  # 安静
        return

    def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(
            code, json.dumps(_finite(obj), ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8"
        )

    def do_GET(self) -> None:  # noqa: N802
        be = Handler
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        path = u.path
        try:
            if path in ("/", "/index.html"):
                self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path in ("/replay", "/replay.html"):
                self._send(200, (WEB / "replay.html").read_bytes(), "text/html; charset=utf-8")
            elif path.startswith("/web/"):
                name = path[len("/web/") :]
                if "/" in name or ".." in name:
                    return self._send(400, b"bad path", "text/plain")
                f = WEB / name
                if not f.exists():
                    return self._send(404, b"not found", "text/plain")
                ctype = "text/css" if f.suffix == ".css" else "application/javascript" if f.suffix == ".js" else "text/html"
                self._send(200, f.read_bytes(), f"{ctype}; charset=utf-8")
            elif path == "/api/state":
                self._json(read_state(q.get("run", "latest")))
            elif path == "/api/train_step":
                self._json(
                    be.train_step(
                        q.get("step", "1"),
                        q.get("slot", "0"),
                        q.get("goal", "0"),
                        q.get("run", "latest"),
                    )
                )
            elif path == "/api/runs":
                self._json({"runs": runs_list()})
            elif path == "/api/history":
                self._json(read_history(q.get("run", "latest"), int(q.get("limit", "600"))))
            elif path == "/api/sim":
                self._json(read_sim())
            elif path == "/api/episodes":
                self._json({"episodes": be.episode_list(q.get("q", ""), int(q.get("limit", "80")))})
            elif path == "/api/episode":
                d = be.episode_frames(q["id"])
                self._json(d if d else {"error": "unknown episode"}, 200 if d else 404)
            elif path == "/api/heat":
                d = be.heat(int(q["row"]), int(q["goal"]))
                self._json(d if d else {"error": "bad row"}, 200 if d else 404)
            elif path == "/api/iin":
                fn = be.iin.get(f"{q['episode']}#{int(q['goal'])}")
                if not fn:
                    return self._json({"error": "no iin"}, 404)
                real = (IIN_DIR / fn).resolve()
                if not str(real).startswith(str(IIN_DIR.resolve())) or not real.exists():
                    return self._send(404, b"missing", "text/plain")
                # IIN 图是静态文件，跟随训练时会反复请求同一 episode 的图，允许浏览器缓存
                self._send(200, real.read_bytes(), "image/png", cache="max-age=86400")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:  # 单一页面服务，错误直接返回即可
            self._json({"error": type(exc).__name__, "detail": str(exc)}, 500)


def read_sim() -> dict:
    """navmesh 测地几何下的分配策略评测（sim_eval.py 产出），文件更新即生效。

    顺带把 verify_navmesh.py 的几何核对结果并进同一个接口，前端少一次往返。
    """
    if not SIM_EVAL.exists():
        return {"status": "not-run"}
    d = json.loads(SIM_EVAL.read_text(encoding="utf-8"))
    if NAVMESH_CHECK.exists():
        d["geometry_check"] = json.loads(NAVMESH_CHECK.read_text(encoding="utf-8"))["agg"]
    return d


def read_state(run: str) -> dict:
    root = RUNS / run
    cand = root / "state.json"
    if not cand.exists() and run == "latest" and root.is_symlink():
        cand = Path(str(root.resolve())) / "state.json"
    if not cand.exists():
        return {"status": "no-run", "history_dir": str(root)}
    st = json.loads(cand.read_text(encoding="utf-8"))
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        st["gpu_free_mb"] = round(free / 1e6)
        st["gpu_total_mb"] = round(total / 1e6)
        st["gpu_used_mb"] = round((total - free) / 1e6)
    st["server_time"] = time.time()
    return st


def runs_list() -> list[dict]:
    out = []
    for d in sorted(RUNS.iterdir(), reverse=True):
        if d.is_symlink() or not d.is_dir() or not (d / "state.json").exists():
            continue
        try:
            st = json.loads((d / "state.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "run": d.name,
                "status": st.get("status"),
                "step": st.get("step"),
                "total_steps": st.get("total_steps"),
                "n_train": st.get("n_train"),
                "started_at": st.get("started_at"),
            }
        )
    return out


def read_history(run: str, limit: int) -> dict:
    root = (RUNS / run).resolve() if (RUNS / run).is_symlink() else RUNS / run
    f = root / "history.jsonl"
    if not f.exists():
        return {"rows": []}
    lines = f.read_text(encoding="utf-8").splitlines()[-limit:]
    return {"rows": [json.loads(x) for x in lines if x.strip()]}


def main() -> None:
    global Handler
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--browse-shards", type=int, default=6)
    ap.add_argument("--ckpt", type=str, default=str(RUNS / "latest" / "best.pt"))
    args = ap.parse_args()
    Handler = Backend(args.browse_shards, args.ckpt)
    srv = ThreadingHTTPServer((args.host, args.port), Req)
    print(f"[board] http://{args.host}:{args.port}/  (Ctrl+C 退出)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
