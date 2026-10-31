"""最小世界模型：从 BEV 地图预测"终端代价"（真测地/欧氏绕路系数 + 可达性）。

pilot（cache/headroom.json）的结论是起点帧没有信息 headroom，但代价误差集中在
10% 的拓扑误导 episode 上（欧氏说近、测地要绕）。这里只学这一件事：
        cost_pred[k,j] = euclid[k,j] * exp(log_detour_pred)
再配 min-max 解码（MSPL 是 makespan 指标，min-sum 是错的目标函数）。

监督来自 episode routing 的真测地矩阵，输入只用部署时有的东西：
BEV 两通道 + 目标热图 + agent/目标坐标。

用法：
    python -m m3assign.worldmodel --train-shards 40 --epochs 30
    python -m m3assign.worldmodel --eval-only --ckpt runs/wm-xxxx/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import load_file
from torch import nn

from .data import CACHE_DIR, K_MAX, M_MAX, SHARD_DIR, episode_meta_lite, geodesic_costs, shard_ids
from .mapgeo import goal_map_positions
from .sim_eval import fill_leftovers, hungarian_assign, load_routing, nn_order, route_len

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MAP_METERS = 12.0
INF = float("inf")


def _sample(x: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    b, q, _ = pts.shape
    grid = pts.view(b, 1, q, 2)
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.view(b, x.shape[1], q).transpose(1, 2)


class DetourNet(nn.Module):
    def __init__(self, ch: int = 32, emb: int = 96, m_max: int = M_MAX):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(2 + m_max, ch, 5, stride=2, padding=2), nn.GroupNorm(8, ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GroupNorm(8, ch), nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 2), nn.ReLU(True),
            nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1), nn.GroupNorm(8, ch * 2), nn.ReLU(True),
        )
        feat = ch * 2
        self.pair = nn.Sequential(
            nn.Linear(feat * 2 + 6, emb), nn.LayerNorm(emb), nn.GELU(),
            nn.Linear(emb, emb), nn.LayerNorm(emb), nn.GELU(),
            nn.Linear(emb, 2),  # log_detour, reach_logit
        )

    def forward(self, batch: dict) -> torch.Tensor:
        geo, heat = batch["geo"], batch["heat"]
        x = torch.cat([geo, heat], dim=1)
        f = self.trunk(x)
        b, _, _, _ = f.shape
        k, m = batch["pose"].shape[1], batch["mpos"].shape[1]
        agent_f = _sample(f, batch["pose"] + 1.0)          # B,K,2C
        goal_f = _sample(f, batch["mpos"] + 1.0)           # B,M,2C
        ag = agent_f[:, :, None].expand(b, k, m, -1)
        gf = goal_f[:, None].expand(b, k, m, -1)
        d = batch["mpos"][:, None] - batch["pose"][:, :, None]  # B,K,M,2
        pair_f = torch.cat(
            [ag, gf, d, d.abs(), batch["mvalid"][:, None, :, None].expand(b, k, m, 1),
             batch["valid_k"][:, :, None, None].expand(b, k, m, 1).float()], dim=-1
        )
        return self.pair(pair_f)  # B,K,M,2


def _episode_positions(meta: dict, lite: dict) -> tuple[np.ndarray, np.ndarray] | None:
    info = lite.get(meta["episode_id"])
    if info is None:
        return None
    starts = np.asarray([a["start_position"] for a in info["agents"]], np.float64)
    goals = np.asarray([g["position"] for g in info["goals"]], np.float64)
    if starts.shape[0] < meta["K"] or goals.shape[0] < meta["M"]:
        return None
    return starts[: meta["K"]], goals[: meta["M"]]


def build_data(shard_id_list: list[int], split: str = "start") -> dict:
    """流式读 shard，只保留 start 帧（只有它们有"当前位置→目标"的真测地）。"""
    tab = geodesic_costs()
    gidx = {e: i for i, e in enumerate(tab["ids"].tolist())}
    lite = episode_meta_lite()
    gmap = goal_map_positions()

    keep = {k: [] for k in ("geo", "heat", "pose", "mpos", "mvalid", "valid_k", "valid_m",
                            "cost", "euclid", "reach", "eid")}
    for sid in shard_id_list:
        arrs = load_file(str(SHARD_DIR / f"shard_{sid:05d}.safetensors"))
        metas = [json.loads(l) for l in
                 open(SHARD_DIR / f"shard_{sid:05d}.meta.jsonl", encoding="utf-8")]
        for i, meta in enumerate(metas):
            if meta.get("reason") != split:
                continue
            eid = meta["episode_id"]
            if eid not in gidx:
                continue
            pos = _episode_positions(meta, lite)
            if pos is None:
                continue
            starts, goals = pos
            k, m = meta["K"], meta["M"]
            sg = tab["sg"][gidx[eid]][:k, :m].astype(np.float64)
            eucl = np.linalg.norm(starts[:, None] - goals[None, :], axis=-1)
            if not np.isfinite(eucl).any() or (eucl > 1e-6).mean() < 0.5:
                continue
            mp = gmap.get(eid)
            if mp is None:
                mp = np.full((M_MAX, 2), np.nan)
            mp = np.asarray(mp, np.float32)
            mvalid = np.isfinite(mp[:, 0]).astype(np.float32)
            mp = np.nan_to_num(mp, nan=0.0)

            keep["geo"].append(arrs["geo"][i])
            keep["heat"].append(arrs["heat"][i])
            keep["pose"].append(arrs["pose"][i])
            keep["mpos"].append(mp)
            keep["mvalid"].append(mvalid)
            keep["valid_k"].append(arrs["valid_k"][i])
            keep["valid_m"].append(arrs["valid_m"][i])
            # 各 episode 的 K/M 不同，统一补到 K_MAX×M_MAX（补齐位用 inf/0 标记）
            cost_p = np.full((K_MAX, M_MAX), np.inf, np.float64)
            eucl_p = np.full((K_MAX, M_MAX), np.inf, np.float64)
            cost_p[:k, :m] = sg
            eucl_p[:k, :m] = eucl
            keep["cost"].append(cost_p)
            keep["euclid"].append(eucl_p)
            keep["reach"].append(np.isfinite(cost_p).astype(np.float32))
            keep["eid"].append(eid)

    out = {}
    for k, v in keep.items():
        if k == "eid":
            out[k] = v
        elif k in ("geo", "heat"):
            out[k] = np.stack(v).astype(np.float16)
        else:
            out[k] = np.stack(v).astype(np.float32)
    return out


def _batch(data: dict, idx: np.ndarray, device: torch.device) -> dict:
    out = {}
    for k, v in data.items():
        if k == "eid":
            continue
        t = torch.from_numpy(v[idx])
        out[k] = t.float().to(device) if t.dtype != torch.float16 else t.to(device).float()
    return out


def predict_cost(model: DetourNet, batch: dict) -> np.ndarray:
    """返回 (B,K,M) 预测代价（米），不可达为 inf。"""
    with torch.no_grad():
        out = model(batch)
    log_det, reach = out[..., 0], out[..., 1]
    eucl_m = batch["euclid"]
    cost = eucl_m * log_det.exp()
    return torch.where(reach > 0, cost, torch.full_like(cost, INF)).cpu().numpy()


def _decode_minmax(pred: np.ndarray, eucl_gg: np.ndarray, K: int, M: int) -> list[int]:
    """在预测代价上做 min-max 枚举；第二段用欧氏 gg（世界模型只预测终端代价）。"""
    import itertools

    gg = np.where(np.isfinite(eucl_gg), eucl_gg, 1e6)
    best, best_a = INF, None
    for assign in itertools.product(range(K), repeat=M):
        groups = [[] for _ in range(K)]
        for j, a in enumerate(assign):
            groups[a].append(j)
        ms = 0.0
        for a in range(K):
            order = nn_order(pred[a], gg, groups[a])
            tot, prev = 0.0, None
            bad = False
            for j in order:
                leg = pred[a][j] if prev is None else gg[prev, j]
                if not np.isfinite(leg):
                    bad = True
                    break
                tot += leg
                prev = j
            ms = max(ms, INF if bad else tot)
        if ms < best:
            best, best_a = ms, list(assign)
    return best_a


def evaluate(model: DetourNet, data: dict, device: torch.device, ckpt_name: str) -> dict:
    """回归质量 + 400 个 val episode 的分配收益。"""
    idx = np.arange(len(data["eid"]))
    preds = []
    for s in range(0, len(idx), 256):
        preds.append(predict_cost(model, _batch(data, idx[s:s + 256], device)))
    pred = np.concatenate(preds)

    valid = data["valid_k"][:, :, None] * data["valid_m"][:, None, :]
    cost, eucl, reach = data["cost"], data["euclid"], data["reach"]
    ok = (valid > 0.5) & np.isfinite(cost) & (eucl > 0.3) & (reach > 0.5)
    from scipy.stats import spearmanr
    rhos, reld = [], []
    for i in range(len(idx)):
        m = ok[i]
        if m.sum() >= 3 and np.isfinite(pred[i][m]).all():
            rhos.append(spearmanr(pred[i][m], cost[i][m]).statistic)
            reld.append(float(np.median(np.abs(pred[i][m] - cost[i][m]) / np.maximum(cost[i][m], 1e-6))))
    reg = {
        "n": int(len(idx)),
        "spearman": round(float(np.nanmean(rhos)), 4),
        "median_rel_err": round(float(np.nanmedian(reld)), 4),
    }

    # 分配层：与 headroom 同口径（真测地 + NN 评 makespan）
    infos = load_routing({e for e in data["eid"]})
    by_eid = {e: i for i, e in enumerate(data["eid"])}
    rows, seen = [], set()
    for eid, i in by_eid.items():
        if eid not in infos or eid in seen:
            continue
        seen.add(eid)
        info = infos[eid]
        sg, gg, K, M = info["sg"], info["gg"], info["K"], info["M"]
        d_star = info["d_star"]
        p = pred[i][:K, :M]
        if not np.isfinite(p).any():
            continue
        a_wm = _decode_minmax(p, np.linalg.norm(
            np.asarray([g["position"] for g in info["episode"]["goals_set"]])[:, None]
            - np.asarray([g["position"] for g in info["episode"]["goals_set"]])[None, :], axis=-1), K, M)

        def mspl(ms):
            return d_star / max(d_star, ms) if np.isfinite(ms) and ms > 0 else 0.0

        ms_wm = route_len_max(sg, gg, a_wm, K)
        eu = np.linalg.norm(
            np.asarray([a["start_position"] for a in info["episode"]["agents"]])[:, None]
            - np.asarray([g["position"] for g in info["episode"]["goals_set"]])[None, :], axis=-1)
        gg_e = np.linalg.norm(
            np.asarray([g["position"] for g in info["episode"]["goals_set"]])[:, None]
            - np.asarray([g["position"] for g in info["episode"]["goals_set"]])[None, :], axis=-1)
        a_eu = _decode_minmax(eu, gg_e, K, M)
        rows.append({
            "episode_id": eid,
            "mspl_wm": mspl(ms_wm),
            "mspl_eucl": mspl(route_len_max(sg, gg, a_eu, K)),
            "mspl_minsum": mspl(route_len_max(sg, gg,
                                 fill_leftovers(info, hungarian_assign(sg, K, M)), K)),
        })

    def mean(k):
        return round(float(np.mean([r[k] for r in rows])), 4)

    delta = np.asarray([r["mspl_wm"] - r["mspl_eucl"] for r in rows])
    return {
        "ckpt": ckpt_name,
        "regression": reg,
        "assign": {"n": len(rows), "wm_minmax": mean("mspl_wm"),
                   "eucl_minmax": mean("mspl_eucl"), "minsum_geo": mean("mspl_minsum"),
                   "delta_vs_eucl": round(float(delta.mean()), 4),
                   "win": int((delta > 0).sum()), "lose": int((delta < 0).sum())},
        "rows": rows,
    }


def route_len_max(sg, gg, assign, K):
    groups = [[] for _ in range(K)]
    for j, a in enumerate(assign):
        if 0 <= a < K:
            groups[a].append(j)
    out = []
    for a in range(K):
        order = nn_order(sg[a], gg, groups[a])
        tot, found = route_len(sg[a], gg, order)
        out.append(tot if len(found) == len(order) else INF)
    return max(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", type=int, default=40)
    ap.add_argument("--val-shards", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ids = shard_ids(limit=args.val_shards, offset=98 - args.val_shards)
    t0 = time.time()
    val = build_data(val_ids)
    print(f"[wm] val {len(val['eid'])} 个 start 样本, {time.time() - t0:.0f}s", flush=True)

    if args.eval_only:
        blob = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = DetourNet(**blob["meta"]).to(device).eval()
        model.load_state_dict(blob["state_dict"])
        res = evaluate(model, val, device, str(args.ckpt))
        print(json.dumps({k: res[k] for k in ("regression", "assign")}, ensure_ascii=False, indent=1))
        return 0

    train = build_data(shard_ids(limit=args.train_shards))
    print(f"[wm] train {len(train['eid'])} 个 start 样本, 累计 {time.time() - t0:.0f}s", flush=True)
    model = DetourNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    n = len(train["eid"])
    steps_per_epoch = max(1, n // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs)
    rng = np.random.default_rng(0)
    best = {"spearman": -9, "state": None, "epoch": -1}

    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n - args.batch + 1, args.batch):
            b = _batch(train, perm[s:s + args.batch], device)
            out = model(b)
            log_det, reach = out[..., 0], out[..., 1]
            valid = b["valid_k"][:, :, None] * b["valid_m"][:, None, :]
            fin = b["reach"] > 0.5
            m = (valid > 0.5) & fin & (b["euclid"] > 0.3)
            # 补齐位 cost/euclid 都是 inf，inf/inf=nan；先算再 where 掉，避免 nan*0
            tgt = torch.log((b["cost"] / b["euclid"].clamp_min(0.3)).clamp_min(1e-3))
            tgt = torch.where(m, tgt, torch.zeros_like(tgt))
            loss_reg = (F.huber_loss(log_det, tgt, reduction="none", delta=1.0) * m).sum() / m.sum().clamp_min(1)
            loss_cls = (F.binary_cross_entropy_with_logits(reach, b["reach"], reduction="none")
                        * (valid > 0.5)).sum() / (valid > 0.5).sum().clamp_min(1)
            loss = loss_reg + 0.5 * loss_cls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            res = evaluate(model, val, device, "val")
            print(f"[wm] ep{ep + 1} loss {tot / max(1, steps_per_epoch):.4f} "
                  f"spearman {res['regression']['spearman']:.3f} "
                  f"rel_err {res['regression']['median_rel_err']:.3f} "
                  f"| wm_minmax {res['assign']['wm_minmax']:.4f} "
                  f"eucl {res['assign']['eucl_minmax']:.4f} "
                  f"(+{res['assign']['delta_vs_eucl']:.4f}, {res['assign']['win']}W/{res['assign']['lose']}L)",
                  flush=True)
            if res["regression"]["spearman"] > best["spearman"]:
                best = {"spearman": res["regression"]["spearman"], "epoch": ep + 1,
                        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
                torch.save({"meta": {"ch": 32, "emb": 96, "m_max": M_MAX},
                            "state_dict": best["state"], "step": ep + 1}, args.out or (RUNS / "wm" / "best.pt"))

    out = args.out or (RUNS / "wm" / "best.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = evaluate(model, val, device, str(out))
    (CACHE_DIR / "wm_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("regression", "assign")}, ensure_ascii=False, indent=1))
    print(f"[wm] 写入 {out} 与 cache/wm_eval.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
