"""HM3D-M3 官方 teacher_100k 样本的特征质量审计。

逐 shard 流式扫描（单 shard ~132 MB，峰值内存不超过一个 shard），检查
SCHEMA.md 声称的字段语义在数据里是否成立。结果打印成人读表格，同时写
cache/audit_teacher.json.gz 供报告引用。

用法：
    python -m m3assign.audit_data                 # 均匀抽 12 个 shard
    python -m m3assign.audit_data --limit 24
    python -m m3assign.audit_data --geom 300      # 追加 travel 与 navmesh 真测地的相关性
"""
from __future__ import annotations

import argparse
import gzip
import json
import time

import numpy as np
from safetensors.numpy import load_file
from scipy.stats import spearmanr

from .data import CACHE_DIR, M_MAX, MAP_SIZE, SHARD_DIR, UNREACHABLE, shard_ids

OUT = CACHE_DIR / "audit_teacher.json.gz"
REASONS = ("start", "periodic", "found")


def med(v):
    return round(float(np.median(v)), 4) if len(v) else None


def cell(u):
    """归一化坐标（帧 [-2,0]，与 m3assign.model 的 +1 仿射一致）-> 96x96 单元格坐标。"""
    return (np.asarray(u, dtype=np.float64) + 2.0) * MAP_SIZE / 2.0 - 0.5


def nearest_dists(cells, pts, rad=None):
    """pts:(Q,2) 单元格坐标 -> 到 cells（非零格集合）最近距离。"""
    if len(cells) == 0:
        return None
    out = []
    for p in np.asarray(pts, dtype=np.float64):
        d = np.hypot(cells[:, 1] - p[0], cells[:, 0] - p[1])
        out.append(float(d.min()))
    return np.asarray(out, dtype=np.float64)


class Reg:
    """坐标配准检验的累加器。"""

    def __init__(self) -> None:
        self.pose_hit, self.pose_null = [], []
        self.peak_hit, self.peak_null = [], []
        self.axis = {"x~col": [], "x~row": [], "y~row": [], "y~col": []}

    def out(self) -> dict:
        def blk(h, n):
            h, n = np.asarray(h, float), np.asarray(n, float)
            if not len(h):
                return {}
            return {"n": int(len(h)),
                    "median_cells": round(float(np.median(h)), 2),
                    "random_baseline_median_cells": round(float(np.median(n)), 2),
                    "frac_within_5": round(float((h <= 5).mean()), 3),
                    "random_baseline_frac_within_5": round(float((n <= 5).mean()), 3)}
        r = {"pose_vs_explored": blk(self.pose_hit, self.pose_null),
             "peak_vs_heat_blob": blk(self.peak_hit, self.peak_null),
             "within_row_axis_spearman": {}}
        for k, v in self.axis.items():
            v = np.asarray(v, float)
            if len(v):
                r["within_row_axis_spearman"][k] = {
                    "n_rows": int(len(v)), "median": round(float(np.median(v)), 3),
                    "frac_positive": round(float((v > 0).mean()), 3)}
        return r


class Acc:
    def __init__(self) -> None:
        self.n = 0
        self.ch0, self.ch1, self.heat = [], [], []
        self.tv_zero, self.tv_unreach = [], []
        self.zero_own, self.zero_other = [], []
        self.peak_u, self.eu_rho, self.geo_rho = [], [], []

    def out(self) -> dict:
        if not self.n:
            return {}
        return {
            "rows": self.n,
            "geo_ch0_nonzero": med(self.ch0),
            "geo_ch1_nonzero": med(self.ch1),
            "heat_nonzero": med(self.heat),
            "travel_eq_zero": med(self.tv_zero),
            "travel_unreachable": med(self.tv_unreach),
            "travel_eq_zero_if_teacher_here": med(self.zero_own),
            "travel_eq_zero_if_not": med(self.zero_other),
            "peak_u_median": med(self.peak_u),
            "spearman_travel_vs_euclid": med(self.eu_rho),
            "spearman_travel_vs_geodesic": med(self.geo_rho),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="扫描全部 98 个 shard")
    ap.add_argument("--limit", type=int, default=12, help="均匀抽样 shard 数")
    ap.add_argument("--geom", type=int, default=0, metavar="N",
                    help="对 N 个 start 帧 episode 追加真测地相关性（读 160 MB episode 文件）")
    args = ap.parse_args()

    all_ids = shard_ids()
    step = max(1, len(all_ids) // args.limit)
    ids = all_ids if args.all else all_ids[::step][: args.limit]
    t0 = time.time()

    # 1) 先把 meta 读干净（很便宜），确定要用哪些 episode 做真测地对照
    metas = {}
    for sid in ids:
        mp = SHARD_DIR / f"shard_{int(sid):05d}.meta.jsonl"
        metas[int(sid)] = [json.loads(l) for l in open(mp, encoding="utf-8")]
    routing = {}
    if args.geom:
        from .sim_eval import load_routing

        cand = sorted({m["episode_id"] for ms in metas.values() for m in ms
                       if m["reason"] == "start"})
        keep = set(cand[: args.geom])
        routing = load_routing(keep)
        print(f"[audit] 真测地对照 {len(routing)} 个 episode, {time.time() - t0:.0f}s", flush=True)

    # 2) 逐 shard 扫描
    per_reason = {r: Acc() for r in REASONS}
    cov = {"rows": 0, "episodes": set(), "reasons": {}}
    chan = {"ch0_cells": 0, "ch0_nz": 0, "ch1_cells": 0, "ch1_nz": 0,
            "heat_cells": 0, "heat_nz": 0}
    co = {"pose_min": np.inf, "pose_max": -np.inf, "peak_min": np.inf, "peak_max": -np.inf,
          "pose_cells": 0, "pose_out": 0, "peak_cells": 0, "peak_out": 0,
          "u_cells": 0, "u_eq_heatmax": 0, "argmax_pairs": 0}
    reg = Reg()
    rng_audit = np.random.default_rng(0)
    lab = {"rows": 0, "valid_k_ne_K": 0, "valid_m_ne_M": 0, "teacher_ne_gt": 0,
           "teacher_ne_core": 0, "teacher_on_invalid_goal": 0,
           "contract_ne_minus1_at_start": 0, "start_rows": 0,
           "valid_m_ne_M_at_start": 0, "teacher_eq_contract_at_start": 0,
           "start_valid_agent_cells": 0, "start_teacher_minus1_on_valid_agent": 0}
    static = {"eps": 0, "rows": 0, "changed_rows": 0}

    for sid in ids:
        meta = metas[int(sid)]
        a = load_file(str(SHARD_DIR / f"shard_{int(sid):05d}.safetensors"))
        geo = a["geo"].astype(np.float32)
        heat = a["heat"].astype(np.float32)
        pose, peak = a["pose"].astype(np.float32), a["peak"].astype(np.float32)
        travel = a["travel"].astype(np.float32)
        vk, vm = a["valid_k"].astype(bool), a["valid_m"].astype(bool)
        tch, core, gt = (a["teacher"].astype(np.int64), a["teacher_core"].astype(np.int64),
                         a["teacher_gt"].astype(np.int64))
        contract = a["contract"].astype(np.int64)
        reason = np.array([m["reason"] for m in meta])

        cov["rows"] += len(meta)
        cov["episodes"].update(m["episode_id"] for m in meta)
        for r in set(reason):
            cov["reasons"][r] = cov["reasons"].get(r, 0) + int((reason == r).sum())

        ch0, ch1 = geo[:, 0], geo[:, 1]
        chan["ch0_cells"] += ch0.size
        chan["ch0_nz"] += int((ch0 != 0).sum())
        chan["ch1_cells"] += ch1.size
        chan["ch1_nz"] += int((ch1 != 0).sum())
        chan["heat_cells"] += heat.size
        chan["heat_nz"] += int((heat != 0).sum())

        pv = pose[vk]
        co["pose_min"], co["pose_max"] = min(co["pose_min"], float(pv.min())), max(co["pose_max"], float(pv.max()))
        co["pose_cells"] += len(pv)
        co["pose_out"] += int(((pv < -1) | (pv > 1)).sum())
        pm = peak[..., :2][vm[:, :, None].repeat(2, axis=-1).astype(bool)]
        co["peak_min"], co["peak_max"] = min(co["peak_min"], float(pm.min())), max(co["peak_max"], float(pm.max()))
        co["peak_cells"] += len(pm)
        co["peak_out"] += int(((pm < -1) | (pm > 1)).sum())
        hmax = heat.max(axis=(2, 3))
        nz = vm & (hmax > 0)
        co["u_cells"] += int(nz.sum())
        co["u_eq_heatmax"] += int((np.abs(peak[..., 2] - hmax) < 1e-3)[nz].sum())

        gi = np.arange(M_MAX)[None, None, :]
        eu = np.linalg.norm(peak[:, None, :, :2] - pose[:, :, None, :], axis=-1)  # (B,K,M)
        pair_all = vk[:, :, None] & vm[:, None, :]
        own_all = pair_all & (tch[:, :, None] == gi)
        for r in REASONS:
            sel = reason == r
            if not sel.any():
                continue
            acc = per_reason[r]
            acc.n += int(sel.sum())
            acc.ch0.append(float((ch0[sel] != 0).mean()))
            acc.ch1.append(float((ch1[sel] != 0).mean()))
            acc.heat.append(float((heat[sel] != 0).mean()))
            pair, own, tv_mat = pair_all[sel], own_all[sel], travel[sel]
            tv = tv_mat[pair]
            acc.tv_zero.append(float((tv == 0).mean()))
            acc.tv_unreach.append(float((tv >= UNREACHABLE).mean()))
            acc.zero_own.append(float((tv_mat[own] == 0).mean()))
            acc.zero_other.append(float((tv[~own[pair]] == 0).mean()))
            pu = peak[sel][vm[sel]]
            acc.peak_u.append(float(np.median(pu[:, 2])) if len(pu) else np.nan)
            ok = pair & (tv_mat < UNREACHABLE) & (eu[sel] > 1e-6)
            if int(ok.sum()) > 50:
                s = spearmanr(tv_mat[ok], eu[sel][ok]).statistic
                if np.isfinite(s):
                    acc.eu_rho.append(float(s))
            if routing and r == "start":
                gv, gge = [], []
                for i in np.flatnonzero(sel):
                    sg = routing.get(meta[i]["episode_id"], {}).get("sg")
                    if sg is None:
                        continue
                    K, M = meta[i]["K"], meta[i]["M"]
                    for a_ in range(K):
                        for j_ in range(M):
                            if not vm[i, j_]:
                                continue
                            g = float(sg[a_, j_])
                            t = float(travel[i, a_, j_])
                            if np.isfinite(g) and t < UNREACHABLE:
                                gv.append(t)
                                gge.append(g)
                if len(gv) > 50:
                    s = spearmanr(np.array(gv), np.array(gge)).statistic
                    if np.isfinite(s):
                        acc.geo_rho.append(float(s))

        # --- 坐标配准：pose 应落在已探索区里，peak 应落在该目标热图斑块里 ---
        pick = np.flatnonzero(reason != "?")
        if len(pick) > 120:
            pick = rng_audit.choice(pick, 120, replace=False)
        for i in pick:
            cells1 = np.argwhere(ch1[i] > 0).astype(np.float64)
            if len(cells1) >= 10 and vk[i].any():
                pc = cell(pose[i][vk[i]])  # (Q,2) = (col, row)
                d = nearest_dists(cells1, pc)
                rnd = rng_audit.integers(0, MAP_SIZE, (30, 2)).astype(np.float64)
                dn = nearest_dists(cells1, rnd)
                if d is not None and dn is not None:
                    reg.pose_hit.extend(d)
                    reg.pose_null.extend(dn)
            for j in np.flatnonzero(vm[i]):
                hc = np.argwhere(heat[i, j] > 0).astype(np.float64)
                if len(hc) < 5:
                    continue
                d = nearest_dists(hc, cell(peak[i, j, :2])[None, :])
                rnd = rng_audit.integers(0, MAP_SIZE, (8, 2)).astype(np.float64)
                dn = nearest_dists(hc, rnd)
                if d is not None and dn is not None:
                    reg.peak_hit.extend(d)
                    reg.peak_null.extend(dn)
            # 逐行轴约定：每个 agent 自己的已探索斑块质心 vs 它的坐标
            if reason[i] == "start" and int(vk[i].sum()) >= 2 and len(cells1) >= 20:
                cen, pts = [], []
                for k in np.flatnonzero(vk[i]):
                    cx, cy = cell(pose[i, k])
                    dd = np.hypot(cells1[:, 1] - cx, cells1[:, 0] - cy)
                    sel = cells1[dd <= 10]
                    if len(sel) < 3:
                        break
                    cen.append(sel.mean(0))
                    pts.append(pose[i, k, :2])
                else:
                    cen = np.asarray(cen)
                    pts = np.asarray(pts)
                    if np.std(cen[:, 0]) >= 1 and np.std(cen[:, 1]) >= 1:
                        for key, (u, v) in (("x~col", (pts[:, 0], cen[:, 1])),
                                            ("x~row", (pts[:, 0], cen[:, 0])),
                                            ("y~row", (pts[:, 1], cen[:, 0])),
                                            ("y~col", (pts[:, 1], cen[:, 1]))):
                            if np.std(u) > 1e-9:
                                s = spearmanr(u, v).statistic
                                if np.isfinite(s):
                                    reg.axis[key].append(float(s))

        lab["rows"] += len(meta)
        lab["valid_k_ne_K"] += int((vk.sum(1) != np.array([m["K"] for m in meta])).sum())
        lab["valid_m_ne_M"] += int((vm.sum(1) != np.array([m["M"] for m in meta])).sum())
        lab["teacher_ne_gt"] += int((tch != gt).sum())
        lab["teacher_ne_core"] += int((tch != core).sum())
        safe = np.clip(tch, 0, M_MAX - 1)
        lab["teacher_on_invalid_goal"] += int(((tch >= 0) & ~np.take_along_axis(vm, safe, 1)).sum())
        st = reason == "start"
        lab["start_rows"] += int(st.sum())
        lab["valid_m_ne_M_at_start"] += int((vm[st].sum(1) != np.array(
            [m["M"] for m, b_ in zip(meta, st) if b_])).sum()) if st.any() else 0
        lab["contract_ne_minus1_at_start"] += int((contract[st] != -1).sum())
        lab["teacher_eq_contract_at_start"] += int((tch[st] == contract[st]).sum())
        svk = vk[st]
        lab["start_valid_agent_cells"] += int(svk.sum())
        lab["start_teacher_minus1_on_valid_agent"] += int(
            ((tch[st] == -1) & svk).sum())

        cur = {}
        for i in range(len(meta)):
            key = meta[i]["episode_id"]
            h = float(ch0[i, ::3, ::3].sum())
            if key in cur:
                static["rows"] += 1
                if abs(h - cur[key]) > 1e-6:
                    static["changed_rows"] += 1
            else:
                cur[key] = h
        static["eps"] += len(cur)

        del a, geo, heat
        print(f"[audit] shard {int(sid):05d} ok, {time.time() - t0:.0f}s", flush=True)

    # 3) 汇总
    reg_out = reg.out()
    rep = {
        "generated_at": time.time(),
        "shards": [int(i) for i in ids],
        "coverage": {"rows": cov["rows"], "episodes": len(cov["episodes"]),
                     "reasons": dict(sorted(cov["reasons"].items()))},
        "channels": {
            "geo_ch0_nonzero_frac": round(chan["ch0_nz"] / chan["ch0_cells"], 4),
            "geo_ch1_nonzero_frac": round(chan["ch1_nz"] / chan["ch1_cells"], 4),
            "heat_nonzero_frac": round(chan["heat_nz"] / chan["heat_cells"], 5),
            "ch0_constant_within_episode": (f"{static['rows'] - static['changed_rows']}/"
                                            f"{static['rows']} 行" if static["rows"] else None),
        },
        "coords": {
            "pose_range": [round(co["pose_min"], 3), round(co["pose_max"], 3)],
            "peak_range": [round(co["peak_min"], 3), round(co["peak_max"], 3)],
            "pose_outside_minus1_1": f"{co['pose_out']}/{co['pose_cells']}",
            "peak_outside_minus1_1": f"{co['peak_out']}/{co['peak_cells']}",
            "peak_u_equals_heat_max": f"{co['u_eq_heatmax']}/{co['u_cells']}",
            "registration": reg_out,
        },
        "labels": lab,
        "per_reason": {r: per_reason[r].out() for r in REASONS},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)

    print(json.dumps({k: rep[k] for k in ("coverage", "channels", "labels")},
                     ensure_ascii=False, indent=1))
    print("\n-- 坐标与热图 --")
    print(json.dumps(rep["coords"], ensure_ascii=False, indent=1))
    print("\n-- 逐 reason --")
    print(f"{'reason':9s} {'rows':>7s} {'ch0%':>6s} {'ch1%':>6s} {'heat%':>6s} {'tv=0%':>6s} "
          f"{'tv=1e4%':>8s} {'=0|分配':>8s} {'=0|未分配':>9s} {'u中位':>6s} {'~欧氏':>7s} {'~测地':>7s}")
    for r, d in rep["per_reason"].items():
        def g(k, scale=100.0, na=float("nan")):
            v = d[k]
            return na if v is None else scale * v
        print(f"{r:9s} {d['rows']:7d} {g('geo_ch0_nonzero'):6.2f} {g('geo_ch1_nonzero'):6.2f} "
              f"{g('heat_nonzero'):6.2f} {g('travel_eq_zero'):6.2f} {g('travel_unreachable'):8.2f} "
              f"{g('travel_eq_zero_if_teacher_here'):8.2f} {g('travel_eq_zero_if_not'):9.2f} "
              f"{d['peak_u_median']:6.3f} {g('spearman_travel_vs_euclid', 1):7.3f} "
              f"{g('spearman_travel_vs_geodesic', 1):7.3f}")
    print(f"\n[audit] {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
