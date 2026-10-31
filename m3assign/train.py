"""在 HM3D-M3 teacher_100k 上训练多智能体目标分配策略，指标实时落盘给看板读取。"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import M_MAX, SHARD_DIR, Store, geodesic_costs, shard_ids
from .decision import df_loss
from .mapgeo import goal_map_positions
from .model import AssignNet, loss_fn, metrics

RUNS = Path(__file__).resolve().parent.parent / "runs"


COLS = (
    "geo",
    "heat",
    "pose",
    "peak",
    "travel",
    "teacher",
    "teacher_core",
    "contract",
    "valid_k",
    "valid_m",
)


def shard_rows(sid: int) -> int:
    with open(SHARD_DIR / f"shard_{sid:05d}.meta.jsonl", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def cpu_tensors(store: Store) -> dict:
    """样本留在主存：100k 行的 BEV+热图 fp16 约 13GB，显存装不下，按 batch 搬运。"""
    out = {}
    for c in COLS:
        x = getattr(store, c)
        if x.dtype == np.int16:
            x = x.astype(np.int64)
        out[c] = torch.from_numpy(x)
    return out


def geo_tensors(store: Store) -> dict:
    """每行的真测地代价 (4,5) 与目标间距离 (5,5)：决策聚焦损失的真值，来自 episode routing。"""
    tab = geodesic_costs()
    pos = {e: i for i, e in enumerate(tab["ids"].tolist())}
    idx = np.asarray([pos[m["episode_id"]] for m in store.meta], dtype=np.int64)
    sg = np.where(np.isfinite(tab["sg"][idx]), tab["sg"][idx], np.float32(1e4))
    gg = np.where(np.isfinite(tab["gg"][idx]), tab["gg"][idx], np.float32(1e4))
    return {
        "gcost": torch.from_numpy(sg.astype(np.float32)),
        "gg": torch.from_numpy(gg.astype(np.float32)),
        "oms": torch.from_numpy(tab["oracle_ms"][idx].astype(np.float32)),
    }


def mgeo_tensors(store: Store) -> dict:
    """每行各目标的地图坐标 (5,2) + 有效位，来自 mapgeo 的 world→map 投影。"""
    gm = goal_map_positions()
    n = store.n
    mpos = np.zeros((n, M_MAX, 2), np.float32)
    mvalid = np.zeros((n, M_MAX), np.float32)
    for eid, rows in store.episode_rows.items():
        p = gm.get(eid)
        if p is None:
            continue
        mask = np.isfinite(p[:, 0])
        if not mask.any():
            continue
        r = np.asarray(sorted(rows), dtype=np.int64)
        mpos[r] = np.where(mask[:, None], np.nan_to_num(p), 0.0)[None, :, :]
        mvalid[r] = mask.astype(np.float32)[None, :]
    return {"mpos": torch.from_numpy(mpos), "mvalid": torch.from_numpy(mvalid)}


def take(batch_tensors: dict, idx, device: torch.device, augment: bool = False) -> dict:
    out = {}
    for k, v in batch_tensors.items():
        x = v[idx]
        if x.dtype == torch.float16:
            x = x.float()
        out[k] = x.to(device, non_blocking=True)
    if augment:
        _flip(out, device)
    return out


def _flip(out: dict, device: torch.device) -> None:
    """BEV 水平/垂直翻转：地图、热图与坐标同步，分配标签与 travel 代价保持不变。"""
    b = out["geo"].shape[0]
    h = torch.randint(0, 2, (b,), device=device).bool()
    v = torch.randint(0, 2, (b,), device=device).bool()
    hi, vi = h.nonzero().flatten(), v.nonzero().flatten()
    if hi.numel():
        out["geo"][hi] = out["geo"][hi].flip(-1)
        out["heat"][hi] = out["heat"][hi].flip(-1)
        out["pose"][hi, :, 0] = -out["pose"][hi, :, 0] - 2.0
        out["peak"][hi, :, 0] = -out["peak"][hi, :, 0] - 2.0
        if "mpos" in out:
            out["mpos"][hi, :, 0] = -out["mpos"][hi, :, 0] - 2.0
    if vi.numel():
        out["geo"][vi] = out["geo"][vi].flip(-2)
        out["heat"][vi] = out["heat"][vi].flip(-2)
        out["pose"][vi, :, 1] = -out["pose"][vi, :, 1] - 2.0
        out["peak"][vi, :, 1] = -out["peak"][vi, :, 1] - 2.0
        if "mpos" in out:
            out["mpos"][vi, :, 1] = -out["mpos"][vi, :, 1] - 2.0


def evaluate(model, tensors, device, n_max: int, batch: int, df_kw: dict | None = None) -> dict:
    n = min(n_max, tensors["geo"].shape[0])
    tot = {"loss": 0.0, "n": 0}
    acc = cf = ratio = tea = 0.0
    df_sum = 0.0
    cnt = 0
    with torch.no_grad():
        for s in range(0, n, batch):
            sel = np.arange(s, min(s + batch, n))
            tb = take(tensors, sel, device)
            logits = model(tb)
            l = loss_fn(logits, tb)
            m = metrics(logits, tb)
            if df_kw:
                df_sum += float(df_loss(logits, tb, **df_kw)) * sel.size
            tot["loss"] += float(l) * sel.size
            tot["n"] += sel.size
            acc += m["acc"] * sel.size
            cf += m["cf_acc"] * sel.size
            tea += m["teacher_acc"] * sel.size
            if not math.isnan(m["makespan_ratio"]):
                ratio += m["makespan_ratio"] * sel.size
            cnt += sel.size
    out = {
        "loss": tot["loss"] / max(1, tot["n"]),
        "acc": acc / max(1, cnt),
        "cf_acc": cf / max(1, cnt),
        "teacher_acc": tea / max(1, cnt),
        "makespan_ratio": ratio / max(1, cnt),
    }
    if df_kw:
        out["df"] = df_sum / max(1, cnt)
    return out


def corrupt_contract(tb: dict, drop_p: float, rand_p: float) -> None:
    """训练时打乱 contract，逼模型别把"上一帧承诺"当标签复读。

    训练数据的 contract 永远是教师自己的承诺，推理时喂的却是我们自己的承诺；
    不改的话模型会把 own=1 当成强先验，错误承诺会自我强化（见 commit_eval）。
    """
    if drop_p <= 0 and rand_p <= 0:
        return
    c = tb["contract"]
    b, k = c.shape
    u = torch.rand(b, device=c.device)
    drop = u < drop_p
    rand = (u >= drop_p) & (u < drop_p + rand_p)
    if drop.any():
        c[drop] = -1
    if rand.any():
        sub = c[rand]
        vm = tb["valid_m"][rand]
        vk = tb["valid_k"][rand]
        g = torch.randint(0, vm.shape[1], sub.shape, device=c.device)
        ok = vm.gather(1, g) & vk
        # 一半概率直接没有承诺，模拟"这帧才第一次决定"
        keep = ok & (torch.rand(sub.shape, device=c.device) < 0.5)
        c[rand] = torch.where(keep, g, torch.full_like(g, -1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shards", type=int, default=16)
    ap.add_argument("--val-shards", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-rows", type=int, default=2048)
    ap.add_argument("--augment", type=int, default=1, help="1=开启 BEV 翻转增强")
    ap.add_argument("--shard-group", type=int, default=0,
                    help=">0 时按 N 个 shard 一组流式训练；全量数据不必同时驻留内存")
    ap.add_argument("--couple-iters", type=int, default=0, help="Sinkhorn 配对迭代次数，0=纯逐对打分")
    ap.add_argument("--contract", type=int, default=0,
                    help="1=把每帧的 contract（先前的承诺）喂成逐对特征，见 m3assign/churn.py")
    ap.add_argument("--contract-drop", type=float, default=0.0,
                    help=">0 时按概率清空 contract，训练模型不依赖承诺也能决定")
    ap.add_argument("--contract-rand", type=float, default=0.0,
                    help=">0 时按概率把 contract 换成随机目标，模拟自己的承诺是错的")
    ap.add_argument("--mgeo", type=int, default=0,
                    help="1=用 mapgeo 投影的目标地图坐标作为真实平面距离特征")
    ap.add_argument("--df-weight", type=float, default=0.0,
                    help=">0 时在 CE 之外加决策聚焦项：soft makespan / oracle makespan（真测地）")
    ap.add_argument("--ce-weight", type=float, default=1.0,
                    help="教师标签 CE 的权重；0=纯决策聚焦（不拟合 teacher）")
    ap.add_argument("--df-temp", type=float, default=0.5, help="softmax over agents 的温度（米）")
    ap.add_argument("--df-tau", type=float, default=0.2,
                    help="软分配温度：越小越接近硬分配，代理与真 makespan 的区分度越大")
    ap.add_argument("--df-iters", type=int, default=10, help="软分配的 Sinkhorn 迭代次数")
    ap.add_argument("--start-only", type=int, default=0,
                    help="1=只训练 reason=start 的帧（初始承诺），见 m3assign/startdata.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-latest", action="store_true", help="不要把 runs/latest 指到本次运行（冒烟测试用）")
    ap.add_argument("--name", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    all_ids = shard_ids()
    tr_ids = all_ids[: args.train_shards]
    va_ids = all_ids[len(all_ids) - args.val_shards :] if args.val_shards else []
    print(f"[data] train shards={len(tr_ids)} val shards={len(va_ids)}", flush=True)

    groups: list[list[int]] = []
    tr_t = None
    if args.start_only:
        from .startdata import StartStore

        if args.shard_group:
            raise SystemExit("--start-only 与 --shard-group 不能同时用")
        t0 = time.time()
        tr_store = StartStore(tr_ids)
        tr_t = cpu_tensors(tr_store)
        if args.df_weight > 0:
            tr_t.update(geo_tensors(tr_store))
        if args.mgeo:
            tr_t.update(mgeo_tensors(tr_store))
        n = tr_t["geo"].shape[0]
        steps_per_epoch = math.ceil(n / args.batch)
        print(f"[data] start-only rows={n} in {time.time()-t0:.1f}s", flush=True)
    elif args.shard_group:
        groups = [tr_ids[i : i + args.shard_group] for i in range(0, len(tr_ids), args.shard_group)]
        rows_per_shard = {sid: shard_rows(sid) for sid in tr_ids}
        n = sum(rows_per_shard.values())
        steps_per_epoch = sum(
            math.ceil(sum(rows_per_shard[s] for s in g) / args.batch) for g in groups
        )
        print(
            f"[data] streaming {n} rows in {len(groups)} groups of <= {args.shard_group} shards",
            flush=True,
        )
    else:
        t0 = time.time()
        tr_store = Store(tr_ids)
        tr_t = cpu_tensors(tr_store)
        if args.df_weight > 0:
            tr_t.update(geo_tensors(tr_store))
        if args.mgeo:
            tr_t.update(mgeo_tensors(tr_store))
        n = tr_t["geo"].shape[0]
        steps_per_epoch = math.ceil(n / args.batch)
        print(f"[data] loaded rows={n} in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    if args.start_only:
        va_store = StartStore(va_ids) if va_ids else None
    else:
        va_store = Store(va_ids) if va_ids else None
    va_t = cpu_tensors(va_store) if va_store else None
    if va_store is not None:
        if args.df_weight > 0:
            va_t.update(geo_tensors(va_store))
        if args.mgeo:
            va_t.update(mgeo_tensors(va_store))
        print(f"[data] loaded val rows={va_store.n} in {time.time()-t0:.1f}s", flush=True)

    model = AssignNet(
        couple_iters=args.couple_iters,
        use_contract=bool(args.contract),
        use_mgeo=bool(args.mgeo),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / (f"{stamp}-{args.name}" if args.name else stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = RUNS / "latest"
    if not args.no_latest:
        if latest.is_symlink():
            latest.unlink()
        latest.symlink_to(run_dir.name, target_is_directory=True)
    hist = (run_dir / "history.jsonl").open("a", encoding="utf-8")
    started = time.time()

    state = {
        "status": "running",
        "run_dir": str(run_dir),
        "started_at": started,
        "updated_at": time.time(),
        "device": str(device),
        "epochs_total": args.epochs,
        "epoch": 0,
        "step": 0,
        "total_steps": total_steps,
        "n_train": n,
        "n_val": va_t["geo"].shape[0] if va_t else 0,
        "train_shards": tr_ids,
        "val_shards": va_ids,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "best": None,
        "last_train": None,
        "last_val": None,
        "args": vars(args),
    }

    def write_state() -> None:
        state["updated_at"] = time.time()
        (run_dir / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def log_entry(d: dict) -> None:
        d["step"] = state["step"]
        d["epoch"] = state["epoch"]
        d["ts"] = time.time()
        hist.write(json.dumps(d, ensure_ascii=False) + "\n")
        hist.flush()

    def batches():
        if not groups:
            perm = np.random.permutation(n)
            for s in range(0, n, args.batch):
                yield tr_t, perm[s : s + args.batch]
            return
        order = list(groups)
        random.shuffle(order)
        for g in order:
            store = Store(g)
            tt = cpu_tensors(store)
            if args.df_weight > 0:
                tt.update(geo_tensors(store))
            if args.mgeo:
                tt.update(mgeo_tensors(store))
            perm = np.random.permutation(store.n)
            for s in range(0, store.n, args.batch):
                yield tt, perm[s : s + args.batch]
            del store, tt

    write_state()
    gstep = 0
    running = []
    df_run = []
    window_start = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            state["epoch"] = epoch
            for src, sel in batches():
                tb = take(src, sel, device, augment=bool(args.augment))
                if args.contract:
                    corrupt_contract(tb, args.contract_drop, args.contract_rand)
                logits = model(tb)
                loss = args.ce_weight * loss_fn(logits, tb)
                dfv = None
                if args.df_weight > 0:
                    dfv = df_loss(logits, tb, temp=args.df_temp, iters=args.df_iters,
                                  tau=args.df_tau)
                    loss = loss + args.df_weight * dfv
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                gstep += 1
                state["step"] = gstep
                # 看板"跟随训练"要按 step 看帧；只在 eval 时写 state 会让画面每 200 步才动一次
                if gstep % 5 == 0:
                    write_state()
                running.append(float(loss))
                running = running[-50:]
                if dfv is not None:
                    df_run.append(float(dfv))
                    df_run = df_run[-50:]

                if gstep % args.eval_every == 0 or gstep == total_steps:
                    sps = args.batch * args.eval_every / max(1e-6, time.time() - window_start)
                    df_kw = (
                        {"temp": args.df_temp, "iters": args.df_iters, "tau": args.df_tau}
                        if args.df_weight > 0 else None
                    )
                    tr_eval = evaluate(model, src, device, args.eval_rows, args.batch, df_kw)
                    tr_eval["loss_window"] = float(np.mean(running))
                    if df_run:
                        tr_eval["df_window"] = float(np.mean(df_run))
                    tr_eval["samples_per_sec"] = sps
                    log_entry({"split": "train", **tr_eval})
                    if va_t is not None:
                        va_eval = evaluate(model, va_t, device, args.eval_rows, args.batch, df_kw)
                        log_entry({"split": "val", **va_eval})
                        state["last_val"] = va_eval
                        best = state["best"] or {}
                        # 决策聚焦训练按 val 软 regret 选 best；纯 BC 仍按验证损失
                        # （cf_acc 每 4096 样本噪声太大且天花板只有 33%，会选中抖动点）。
                        key = "df" if args.df_weight > 0 else "loss"
                        if va_eval[key] < best.get(key, float("inf")):
                            state["best"] = {
                                "loss": va_eval["loss"],
                                "cf_acc": va_eval["cf_acc"],
                                "acc": va_eval["acc"],
                                "teacher_acc": va_eval["teacher_acc"],
                                "makespan_ratio": va_eval["makespan_ratio"],
                                "step": gstep,
                            }
                            if "df" in va_eval:
                                state["best"]["df"] = va_eval["df"]
                            torch.save(
                                {
                                    "state_dict": model.state_dict(),
                                    "meta": {
                                        "train_shards": tr_ids,
                                        "val_shards": va_ids,
                                        "step": gstep,
                                        "args": vars(args),
                                    },
                                },
                                run_dir / "best.pt",
                            )
                    state["last_train"] = tr_eval
                    window_start = time.time()
                    write_state()
                    print(
                        f"[ep {epoch}/{args.epochs} step {gstep}/{total_steps}] "
                        f"train loss={tr_eval['loss']:.3f} cf_acc={tr_eval['cf_acc']:.3f} "
                        f"val={state['last_val'] or {}} {sps:.0f} samp/s",
                        flush=True,
                    )
            state["epoch"] = epoch
            write_state()
    except KeyboardInterrupt:
        state["status"] = "interrupted"
    finally:
        state["status"] = "done" if gstep >= total_steps else state.get("status", "running")
        state["elapsed_sec"] = round(time.time() - started, 1)
        write_state()
        torch.save({"state_dict": model.state_dict()}, run_dir / "last.pt")
        hist.close()
        print(f"[done] run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
