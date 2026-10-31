"""K 个 agent × M 个目标的分配策略：BEV 占用/探索图 + 每个目标的 CLIP 热图 + BFS 代价 → 分配 logits。"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data import K_MAX, M_MAX, UNREACHABLE

MAP_METERS = 12.0  # 地图单位→米：mapgeo 拟合出的 scale 恒为 1/12


def mlp(in_dim: int, out_dim: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def couple(logits: torch.Tensor, valid_k: torch.Tensor, valid_m: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Sinkhorn 式配对（log 域）：在 agent 维和目标维交替归一化打分矩阵。

    逐对打分没有全局视野，多个 agent 会挤在同一个最优点目标上（一帧只覆盖 42% 目标）。
    这里不加任何参数，只把分数推向"每个 agent 分到一个、目标尽量不重复"。
    """
    mask = valid_k[:, :, None] & valid_m[:, None, :]
    z = torch.where(mask, logits, torch.full_like(logits, -1e9))
    for _ in range(iters):
        row = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = torch.where(valid_k[:, :, None], row, z)  # 补齐的 agent 行不参与归一化
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    return torch.where(mask, z, torch.full_like(z, float("-inf")))


class AssignNet(nn.Module):
    # 逐 (agent, goal) 对的工程化特征：log 代价、欧氏、agent 处热响应、峰值置信、两个有效性位；
    # use_contract 时再加三项承诺相关特征（自己承诺于此目标 / 被别人承诺 / 本 agent 有无承诺）；
    # use_mgeo 时用 mapgeo 投影出的目标地图坐标算真实平面距离（与测地 rho≈0.78），
    # 替掉几乎无信息的 peak 距离与 BFS travel。
    def __init__(self, ch: int = 32, emb: int = 96, k_max: int = K_MAX, m_max: int = M_MAX,
                 couple_iters: int = 0, use_contract: bool = False, use_mgeo: bool = False):
        super().__init__()
        self.k_max, self.m_max = k_max, m_max
        self.couple_iters = couple_iters
        self.use_contract = use_contract
        self.use_mgeo = use_mgeo
        self.pair_f = (9 if use_contract else 6) + (2 if use_mgeo else 0)
        self.trunk = nn.Sequential(
            nn.Conv2d(2 + m_max, ch, 5, stride=2, padding=2),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, ch * 2),
            nn.ReLU(inplace=True),
        )
        feat = ch * 2
        self.agent_in = feat + 2 + 1
        self.agent_mlp = mlp(self.agent_in, emb)
        self.goal_mlp = mlp(feat + 1, emb)
        self.head = mlp(2 * emb + self.pair_f, emb)
        self.score = nn.Linear(emb, 1)

    def _sample(self, x: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """x:(B,C,H,W), pts:(B,Q,2) 归一化坐标 → (B,Q,C)"""
        b, q, _ = pts.shape
        grid = pts.view(b, 1, q, 2)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return out.view(b, x.shape[1], q).transpose(1, 2)

    def forward(self, batch: dict) -> torch.Tensor:
        geo, heat = batch["geo"], batch["heat"]
        pose, peak, travel = batch["pose"], batch["peak"], batch["travel"]
        valid_k, valid_m = batch["valid_k"], batch["valid_m"]
        k, m = self.k_max, self.m_max

        x = torch.cat([geo, heat], dim=1)
        f = self.trunk(x)  # (B,2C,12,12)

        # 数据里的归一化坐标区间是 [-2, 0]（SCHEMA 写的 [-1,1] 与实测不符），
        # grid_sample 需要 [-1,1]，故 +1 做仿射变换。
        pose_g = pose + 1.0

        agent_f = self._sample(f, pose_g)  # (B,K,2C)
        agent_geo = self._sample(geo, pose_g)  # (B,K,2)
        explored = geo[:, 1].flatten(1).mean(dim=1).view(-1, 1, 1).expand(-1, k, 1)
        agent = self.agent_mlp(torch.cat([agent_f, agent_geo, explored], dim=-1))

        # heat 峰值位置几乎不含目标几何（与测地 rho≈-0.07）；有真实地图坐标时
        # 直接在目标处采样占用/探索上下文，缺 transform 的 episode 退回峰值。
        gpos = peak[..., :2]
        if self.use_mgeo:
            gpos = torch.where(
                (batch["mvalid"] > 0.5)[:, :, None], batch["mpos"], gpos
            )
        goal_f = self._sample(f, gpos + 1.0)  # (B,M,2C)
        goal = self.goal_mlp(torch.cat([goal_f, peak[..., 2:]], dim=-1))

        # heat_at_agent[b,k,j] = 第 j 个目标的热图在 agent k 位置处的响应
        grid = pose_g.unsqueeze(1)  # (B,1,K,2)
        hat = F.grid_sample(
            heat, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        heat_at_agent = hat.squeeze(2).transpose(1, 2)  # (B,K,M)

        cost = torch.clamp(travel, max=UNREACHABLE)
        log_cost = torch.log1p(cost) / torch.log1p(
            torch.tensor(float(UNREACHABLE), device=x.device)
        )
        dist = torch.linalg.norm(peak[:, None, :, :2] - pose[:, :, None, :], dim=-1) / 2.0
        score_peak = peak[:, None, :, 2].expand(-1, k, -1)
        kv = valid_k[:, :, None].float().expand(-1, -1, m)
        mv = valid_m[:, None, :].float().expand(-1, k, -1)

        feats = [log_cost, dist, heat_at_agent, score_peak, kv, mv]
        if self.use_mgeo:
            # 目标地图坐标（mapgeo 拟合出的 world→map 投影）→ 真实平面距离（米）。
            # 缺 transform 的 episode 用 1.0（最大值）并由 mvalid 位标记，模型学会忽略。
            mpos, mvalid = batch["mpos"], batch["mvalid"]
            mok = mvalid[:, None, :].expand(-1, k, -1)
            mdist = torch.linalg.norm(pose[:, :, None, :] - mpos[:, None, :, :], dim=-1)
            log_m = torch.log1p(torch.clamp(mdist * MAP_METERS, max=60.0)) / math.log1p(60.0)
            log_m = torch.where(mok > 0.5, log_m, torch.ones_like(log_m))
            feats += [log_m, mok]
        if self.use_contract:
            # 释放样本里的 contract（这一帧之前每个 agent 承诺在哪个目标）谁都没喂给网络，
            # 但审计显示 teacher 标签有 67% 就是它的复读 —— 见 m3assign/churn.py。
            contract = batch["contract"]
            gi = torch.arange(m, device=x.device).view(1, 1, m)
            own = ((contract[:, :, None] == gi) & (contract[:, :, None] >= 0)).float()
            taken = own.sum(1, keepdim=True).expand(-1, k, -1) - own
            has = (contract >= 0).float()[:, :, None].expand(-1, -1, m)
            feats += [own, taken, has]

        a = agent[:, :, None, :].expand(-1, -1, m, -1)
        g = goal[:, None, :, :].expand(-1, k, -1, -1)
        pair = torch.stack(feats, dim=-1)
        logits = self.score(self.head(torch.cat([a, g, pair], dim=-1))).squeeze(-1)
        if self.couple_iters:
            logits = couple(logits, valid_k, valid_m, self.couple_iters)
        return logits.masked_fill(mv < 0.5, float("-inf"))


def build(meta: dict | None) -> AssignNet:
    """按 checkpoint 记录的超参重建模型；老权重没有 couple_iters/contract/mgeo，默认关闭。"""
    args = (meta or {}).get("args") or {}
    return AssignNet(couple_iters=int(args.get("couple_iters", 0)),
                     use_contract=bool(args.get("contract", 0)),
                     use_mgeo=bool(args.get("mgeo", 0)))


def labels_of(batch: dict, key: str = "teacher") -> torch.Tensor:
    """teacher（每个在场 agent 的目标）为标签，无效 agent/目标置 -1 由 CE 忽略。"""
    y = batch[key].clone()
    y[~batch["valid_k"]] = -1
    y[y < 0] = -1
    y[y >= batch["valid_m"].shape[-1]] = -1
    safe = batch["valid_m"].gather(1, y.clamp(min=0))
    y[~safe] = -1
    return y


def loss_fn(logits: torch.Tensor, batch: dict) -> torch.Tensor:
    y = labels_of(batch)
    b, k, m = logits.shape
    ce = F.cross_entropy(
        logits.reshape(b * k, m), y.reshape(b * k), ignore_index=-1, reduction="none"
    )
    return ce[y.reshape(b * k) >= 0].mean()


@torch.no_grad()
def metrics(logits: torch.Tensor, batch: dict) -> dict:
    from scipy.optimize import linear_sum_assignment

    y = labels_of(batch)
    pred = logits.argmax(dim=-1)
    eval_mask = batch["valid_k"] & (y >= 0)
    correct = (pred == y) & eval_mask
    acc = correct.sum().item() / max(1, eval_mask.sum().item())

    logits_np = logits.detach().float().cpu().numpy()
    teacher_np = y.detach().cpu().numpy()
    core_np = batch["teacher_core"].detach().cpu().numpy()
    travel_np = batch["travel"].detach().float().cpu().numpy()
    valid_k_np = batch["valid_k"].detach().cpu().numpy()
    valid_m_np = batch["valid_m"].detach().cpu().numpy()
    cf_hit = cf_tot = 0
    lab_hit = lab_tot = 0
    ratio_hit = []
    n = logits_np.shape[0]
    for i in range(n):
        vk = valid_k_np[i]
        if not vk.any():
            continue
        kk = np.where(vk)[0]
        mm = np.where(valid_m_np[i])[0]
        if mm.size < kk.size:
            continue
        # 把模型打分做成一对一分配（匈牙利），避免同一目标被两个 agent 抢
        cost = -logits_np[i][kk][:, mm]
        cost = np.where(np.isfinite(cost), cost, 1e6)
        r, c = linear_sum_assignment(cost)
        ours = np.full(vk.size, -1)
        for a, b_ in zip(r, c):
            ours[kk[a]] = mm[b_]
        tl = teacher_np[i][kk]
        tk = tl >= 0
        if tk.any():
            lab_hit += int((ours[kk][tk] == tl[tk]).sum())
            lab_tot += int(tk.sum())
        cc = core_np[i][kk]
        has = cc >= 0
        if has.any():
            cf_hit += int((ours[kk][has] == cc[has]).sum())
            cf_tot += int(has.sum())
        lab = teacher_np[i][kk]
        keep = lab >= 0
        if keep.any():
            t_cost = travel_np[i][kk[keep], lab[keep]]
            o_cost = travel_np[i][kk[keep], ours[kk][keep]]
            if t_cost.max() > 0 and np.isfinite(t_cost.max()):
                # teacher 已贴近目标时比值会被小数分母炸开，截断到 5 倍
                ratio_hit.append(min(float(o_cost.max() / t_cost.max()), 5.0))

    return {
        "acc": acc,
        "cf_acc": cf_hit / max(1, cf_tot),
        "makespan_ratio": float(np.median(ratio_hit)) if ratio_hit else float("nan"),
        # 一对一化以后跟全部教师标签的一致率。cf_acc 只统计无冲突子集，天花板 33%，
        # 单看它会误判模型好坏；teacher_acc 才是真实的教师复现率。
        "teacher_acc": lab_hit / max(1, lab_tot),
        "n_eval": int(eval_mask.sum().item()),
    }
