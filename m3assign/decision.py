"""决策聚焦损失：把分配从"逐对分类"改成直接优化团队 makespan。

BC 的上限是教师本人（教师在真测地下 regret 1.19，还不如几何匈牙利 1.12），
所以再拟合 teacher 也不会更好。这里对 logits 做 Sinkhorn 软分配 P，用真测地
代价算每人的期望行程（第一段 + 目标间期望距离），再对 agent 取 softmax，
得到可微的 makespan 代理，按每个 episode 的 oracle makespan 归一化成软 regret。
"""
from __future__ import annotations

import torch


def sinkhorn(
    logits: torch.Tensor, valid_k: torch.Tensor, valid_m: torch.Tensor,
    iters: int = 6, tau: float = 1.0,
) -> torch.Tensor:
    """logits (B,K,M) -> 双随机软分配 P (B,K,M)，无效格恒为 0。"""
    mask = valid_k[:, :, None] & valid_m[:, None, :]
    z = torch.where(mask, logits / tau, torch.full_like(logits, -1e9))
    for _ in range(iters):
        z = z - torch.logsumexp(z, dim=2, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    p = torch.where(mask, torch.exp(z), torch.zeros_like(z))
    # 目标多于 agent 时列和与行和冲突，按行归一化，保证"每人的期望行程"是加权平均
    return p / p.sum(2, keepdim=True).clamp_min(1e-9)


def soft_makespan(
    logits: torch.Tensor, batch: dict, temp: float = 0.5, iters: int = 10, tau: float = 0.2
) -> torch.Tensor:
    """(B,) 软 makespan（米）。cost 是真测地，不是 shard 里的 BFS travel。"""
    p = sinkhorn(logits, batch["valid_k"], batch["valid_m"], iters, tau)
    own = (p * batch["gcost"]).sum(-1)  # (B,K) 起点到分配目标的期望距离
    spread = torch.einsum("bkj,bkl,bjl->bk", p, p, batch["gg"])  # 目标两两距离，算了两次
    tour = own + 0.5 * spread
    tour = tour.masked_fill(~batch["valid_k"], float("-inf"))
    return temp * torch.logsumexp(tour / temp, dim=1)


def df_loss(
    logits: torch.Tensor, batch: dict, temp: float = 0.5, iters: int = 10, tau: float = 0.2
) -> torch.Tensor:
    return (soft_makespan(logits, batch, temp, iters, tau) / batch["oms"].clamp_min(1e-3)).mean()
