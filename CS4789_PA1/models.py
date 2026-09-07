"""Shared scoring model utilities."""

from typing import Optional

import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim_emb: int = 10,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim_emb = dim_emb

        self.user_encoder = nn.Embedding(n_users, dim_emb)
        self.item_encoder = nn.Embedding(n_items, dim_emb)

    def forward(
        self,
        user_ids: torch.Tensor,  # (batch_size,)
        item_ids: Optional[torch.Tensor] = None,  # (batch_size,) or (batch_size, ranking_length)
        requires_grad: bool = True,
        **kwargs,
    ):
        is_single_item = item_ids is not None and item_ids.dim() == 1

        if item_ids is None:
            item_ids = torch.arange(self.n_items, device=user_ids.device).unsqueeze(0)

        if is_single_item:
            item_ids = item_ids.unsqueeze(1)

        if requires_grad:
            learned_user_emb = self.user_encoder(user_ids)
            learned_item_emb = self.item_encoder(item_ids)
        else:
            with torch.no_grad():
                learned_user_emb = self.user_encoder(user_ids)
                learned_item_emb = self.item_encoder(item_ids)

        learned_user_emb = learned_user_emb.unsqueeze(1)
        logits = (learned_user_emb * learned_item_emb).sum(dim=-1)

        if is_single_item:
            logits = logits.squeeze(1)

        return logits


def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    """Sample Gumbel(0, 1) noise with the same shape as `x`."""
    return torch.distributions.Gumbel(torch.zeros_like(x), torch.ones_like(x)).sample()
