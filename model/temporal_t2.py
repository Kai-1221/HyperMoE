from __future__ import annotations

import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):

    def __init__(
        self,
        dim: int = 512,
        heads: int = 4,
        dropout: float = 0.1,
        n_times: int = 4,
        n_states: int = 3,
        in_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.in_dim = int(in_dim) if in_dim else dim
        self.proj_in = nn.Linear(self.in_dim, dim) if self.in_dim != dim else None
        self.proj_out = nn.Linear(dim, self.in_dim) if self.in_dim != dim else None
        self.time_emb = nn.Embedding(n_times, dim)
        self.state_emb = nn.Embedding(n_states, dim) if n_states > 1 else None
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        time_ids: torch.Tensor,
        state_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,T,D], got {tuple(tokens.shape)}")
        b, t, d = tokens.shape
        if d != self.in_dim:
            raise ValueError(f"expected dim {self.in_dim}, got {d}")
        if time_ids.shape != (b, t):
            raise ValueError(f"time_ids must be [{b},{t}], got {tuple(time_ids.shape)}")

        if self.proj_in is not None:
            tokens = self.proj_in(tokens)

        x = tokens + self.time_emb(time_ids)
        if self.state_emb is not None and state_ids is not None:
            x = x + self.state_emb(state_ids)

        if key_padding_mask is not None and bool(key_padding_mask.all()):
            return tokens.new_zeros((b, self.in_dim))

        q = self.query.expand(b, -1, -1)
        attn, _ = self.attn(
            q, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        base = x.mean(dim=1, keepdim=True)
        h = self.norm1(base + self.drop(attn))
        h = self.norm2(h + self.ffn(h))
        h = h.squeeze(1)
        return self.proj_out(h) if self.proj_out is not None else h


class MeanTemporalPool(nn.Module):

    def __init__(self, in_dim: int | None = None) -> None:
        super().__init__()
        self.in_dim = int(in_dim) if in_dim else 512

    def forward(
        self,
        tokens: torch.Tensor,
        time_ids: torch.Tensor | None = None,
        state_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,T,D], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.in_dim:
            raise ValueError(f"expected dim {self.in_dim}, got {tokens.shape[-1]}")
        return tokens.mean(dim=1)


class DualTemporalMean(nn.Module):

    def __init__(self, share: bool = False, in_dim: int | None = None) -> None:
        super().__init__()
        self.share = share
        self.fs = MeanTemporalPool(in_dim=in_dim)
        self.roi = self.fs if share else MeanTemporalPool(in_dim=in_dim)

    def forward(
        self,
        fs_tokens: torch.Tensor,
        roi_tokens: torch.Tensor,
        time_ids: torch.Tensor | None = None,
        roi_state: torch.Tensor | None = None,
        roi_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fs(fs_tokens), self.roi(roi_tokens)


class DeltaFusion(nn.Module):

    def __init__(self, dim: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.fs = self._branch(dim, dropout)
        self.roi = self._branch(dim, dropout)
        self.gate_fs = nn.Parameter(torch.zeros(1))
        self.gate_roi = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _branch(dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _delta(tokens: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != 4:
            raise ValueError(f"expected [B,4,D] tokens, got {tuple(tokens.shape)}")
        d = torch.cat(
            (tokens[:, 2] - tokens[:, 0], tokens[:, 3] - tokens[:, 1]), dim=-1
        )
        if valid is not None:
            m = valid.unsqueeze(-1).to(d.dtype)
            d = (d.view(d.shape[0], 2, -1) * m).view(d.shape[0], -1)
        return d

    def forward(
        self,
        fs_tokens: torch.Tensor,
        roi_tokens: torch.Tensor,
        roi_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid: torch.Tensor | None = None
        if roi_state is not None:
            obs = (roi_state > 0).to(torch.bool)
            valid = torch.stack((obs[:, 2] & obs[:, 0], obs[:, 3] & obs[:, 1]), dim=-1)
        d_fs = torch.tanh(self.gate_fs) * self.fs(self._delta(fs_tokens))
        d_roi = self.roi(self._delta(roi_tokens))
        if valid is not None:
            m = valid.unsqueeze(-1).to(d_roi.dtype)
            d_roi = (d_roi.view(d_roi.shape[0], 2, -1) * m).view(d_roi.shape[0], -1)
        d_roi = torch.tanh(self.gate_roi) * d_roi
        return d_fs, d_roi


class DualTemporalT2(nn.Module):

    def __init__(
        self,
        dim: int = 512,
        heads: int = 4,
        dropout: float = 0.1,
        share: bool = False,
        in_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.share = share
        self.fs = TemporalAttentionPool(
            dim=dim, heads=heads, dropout=dropout, n_times=4, n_states=1, in_dim=in_dim
        )
        self.roi = (
            self.fs
            if share
            else TemporalAttentionPool(
                dim=dim, heads=heads, dropout=dropout, n_times=4, n_states=3,
                in_dim=in_dim,
            )
        )

    def forward(
        self,
        fs_tokens: torch.Tensor,
        roi_tokens: torch.Tensor,
        time_ids: torch.Tensor,
        roi_state: torch.Tensor,
        roi_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fs_node = self.fs(fs_tokens, time_ids)
        roi_mask = None if roi_valid is None else ~roi_valid
        roi_node = self.roi(roi_tokens, time_ids, roi_state, roi_mask)
        return fs_node, roi_node

if __name__ == "__main__":
    torch.manual_seed(0)
    b, t, d = 2, 4, 512
    fs_tokens = torch.randn(b, t, d)
    roi_tokens = torch.randn(b, t, d)
    time_ids = torch.arange(t).unsqueeze(0).expand(b, -1)
    roi_state = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 2]], dtype=torch.long)
    roi_valid = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)

    model = DualTemporalT2()
    fs_node, roi_node = model(fs_tokens, roi_tokens, time_ids, roi_state, roi_valid)
    print("fs_node", tuple(fs_node.shape))
    print("roi_node", tuple(roi_node.shape))
    assert fs_node.shape == (b, d)
    assert roi_node.shape == (b, d)
    print("ok")
