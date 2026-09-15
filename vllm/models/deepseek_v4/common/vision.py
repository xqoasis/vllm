# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 vision tower (ViT + aligner) with TP-sharded linears.

Ported from the official reference implementation
(deepseek-ai/DeepSeek-V4-Flash-Vision-Exp). Weight names match the HF
checkpoint so no renaming is needed at load time. Attention and MLP weights
are tensor-parallel sharded (replicated when the vision head count is not
divisible by TP size, or under ``--mm-encoder-tp-mode data``); the patch
embed and norms are replicated, so the residual stream is full-width on
every rank.
"""

import itertools
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.vision import (
    get_load_balance_assignment,
    is_vit_use_data_parallel,
)
from vllm.models.deepseek_v4.common.ops.fused_vision import (
    vision_rms_norm,
    vision_rotary,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@lru_cache(8)
def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if (
        current_platform.is_cuda()
        and x.is_cuda
        and x.ndim == 3
        and x.stride(-1) == 1
        and cos.is_contiguous()
        and sin.is_contiguous()
        and cos.shape == sin.shape == (x.shape[0], 1, x.shape[-1] // 2)
        and cos.dtype == sin.dtype == torch.float32
        and not torch.is_grad_enabled()
    ):
        return vision_rotary(x, cos, sin)
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            current_platform.is_cuda()
            and x.is_cuda
            and x.ndim == 2
            and x.stride(-1) == 1
            and self.weight.is_contiguous()
            and self.weight.dtype != torch.float64
            and not torch.is_grad_enabled()
        ):
            return vision_rms_norm(x, self.weight, self.eps)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class DeepseekV4PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Replicated: the residual stream is full-width on every rank.
        self.proj = ReplicatedLinear(
            3 * config.vision_patch_size**2,
            config.vision_dim,
            bias=True,
            quant_config=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.proj(x.flatten(1))
        return out


class DeepseekV4VisionAttention(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        # Convention from minimax_m3/siglip: compute the TP/DP choice locally
        # instead of threading a flag through every constructor.
        use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.n_heads = config.vision_n_heads // self.tp_size
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.hidden_size = config.vision_dim
        self.wqkv = QKVParallelLinear(
            config.vision_dim,
            self.head_dim,
            config.vision_n_heads,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.wqkv",
            disable_tp=use_data_parallel,
        )
        self.wo = RowParallelLinear(
            config.vision_dim,
            config.vision_dim,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.wo",
            disable_tp=use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = x.size(0)
        qkv, _ = self.wqkv(x)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in qkv.chunk(3, -1))
        q = apply_rotary(q, cos, sin).unsqueeze(0)  # (b=1, n, h, d)
        k = apply_rotary(k, cos, sin).unsqueeze(0)
        o = self.attn(
            q, k, v.unsqueeze(0), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
        )
        out, _ = self.wo(o.reshape(n, -1))
        return out


class DeepseekV4VisionMLP(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
        self.w1 = MergedColumnParallelLinear(
            config.vision_dim,
            [config.vision_inter_dim] * 2,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.w1",
            disable_tp=use_data_parallel,
        )
        self.w2 = RowParallelLinear(
            config.vision_inter_dim,
            config.vision_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.w2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w1(x)
        out, _ = self.w2(self.act_fn(gate_up))
        return out


class DeepseekV4VisionBlock(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.norm1 = DeepseekV4RMSNorm(config.vision_dim)
        self.attn = DeepseekV4VisionAttention(config, prefix=f"{prefix}.attn")
        self.norm2 = DeepseekV4RMSNorm(config.vision_dim)
        self.mlp = DeepseekV4VisionMLP(config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens, max_seqlen)
        return x + self.mlp(self.norm2(x))


class DeepseekV4ViT(nn.Module):
    """DeepSeek-V4 ViT: full bidirectional attention per image, 2D RoPE."""

    def __init__(self, config):
        super().__init__()
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = DeepseekV4PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [
                DeepseekV4VisionBlock(config, prefix=f"blocks.{i}")
                for i in range(config.vision_n_layers)
            ]
        )
        self.norm = DeepseekV4RMSNorm(config.vision_dim)

    @property
    def supports_packed_attention(self) -> bool:
        return current_platform.is_cuda() and all(
            block.attn.attn.attn_backend == AttentionBackendEnum.FLASH_ATTN
            and not block.attn.attn.fp8_enabled
            for block in self.blocks
        )

    def forward_packed(
        self, patches: torch.Tensor, vit_grid: list[list[int]]
    ) -> torch.Tensor:
        """Encode packed image tokens without attention across image boundaries."""
        sizes = [h * w for h, w in vit_grid]
        if not sizes or min(sizes) <= 0 or sum(sizes) != patches.shape[0]:
            raise ValueError("Image grids must partition the supplied patches")
        x = self.patch_embed(patches)
        tables = [
            get_vision_cos_sin(h, w, self.rope_dim, self.rope_theta)
            for h, w in vit_grid
        ]
        cos, sin = (torch.cat(parts).to(x.device) for parts in zip(*tables))
        cu_seqlens = torch.tensor(
            [0, *itertools.accumulate(sizes)], dtype=torch.int32, device=x.device
        )
        # The attention wrapper reads this scalar on the host at every layer.
        max_seqlen = torch.tensor(max(sizes), dtype=torch.int32, device="cpu")
        for block in self.blocks:
            x = block(x, cos, sin, cu_seqlens, max_seqlen)
        return self.norm(x)

    def forward(
        self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int
    ) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_vit_h, n_vit_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=x.device)
        sin = sin.to(device=x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4Aligner(nn.Module):
    """Spatial merge (downsample_ratio x downsample_ratio) + MLP projector."""

    def __init__(self, config):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
        self.downsample_ratio = config.vision_downsample_ratio
        self.out_dim = config.hidden_size
        in_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = ColumnParallelLinear(
            in_dim,
            config.hidden_size,
            bias=True,
            quant_config=None,
            disable_tp=use_data_parallel,
        )
        self.w2 = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            quant_config=None,
            disable_tp=use_data_parallel,
        )

    def _merge(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_vit_h, n_vit_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_vit_w % r, 0, -n_vit_h % r))
        return F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)

    def forward(self, x: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        x = self._merge(x, n_vit_h, n_vit_w)
        hidden, _ = self.w1(x)
        out, _ = self.w2(F.gelu(hidden))
        return out

    def forward_packed(
        self, x: torch.Tensor, vit_grid: list[list[int]]
    ) -> tuple[torch.Tensor, ...]:
        """Preserve each spatial merge and batch the final projection."""
        sizes = [h * w for h, w in vit_grid]
        merged = [
            self._merge(image, h, w)
            for image, (h, w) in zip(x.split(sizes), vit_grid, strict=True)
        ]
        # Changing the first projection's row count changes its BF16 rounding.
        hidden = torch.cat([self.w1(image)[0] for image in merged])
        out, _ = self.w2(F.gelu(hidden))
        return out.split([image.shape[0] for image in merged])


def run_dp_sharded_vision_tower(
    vision_model: DeepseekV4ViT,
    aligner: DeepseekV4Aligner,
    patches: torch.Tensor,
    vit_grid: list[list[int]],
) -> list[torch.Tensor]:
    """Run the ViT + aligner with images sharded across TP ranks.

    Every rank holds the full tower weights (``--mm-encoder-tp-mode data``)
    and receives the full ``patches`` batch. Images are assigned to ranks by
    patch count (greedy load balancing), each rank encodes only its share,
    and per-image embeddings are exchanged with one padded all-gather and
    returned in the original image order.

    Args:
        vision_model: The (weight-replicated) ViT tower.
        aligner: The (weight-replicated) spatial-merge projector.
        patches: ``(sum(n_vit_h * n_vit_w), 3, p, p)`` patches of all images.
        vit_grid: ``[n_vit_h, n_vit_w]`` per image.

    Returns:
        One ``(n_aligner_rows, hidden_size)`` embedding tensor per image.
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    sizes = [h * w for h, w in vit_grid]
    cum_patches = [0, *itertools.accumulate(sizes)]
    image_to_tp_rank, gpu_sample_counts, _ = get_load_balance_assignment(sizes, tp_size)
    cum_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    # Rows the aligner emits per image: each grid dim is padded up to a
    # multiple of the merge ratio before r x r patches fold into one row.
    r = aligner.downsample_ratio
    rows_per_image = [-(-h // r) * -(-w // r) for h, w in vit_grid]

    def rank_image_idxs(g: int) -> list[int]:
        return image_to_tp_rank[cum_sample_counts[g] : cum_sample_counts[g + 1]]

    # All-gather needs one shape on every rank; pad each rank's packed
    # output to the largest per-rank row count (computable locally).
    max_rows = max(
        sum(rows_per_image[i] for i in rank_image_idxs(g)) for g in range(tp_size)
    )

    local_embeds = [
        aligner(
            vision_model(
                patches[cum_patches[i] : cum_patches[i + 1]],
                vit_grid[i][0],
                vit_grid[i][1],
            ),
            vit_grid[i][0],
            vit_grid[i][1],
        )
        for i in rank_image_idxs(tp_rank)
    ]
    if local_embeds:
        embeds_local = torch.cat(local_embeds, dim=0)
    else:
        embeds_local = patches.new_zeros((0, aligner.out_dim))
    if embeds_local.shape[0] < max_rows:
        embeds_local = torch.cat(
            [
                embeds_local,
                embeds_local.new_zeros(
                    (max_rows - embeds_local.shape[0], embeds_local.shape[1])
                ),
            ],
            dim=0,
        )
    gathered = tensor_model_parallel_all_gather(embeds_local.contiguous(), dim=0)

    out: list[torch.Tensor] = [None] * len(vit_grid)  # type: ignore[list-item]
    for g in range(tp_size):
        offset = g * max_rows
        for i in rank_image_idxs(g):
            out[i] = gathered[offset : offset + rows_per_image[i]]
            offset += rows_per_image[i]
    return out
