# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4RMSNorm,
    DeepseekV4ViT,
    apply_rotary,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA vision fusion"
)


@pytest.mark.parametrize("tokens", [0, 1, 37, 5329])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_vision_norm_preserves_fp32_arithmetic(tokens, dtype):
    torch.manual_seed(915)
    x = torch.randn(tokens, 2048, device="cuda", dtype=dtype)[:, :1024]
    norm = DeepseekV4RMSNorm(1024).cuda().to(dtype)
    norm.weight.uniform_(0.1, 1.5)
    xf = x.float()
    expected = (
        norm.weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps))
    ).to(dtype)
    actual = norm(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype


@pytest.mark.parametrize("tokens", [0, 1, 37, 5329])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_vision_rotary_handles_packed_qkv_views(tokens, dtype):
    torch.manual_seed(915)
    packed = torch.randn(tokens, 3 * 1024, device="cuda", dtype=dtype)
    cos = torch.randn(tokens, 1, 32, device="cuda")
    sin = torch.randn_like(cos)
    original = packed.clone()
    for x in packed.chunk(3, -1)[:2]:
        x = x.view(tokens, 16, 64)
        x1, x2 = x.float().chunk(2, -1)
        expected = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(
            dtype
        )
        torch.testing.assert_close(apply_rotary(x, cos, sin), expected, rtol=0, atol=0)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 2048), (2, 3, 2048)])
@torch.inference_mode()
def test_vision_norm_keeps_native_fallback_for_other_layouts(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)[..., ::2]
    norm = DeepseekV4RMSNorm(1024).cuda().bfloat16()
    xf = x.float()
    expected = (
        norm.weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps))
    ).to(x.dtype)
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("grids", [[[3, 5], [2, 7], [1, 4]], [[17, 19], [17, 19]]])
@torch.inference_mode()
def test_packed_tower_preserves_image_boundaries_and_spatial_merge(dist_init, grids):
    config = DeepseekV41Config(
        text_config={"hidden_size": 128},
        vision_config={
            "hidden_size": 128,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 256,
            "patch_size": 2,
            "downsample_ratio": 3,
        },
    )
    torch.manual_seed(915)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        vision = DeepseekV4ViT(config).cuda().eval()
        aligner = DeepseekV4Aligner(config).cuda().eval()
    finally:
        torch.set_default_dtype(old_dtype)
    if not vision.supports_packed_attention:
        pytest.skip("Packed CUDA FlashAttention is unavailable")
    for module in (vision, aligner):
        for name, param in module.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1)
            else:
                param.normal_(std=0.02)
    sizes = [h * w for h, w in grids]
    patches = torch.randn(sum(sizes), 3, 2, 2, device="cuda", dtype=torch.bfloat16)
    serial = [
        aligner(vision(image, h, w), h, w)
        for image, (h, w) in zip(patches.split(sizes), grids, strict=True)
    ]
    packed = aligner.forward_packed(vision.forward_packed(patches, grids), grids)
    for actual, expected in zip(packed, serial, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-3)
    changed = patches.clone()
    changed[sizes[0] :] *= -4
    isolated = aligner.forward_packed(vision.forward_packed(changed, grids), grids)
    torch.testing.assert_close(isolated[0], packed[0], rtol=0, atol=0)

    if len(grids) == 3:
        from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM

        model = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
        torch.nn.Module.__init__(model)
        model.vision, model.aligner = vision, aligner
        for limits, expected_batches in [
            ((1, 1000, 10000), [15, 14, 4]),
            ((8, 28, 10000), [15, 18]),
            ((8, 1000, 421), [29, 4]),
            ((8, 10, 100), [15, 14, 4]),
        ]:
            model._vit_batch_limits = limits
            batch_tokens: list[int] = []
            hook = vision.patch_embed.register_forward_pre_hook(
                lambda module, args, counts=batch_tokens: counts.append(
                    args[0].shape[0]
                )
            )
            try:
                bounded = model._encode_image_batches(patches, grids)
            finally:
                hook.remove()
            assert batch_tokens == expected_batches
            for actual, expected in zip(bounded, serial, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-3)
