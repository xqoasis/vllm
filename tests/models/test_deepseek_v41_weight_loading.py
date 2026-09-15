# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from importlib import import_module

import pytest
import torch
from torch import nn

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_test


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = nn.Module()
        layer.tensor_a = nn.Parameter(torch.zeros(1), requires_grad=False)
        layer.tensor_c = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.layers = nn.ModuleList([layer])
        self.finalized_values: list[tuple[float, float]] = []
        self.finalize_calls: list[str] = []

    def load_weights(self, weights) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    def finalize_mega_moe_weights(self) -> None:
        layer = self.layers[0]
        self.finalized_values.append((layer.tensor_a.item(), layer.tensor_c.item()))
        self.finalize_calls.append("mega_moe")
        # Finalization replaces loader parameters with kernel-specific storage.
        layer.tensor_a = layer.tensor_c = None

    def finalize_mhc_broadcast_weights(self) -> None:
        self.finalize_calls.append("mhc")


@pytest.fixture
def model():
    backend = "amd" if current_platform.is_rocm() else "nvidia"
    vl_module = import_module(f"vllm.models.deepseek_v41.{backend}.vl_model")
    text_module = import_module(f"vllm.models.deepseek_v41.{backend}.model")
    language_model = object.__new__(text_module.DeepseekV41LLMForCausalLM)
    nn.Module.__init__(language_model)
    language_model.model = _Backbone()
    language_model.lm_head = nn.Linear(1, 1, bias=False)
    language_model.hf_to_vllm_mapper = WeightsMapper()

    wrapper = object.__new__(vl_module.DeepseekV41ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.language_model = language_model
    wrapper.vision = nn.Linear(1, 1, bias=False)
    wrapper.image_start = nn.Parameter(torch.zeros(1), requires_grad=False)
    wrapper.hf_to_vllm_mapper = vl_module._make_deepseek_v4_vl_weights_mapper(
        "fp4", "weight_scale_inv"
    )
    return wrapper


@pytest.mark.parametrize("skip_vision", [False, True])
def test_interleaved_weights_stream_and_finalize_once(model, skip_vision):
    backbone = model.language_model.model
    if skip_vision:
        model.vision = StageMissingLayer("image", model.vision)
    callback_values = []
    param = backbone.layers[0].tensor_c
    original_pointer = param.data_ptr()

    def weight_loader(param, value):
        callback_values.append(value.item())
        param.data.copy_(value)

    param.weight_loader = weight_loader

    def weights():
        yield "layers.0.tensor_a", torch.tensor([1.0])
        yield "image_start", torch.tensor([2.0])
        assert backbone.layers[0].tensor_a.item() == 1.0
        assert backbone.finalize_calls == []
        yield "layers.0.tensor_c", torch.tensor([3.0])
        yield "vision.weight", torch.tensor([[4.0]])
        assert callback_values == [3.0]
        assert param.data_ptr() == original_pointer
        yield "mtp.0.unused", torch.tensor([5.0])
        yield "head.weight", torch.tensor([[6.0]])
        assert backbone.finalize_calls == []

    loaded = model.load_weights(weights())

    expected = {
        "language_model.model.layers.0.tensor_a",
        "image_start",
        "language_model.model.layers.0.tensor_c",
        "language_model.lm_head.weight",
    }
    if not skip_vision:
        expected.add("vision.weight")
        assert model.vision.weight.item() == 4.0
    assert loaded == expected
    assert model.language_model.lm_head.weight.item() == 6.0
    assert backbone.finalized_values == [(1.0, 3.0)]
    assert backbone.finalize_calls == ["mega_moe", "mhc"]
    model.process_weights_after_loading()
    assert backbone.finalize_calls == ["mega_moe", "mhc"]


def test_stream_releases_consumed_checkpoint_buffers(model):
    buffers: list[weakref.ReferenceType[torch.Tensor]] = []

    def weights():
        for index in range(8):
            if index == 4:
                assert buffers[0]() is None
            storage = torch.full((1024,), float(index))
            buffers.append(weakref.ref(storage))
            yield "layers.0.tensor_a", storage[:1]
            del storage

    model.load_weights(weights())

    assert all(buffer() is None for buffer in buffers)
    assert model.language_model.model.finalized_values == [(7.0, 0.0)]


def test_dummy_weights_finalize_once(model):
    model.process_weights_after_loading()
    model.process_weights_after_loading()

    assert model.language_model.model.finalize_calls == ["mega_moe", "mhc"]


def test_failed_stream_does_not_finalize_partial_weights(model):
    def weights():
        yield "layers.0.tensor_a", torch.tensor([1.0])
        raise OSError("checkpoint read failed")

    with pytest.raises(OSError, match="checkpoint read failed"):
        model.load_weights(weights())

    assert model.language_model.model.finalize_calls == []
    assert not model._weights_finalized
