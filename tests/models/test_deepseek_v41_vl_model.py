# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Iterable

import pytest
import torch

from vllm.model_executor.models.utils import StageMissingLayer, WeightsMapper
from vllm.platforms import current_platform


@pytest.fixture(params=["nvidia", "amd"])
def wrapper_class(request):
    if request.param == "amd" and not current_platform.is_rocm():
        pytest.skip("AMD model imports require a ROCm device")
    return importlib.import_module(
        f"vllm.models.deepseek_v41.{request.param}.vl_model"
    ).DeepseekV41ForCausalLM


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))
        self.load_calls = 0
        self.finalize_calls = 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.load_calls += 1
        loaded = set()
        for name, value in weights:
            self.get_parameter(name).data.copy_(value)
            loaded.add(name)
        assert loaded == {"a", "b"}, "finalization requires all language weights"
        self.process_weights_after_loading()
        return loaded

    def process_weights_after_loading(self):
        self.finalize_calls += 1


class _VLWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.vision = torch.nn.Linear(1, 1, bias=False)
        self.image_start = torch.nn.Parameter(torch.zeros(1))
        self.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"text.": "language_model.", "mtp.": None}
        )


def test_streams_language_weights_and_finalizes_once(wrapper_class):
    model = _VLWrapper()

    def weights():
        yield "text.a", torch.tensor([7.0])
        assert model.language_model.a.item() == 7, (
            "source was advanced before the previous language tensor was loaded"
        )
        yield "vision.weight", torch.tensor([[3.0]])
        yield "text.b", torch.tensor([9.0])
        yield "image_start", torch.tensor([2.0])
        yield "mtp.unused", torch.tensor([0.0])

    loaded = wrapper_class.load_weights(model, weights())
    wrapper_class.process_weights_after_loading(model)

    assert loaded == {
        "language_model.a",
        "language_model.b",
        "vision.weight",
        "image_start",
    }
    assert model.language_model.b.item() == 9
    assert model.language_model.load_calls == 1
    assert model.language_model.finalize_calls == 1
    assert model.vision.weight.item() == 3
    assert model.image_start.item() == 2
    assert model._weights_finalized


def test_encoder_only_skips_language_model(wrapper_class):
    model = _VLWrapper()
    retained = model.language_model
    model.language_model = StageMissingLayer("language_model", retained)
    loaded = wrapper_class.load_weights(
        model,
        iter(
            [
                ("text.a", torch.tensor([7.0])),
                ("vision.weight", torch.tensor([[3.0]])),
                ("text.b", torch.tensor([9.0])),
            ]
        ),
    )
    assert loaded == {"vision.weight"}
    assert retained.load_calls == 0
    assert retained.a.item() == 0
    assert model.vision.weight.item() == 3


def test_decoder_only_skips_vision_tower(wrapper_class):
    model = _VLWrapper()
    retained = model.vision
    model.vision = StageMissingLayer("vision", retained)
    loaded = wrapper_class.load_weights(
        model,
        iter(
            [
                ("text.a", torch.tensor([7.0])),
                ("vision.weight", torch.tensor([[3.0]])),
                ("text.b", torch.tensor([9.0])),
            ]
        ),
    )
    assert loaded == {"language_model.a", "language_model.b"}
    assert model.language_model.finalize_calls == 1


def test_failed_load_does_not_mark_weights_finalized(wrapper_class):
    model = _VLWrapper()
    with pytest.raises(AssertionError, match="all language weights"):
        wrapper_class.load_weights(model, [("text.a", torch.tensor([7.0]))])
    assert not hasattr(model, "_weights_finalized")
