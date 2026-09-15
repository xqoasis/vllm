# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch an installed V4.1 preview before starting model worker processes."""

import argparse
import ast
import importlib.util
import os
import shutil
import tempfile
import textwrap
from pathlib import Path

_TEXT_LOAD = """\
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self)
    loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
    self.process_weights_after_loading()
    return loaded_params
"""
_STREAM_TEXT_LOAD = """\
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loader = AutoWeightsLoader(self)
    return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
"""
_VL_LOAD = """\
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    mapped = sorted(self.hf_to_vllm_mapper.apply(weights), key=lambda x: x[0])
    loader = AutoWeightsLoader(self)
    loaded_params = loader.load_weights(mapped)
    self._weights_finalized = True
    return loaded_params
"""
_STREAM_VL_LOAD = """\
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    self._weights_finalized = False
    loader = AutoWeightsLoader(self)
    loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
    self.process_weights_after_loading()
    return loaded_params
"""
_VL_FINALIZE = """\
def process_weights_after_loading(self) -> None:
    if getattr(self, "_weights_finalized", False):
        return
    self.language_model.process_weights_after_loading()
"""
_STREAM_VL_FINALIZE = _VL_FINALIZE + "    self._weights_finalized = True\n"


def _replace_method(source: str, class_name: str, old: str, new: str) -> str:
    expected = ast.parse(old).body[0]
    replacement = ast.parse(new).body[0]
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == expected.name
    )
    if ast.dump(method) == ast.dump(replacement):
        return source
    if ast.dump(method) != ast.dump(expected):
        raise RuntimeError(f"Unsupported source: {class_name}.{expected.name}")
    lines = source.splitlines(keepends=True)
    lines[method.lineno - 1 : method.end_lineno] = [textwrap.indent(new, "    ")]
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, help="Path to the vllm package")
    args = parser.parse_args()
    root = args.vllm_root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot locate the active vllm package")
        root = Path(spec.origin).parent

    updates: dict[Path, str] = {}
    for backend in ("nvidia", "amd"):
        directory = root / "models" / "deepseek_v41" / backend
        for filename, class_name, methods in (
            (
                "model.py",
                "DeepseekV41LLMForCausalLM",
                [(_TEXT_LOAD, _STREAM_TEXT_LOAD)],
            ),
            (
                "vl_model.py",
                "DeepseekV41ForCausalLM",
                [(_VL_LOAD, _STREAM_VL_LOAD), (_VL_FINALIZE, _STREAM_VL_FINALIZE)],
            ),
        ):
            path = directory / filename
            original = source = path.read_text()
            for old, new in methods:
                source = _replace_method(source, class_name, old, new)
            compile(source, str(path), "exec")
            if source != original:
                updates[path] = source

    # Validate every file before making backups or replacing installed sources.
    staged: dict[Path, Path] = {}
    try:
        for path, source in updates.items():
            backup = path.with_name(path.name + ".before-streaming-load")
            if not backup.exists():
                shutil.copy2(path, backup)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False
            ) as f:
                staged[path] = Path(f.name)
                f.write(source)
            shutil.copymode(path, staged[path])
        for path, temporary in staged.items():
            os.replace(temporary, path)
            print(f"Patched {path}", flush=True)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    if not updates:
        print(f"Already patched: {root}", flush=True)


if __name__ == "__main__":
    main()
