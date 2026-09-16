# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[2] / "tools/download_cuda_runfile.sh"
# MD5 of b"abc", used only to test transfer-integrity checks.
DIGEST = "900150983cd24fb0d6963f7d28e17f72"


@pytest.mark.parametrize(
    ("download_status", "payload", "digest", "success"),
    [
        (0, "abc", DIGEST, True),
        (0, "bad", DIGEST, False),
        (42, "abc", DIGEST, False),
        (0, "abc", "invalid", False),
    ],
)
def test_requires_successful_transfer_and_matching_digest(
    tmp_path: Path, download_status: int, payload: str, digest: str, success: bool
):
    downloader = tmp_path / "aria2c"
    downloader.write_text(
        "#!/bin/bash\n"
        'printf %s "$TEST_PAYLOAD" > cuda.run\n'
        'exit "$TEST_DOWNLOAD_STATUS"\n'
    )
    downloader.chmod(0o755)
    result = subprocess.run(
        ["bash", str(HELPER), "https://example.invalid/cuda.run", "cuda.run", digest],
        cwd=tmp_path,
        env={
            "PATH": f"{tmp_path}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "TEST_PAYLOAD": payload,
            "TEST_DOWNLOAD_STATUS": str(download_status),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    if success:
        assert (tmp_path / "cuda.run").read_text() == payload
    elif digest == "invalid":
        assert not (tmp_path / "cuda.run").exists()
