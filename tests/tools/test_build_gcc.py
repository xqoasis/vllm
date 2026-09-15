# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import shlex
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[2] / "tools/setup_build_gcc.sh"


def executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/bash\nset -eu\n{body}\n")
    path.chmod(0o755)


def compiler(
    path: Path, version: int, probe_status: int = 0, runtime_status: int = 0
) -> None:
    executable(
        path,
        f"""case "$1" in
    -dumpfullversion) echo {version}.2.0 ;;
    --version) echo 'gcc {version}.2.0' ;;
    *)
        cat >/dev/null
        if [ {probe_status} -ne 0 ]; then exit {probe_status}; fi
        while [ "$1" != -o ]; do shift; done
        printf '#!/bin/bash\\nexit {runtime_status}\\n' > "$2"
        chmod +x "$2"
        ;;
esac
""",
    )


def run_setup(
    tmp_path: Path,
    installed: int = 13,
    cxx_version: int | None = None,
    explicit_compilers: bool = False,
    probe_status: int = 0,
    runtime_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    compiler(bin_dir / "gcc", installed, probe_status, runtime_status)
    compiler(bin_dir / "g++", cxx_version or installed, probe_status, runtime_status)
    apt_log = tmp_path / "apt.log"
    for tool in ("apt-get", "apt-cache"):
        executable(
            bin_dir / tool,
            f"echo unexpected-package-operation >> {shlex.quote(str(apt_log))}\n"
            "exit 99",
        )
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "CMAKE_ARGS": "-DKEEP_EXISTING_OPTION=ON",
        "LD_LIBRARY_PATH": "/existing/lib",
    }
    if explicit_compilers:
        compiler(bin_dir / "custom-gcc", 13, probe_status, runtime_status)
        compiler(bin_dir / "custom-g++", 13, probe_status, runtime_status)
        env["CC"] = str(bin_dir / "custom-gcc")
        env["CXX"] = str(bin_dir / "custom-g++")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""set -euo pipefail
source {shlex.quote(str(HELPER))}
setup_build_gcc
printf '%s\\n' "$LD_LIBRARY_PATH"
printf '%s\\n' "$CC" "$CXX" "$CUDAHOSTCXX" "$CMAKE_ARGS"
""",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    return result, apt_log


@pytest.mark.parametrize(
    ("installed", "explicit_compilers"),
    [(13, False), (14, False), (12, True)],
)
def test_uses_installed_toolchain_without_package_operations(
    tmp_path: Path, installed: int, explicit_compilers: bool
):
    result, apt_log = run_setup(
        tmp_path, installed, explicit_compilers=explicit_compilers
    )
    assert result.returncode == 0, result.stderr
    prefix = "custom-" if explicit_compilers else ""
    cc = str(tmp_path / "bin" / f"{prefix}gcc")
    cxx = str(tmp_path / "bin" / f"{prefix}g++")
    assert result.stdout.splitlines()[-4:] == [
        cc,
        cxx,
        cxx,
        f"-DKEEP_EXISTING_OPTION=ON --fresh -DCMAKE_C_COMPILER:FILEPATH={cc} "
        f"-DCMAKE_CXX_COMPILER:FILEPATH={cxx} "
        f"-DCMAKE_CUDA_HOST_COMPILER:FILEPATH={cxx}",
    ]
    assert not apt_log.exists()


@pytest.mark.parametrize(("installed", "cxx_version"), [(12, 12), (13, 14)])
def test_rejects_unsuitable_compilers_without_trying_to_install(
    tmp_path: Path, installed: int, cxx_version: int
):
    result, apt_log = run_setup(tmp_path, installed, cxx_version)
    assert result.returncode != 0
    assert "Provide matching GCC/G++ >= 13" in result.stderr
    assert not apt_log.exists()
    assert "CC=" not in result.stdout


def test_missing_format_support_stops_before_cmake(tmp_path: Path):
    result, _ = run_setup(tmp_path, installed=13, probe_status=1)
    assert result.returncode != 0
    assert "cannot compile/link C++20 std::format" in result.stderr


def test_missing_runtime_support_stops_before_cmake(tmp_path: Path):
    result, _ = run_setup(tmp_path, installed=13, runtime_status=1)
    assert result.returncode != 0
    assert "check the selected libstdc++ runtime" in result.stderr
