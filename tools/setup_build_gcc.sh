#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Select and validate the image's existing compilers without installing packages.
setup_build_gcc() {
    local cc="${CC:-gcc}" cxx="${CXX:-g++}"
    local cc_version cxx_version

    cc_version="$("${cc}" -dumpfullversion -dumpversion 2>/dev/null || true)"
    cxx_version="$("${cxx}" -dumpfullversion -dumpversion 2>/dev/null || true)"
    if [[ ! "${cc_version}" =~ ^[0-9]+([.][0-9]+)*$ ||
          "${cc_version}" != "${cxx_version}" ]] ||
       (( ${cc_version%%.*} < 13 )); then
        echo "Provide matching GCC/G++ >= 13 in the build image or select installed compilers via CC and CXX." >&2
        return 1
    fi

    CC="$(command -v "${cc}")" || return 1
    CXX="$(command -v "${cxx}")" || return 1
    export CC CXX
    export CUDAHOSTCXX="${CXX}"

    echo "CC=${CC}"
    echo "CXX=${CXX}"
    echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
    "${CC}" --version
    "${CXX}" --version

    local probe_dir
    probe_dir="$(mktemp -d)" || return 1
    if ! "${CXX}" -std=c++20 -x c++ -o "${probe_dir}/format" - <<'CPP'
#include <format>
#if defined(__clang__) || !defined(__GNUC__) || __GNUC__ < 13
#error GCC >= 13 is required
#endif
int main() { return std::format("{}", 13) != "13"; }
CPP
    then
        rm -rf "${probe_dir}"
        echo "${CXX} cannot compile/link C++20 std::format; install GCC >= 13 and its matching libstdc++ development package." >&2
        return 1
    fi
    if ! "${probe_dir}/format"; then
        rm -rf "${probe_dir}"
        echo "The C++20 std::format probe cannot run; check the selected libstdc++ runtime." >&2
        return 1
    fi
    rm -rf "${probe_dir}"

    # setup.py splits CMAKE_ARGS on whitespace; compiler paths must stay intact.
    if [[ "${CC}${CXX}" =~ [[:space:]] ]]; then
        echo "Compiler paths must not contain whitespace" >&2
        return 1
    fi
    # --fresh prevents a previous build's cached GCC/CUDA host compiler selection.
    export CMAKE_ARGS="${CMAKE_ARGS:-} --fresh -DCMAKE_C_COMPILER:FILEPATH=${CC} -DCMAKE_CXX_COMPILER:FILEPATH=${CXX} -DCMAKE_CUDA_HOST_COMPILER:FILEPATH=${CUDAHOSTCXX}"
}
