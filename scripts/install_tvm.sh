#! /bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")" || exit; pwd)

if [ $# -ne 1 ]; then
    echo "Usage: $0 [cpu | gpu]" 1>&2
    exit 1
fi

TARGET="$1"
TVM_SOURCE_DIR=${SCRIPT_DIR}/../3rdparty/tvm
BUILD_DIR=${SCRIPT_DIR}/../build/tvm

mkdir -p "${BUILD_DIR}"
cp "${TVM_SOURCE_DIR}/cmake/config.cmake" "${BUILD_DIR}/config.cmake"

if [ "${TARGET}" = "cpu" ]; then
    :
else # TARGET = gpu
    {
        echo "set(USE_LLVM llvm-config-11)"
        echo "set(USE_BLAS openblas)"
        echo "set(USE_CUDA /usr/local/cuda)"
        echo "set(USE_CUDNN ON)"
        echo "set(USE_CUBLAS ON)"
        # echo "set(USE_TENSORRT_CODEGEN ON)"
        # echo "set(USE_TENSORRT_RUNTIME ON)"
    } >> "${BUILD_DIR}/config.cmake"
fi

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S "${TVM_SOURCE_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}"

cd "${TVM_SOURCE_DIR}/python" || exit
python -m pip install -e .
