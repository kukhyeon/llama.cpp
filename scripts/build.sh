#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
JOBS="${JOBS:-2}"
TARGET="${TARGET:-ignite}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" || exit

cmake -S "${SCRIPT_DIR}/.." \
      -DLLAMA_CURL=ON \
      -DLLAMA_BUILD_IGNITE=ON \
      -DLLAMA_IGNITE_INSTALL=ON \
      -DIGNITE_USE_SYSTEM_DVFS=ON
cmake --build . --config Debug --target "${TARGET}" -j "${JOBS}"
