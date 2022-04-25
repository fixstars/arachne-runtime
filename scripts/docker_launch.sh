#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run \
    --rm \
    -it \
    -u root \
    --gpus all \
    -v $(pwd):/workspaces/arachne-runtime \
    -w /workspaces/arachne-runtime \
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GID=$(id -g)" \
    -e "PYTHONPATH=/workspaces/arachne-runtime/python" \
    arachne-runtime:devel \
    bash /workspaces/arachne-runtime/scripts/_docker_init.sh
# -e "TVM_LIBRARY_PATH=/workspaces/arachne-runtime/build/tvm" \
