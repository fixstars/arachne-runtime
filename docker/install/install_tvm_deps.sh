#!/bin/bash

set -e
set -u
set -o pipefail

apt-get update
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev libopenblas-dev
