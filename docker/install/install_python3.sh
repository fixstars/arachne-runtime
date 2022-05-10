#!/bin/bash

set -e
set -u
set -o pipefail

apt-get update
apt-get install -y python3 python3-dev python3-pip python3-venv

# python -> python3
ln -s $(which python3) /usr/local/bin/python
ln -s $(which pip3) /usr/local/bin/pip
