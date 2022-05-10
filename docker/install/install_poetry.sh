#!/bin/bash

set -e
set -u
set -o pipefail

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=/usr/local/poetry python3 - --version 1.2.0a2
