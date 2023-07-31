#!/bin/bash

set -e
set -u
set -o pipefail

curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local/poetry python3 - --version 1.5.1
