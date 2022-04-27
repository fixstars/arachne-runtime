FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV PYTHONIOENCODING=utf-8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # for open-cv
    libgl1-mesa-dev \
    sudo \
    curl \
    git

# python -> python3
RUN ln -s $(which python3) /usr/local/bin/python
RUN ln -s $(which pip3) /usr/local/bin/pip

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=/usr/local/poetry python - --version 1.2.0a2
ENV PATH /usr/local/poetry/bin:$PATH

COPY . /workspaces/arachne-runtime
WORKDIR /workspaces/arachne-runtime

ENV PYTHONPATH=/workspaces/arachne-runtime/python
ENV TVM_LIBRARY_PATH=/workspaces/arachne-runtime/build/tvm

RUN git clone --recursive https://github.com/fixstars/tvm 3rdparty/tvm
RUN poetry install
RUN poetry run ./scripts/install_tvm.sh cpu

ENTRYPOINT [ "poetry", "run", "pytest", "tests", "--forked" ]
