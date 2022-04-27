FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV PYTHONIOENCODING utf-8
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
    git \
    ssh

# python -> python3
RUN ln -s "$(which python3)" /usr/local/bin/python
RUN ln -s "$(which pip3)" /usr/local/bin/pip

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | POETRY_HOME=/usr/local/poetry python - --version 1.2.0a2

# Add a user that UID:GID will be updated by vscode
ARG USERNAME=developer
ARG GROUPNAME=developer
ARG UID=1000
ARG GID=1000
ARG PASSWORD=developer
RUN groupadd -g $GID $GROUPNAME && \
    useradd -l -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME
ENV HOME /home/developer
ENV PATH $HOME/.local/bin:/usr/local/poetry/bin:$PATH

