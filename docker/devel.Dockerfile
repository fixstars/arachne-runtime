FROM ubuntu:20.04

ENV LANG C.UTF-8
ENV PYTHONIOENCODING utf-8
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY install/install_python3.sh /install/install_python3.sh
RUN bash /install/install_python3.sh

COPY install/install_devtools.sh /install/install_devtools.sh
RUN bash /install/install_devtools.sh

COPY install/install_tvm_deps.sh /install/install_tvm_deps.sh
RUN bash /install/install_tvm_deps.sh

COPY install/install_opencv_deps.sh /install/install_opencv_deps.sh
RUN bash /install/install_opencv_deps.sh

COPY install/install_poetry.sh /install/install_poetry.sh
RUN bash /install/install_poetry.sh

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

