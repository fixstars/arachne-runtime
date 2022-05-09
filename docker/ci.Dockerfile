FROM ubuntu:18.04

ENV LANG C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY docker/install/install_python3.sh /install/install_python3.sh
RUN bash /install/install_python3.sh

COPY docker/install/install_devtools.sh /install/install_devtools.sh
RUN bash /install/install_devtools.sh

COPY docker/install/install_tvm_deps.sh /install/install_tvm_deps.sh
RUN bash /install/install_tvm_deps.sh

COPY docker/install/install_opencv_deps.sh /install/install_opencv_deps.sh
RUN bash /install/install_opencv_deps.sh

COPY docker/install/install_poetry.sh /install/install_poetry.sh
RUN bash /install/install_poetry.sh

ENV PATH /usr/local/poetry/bin:$PATH

COPY . /workspaces/arachne-runtime
WORKDIR /workspaces/arachne-runtime

ENV PYTHONPATH=/workspaces/arachne-runtime/python
ENV TVM_LIBRARY_PATH=/workspaces/arachne-runtime/build/tvm

RUN git clone --recursive https://github.com/fixstars/tvm 3rdparty/tvm
RUN poetry install
RUN poetry run ./scripts/install_tvm.sh cpu

ENTRYPOINT [ "poetry", "run", "pytest", "tests" ]
