FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

ENV LANG C.UTF-8
ENV PYTHONIOENCODING=utf-8

# Install LLVM to build TVM
RUN echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main >> /etc/apt/sources.list.d/llvm.list \
    && echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main >> /etc/apt/sources.list.d/llvm.list \
    && apt-key adv --fetch-keys http://apt.llvm.org/llvm-snapshot.gpg.key \
    && apt-get update && apt-get install -y llvm-11 clang-11

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # for tvm
    libopenblas-dev \
    # for open-cv
    libgl1-mesa-dev \
    sudo \
    curl \
    git \
    ssh

# python -> python3
RUN ln -s $(which python3) /usr/local/bin/python
RUN ln -s $(which pip3) /usr/local/bin/pip

# Install dependencies for arachne-runtime
RUN python3 -m pip install --upgrade pip --no-cache-dir \
    && python3 -m pip install --no-cache-dir \
    tensorflow==1.6.0 \
    onnxruntime-gpu==1.10.0 \
    numpy \
    packaging \
    pyyaml \
    grpcio \
    grpcio-tools \
    opencv-python \
    pytest \
    cmake \
    ninja

RUN git clone https://github.com/fixstars/arachne-runtime.git /workspaces/arachne-runtime

WORKDIR /workspaces/arachne-runtime
ENV PYTHONPATH=/workspaces/arachne-runtime/python
ENV TVM_LIBRARY_PATH=/workspaces/arachne-runtime/build/tvm

RUN git checkout feature/port-from-arachne
RUN git clone --recursive https://github.com/fixstars/tvm 3rdparty/tvm
# RUN ./scripts/install_tvm.sh gpu

# ENTRYPOINT [ "poetry", "run", "pytest", "tests" ]
