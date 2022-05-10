FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV LANG C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Asia/Tokyo

# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# A hotfix for https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# Install LLVM to build TVM
RUN echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal-11 main >> /etc/apt/sources.list.d/llvm.list \
    && echo deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-11 main >> /etc/apt/sources.list.d/llvm.list \
    && apt-key adv --fetch-keys http://apt.llvm.org/llvm-snapshot.gpg.key \
    && apt-get update && apt-get install -y llvm-11 clang-11

COPY install/install_python3.sh /install/install_python3.sh
RUN bash /install/install_python3.sh

COPY install/install_devtools.sh /install/install_devtools.sh
RUN bash /install/install_devtools.sh

COPY install/install_tvm_deps.sh /install/install_tvm_deps.sh
RUN bash /install/install_tvm_deps.sh

COPY install/install_opencv_deps.sh /install/install_opencv_deps.sh
RUN bash /install/install_opencv_deps.sh

# Install dependencies for arachne-runtime
RUN python3 -m pip install --upgrade pip --no-cache-dir \
    && python3 -m pip install --no-cache-dir \
    tensorflow==2.6.0 \
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
RUN ./scripts/install_tvm.sh gpu

ENTRYPOINT [ "pytest", "tests" ]
