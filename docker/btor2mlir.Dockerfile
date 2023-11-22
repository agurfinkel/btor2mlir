FROM seahorn/seahorn-llvm14:nightly

## install required pacakges
USER root

## Install latest cmake
RUN apt -y remove --purge cmake
RUN apt -y update
RUN apt -y install wget python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install cmake --upgrade
RUN apt install -y libmlir-14-dev mlir-14-tools


# Assume that docker-build is ran in the top-level directory
WORKDIR /opt
COPY . btor2mlir

RUN mkdir -p /opt/btor2mlir/build
WORKDIR /opt/btor2mlir/build

ARG BUILD_TYPE=RelWithDebInfo

# Build configuration
RUN cmake -G Ninja .. \
    -DCMAKE_C_COMPILER=clang-14 \
    -DCMAKE_CXX_COMPILER=clang++-14 \
    -DMLIR_DIR=/usr/lib/llvm-14/lib/cmake/mlir \
    -DLLVM_DIR=/usr/lib/llvm-14/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=BUILD_TYPE \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/run && \
    ninja && \
    ninja install

RUN cp bin/* /usr/bin

# get btor2tools
WORKDIR /opt
RUN git clone https://github.com/Boolector/btor2tools.git
WORKDIR /opt/btor2tools
RUN ./configure.sh
WORKDIR /opt/btor2tools/build
RUN make
RUN cp bin/* /usr/bin

WORKDIR /opt/btor2mlir
