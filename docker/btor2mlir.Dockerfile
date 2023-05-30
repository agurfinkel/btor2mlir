FROM seahorn/seahorn-llvm14:nightly

# ENV SEAHORN=/home/usea/seahorn/bin/sea PATH="$PATH:/home/usea/seahorn/bin:/home/usea/bin"

## install required pacakges
USER root

## Install latest cmake
RUN apt -y remove --purge cmake
RUN apt -y update
RUN apt -y install wget python3-pip
RUN python3 -m pip install --upgrade pip
RUN pip3 install cmake --upgrade
# RUN apt-get install libmlir-14-dev mlir-14-tools


# Assume that docker-build is ran in the top-level directory
WORKDIR /opt
COPY . btor2mlir

# Get btor2parser files
RUN mkdir -p /opt/btor2mlir/include/Target/Btor/btor2parser
RUN curl -O -L https://raw.githubusercontent.com/Boolector/btor2tools/master/src/btor2parser/btor2parser.c
RUN curl -O -L https://raw.githubusercontent.com/Boolector/btor2tools/master/src/btor2parser/btor2parser.h


RUN mkdir -p /opt/btor2mlir/debug 
WORKDIR /opt/btor2mlir/debug 

ARG BUILD_TYPE=RelWithDebInfo

# Build configuration
RUN cmake -G Ninja .. \
    -DCMAKE_C_COMPILER=clang-14 \
    -DCMAKE_CXX_COMPILER=clang++-14 \
    -DMLIR_DIR=/opt/llvm/run/lib/cmake/mlir \
    -DLLVM_DIR=/opt/llvm/run/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=BUILD_TYPE \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/run && \
    ninja && \
    ninja install