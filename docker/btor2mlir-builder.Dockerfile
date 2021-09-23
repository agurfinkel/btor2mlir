# SeaHorn builder image that builds binary SeaHorn release package
# Primarily used by the CI
# Arguments:
#  - BASE-IMAGE: bionic-llvm10, focal-llvm10
#  - BUILD_TYPE: Debug, RelWithDebInfo, Coverage
ARG BASE_IMAGE=bionic-llvm10
FROM agurfinkel/buildback-deps-btor2mlir

# Assume that docker-build is ran in the top-level SeaHorn directory
WORKDIR /opt
COPY . btor2mlir
RUN mkdir -p /opt/btor2mlir/debug 
WORKDIR /opt/btor2mlir/debug 

ARG BUILD_TYPE=Debug

# Build configuration

RUN cmake -G Ninja .. \
    -DCMAKE_C_COMPILER=clang-10 \
    -DCMAKE_CXX_COMPILER=clang++-10 \
    -DMLIR_DIR=/opt/llvm/run/lib/cmake/mlir \
    -DLLVM_DIR=/opt/llvm/run/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/run && \
    ninja && \
    ninja check-btor2mlir	

