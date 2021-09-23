# Builder image that builds binary packages
# Primarily used by the CI
# Arguments:
#  - BASE-IMAGE: bionic-llvm10, focal-llvm10
#  - BUILD_TYPE: Debug, RelWithDebInfo, Coverage
ARG BASE_IMAGE=bionic-llvm10
FROM seahorn/buildpack-deps-seahorn:$BASE_IMAGE

# Install latest cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt -y update && \
    apt -y install cmake

# Install LLVM
RUN mkdir -p /opt cd opt && \
    git clone --depth 1 --branch main https://github.com/llvm/llvm-project.git && \
    cd llvm-project && mkdir debug && cd debug && \
    cmake -G Ninja ../llvm \
        -DCMAKE_C_COMPILER=clang-10 -DCMAKE_CXX_COMPILER=clang++-10 \
        -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON  \ 
        -DCMAKE_BUILD_TYPE=Debug \ 
        -DLLVM_TARGETS_TO_BUILD="X86"  \ 
        -DLLVM_ENABLE_LLD=ON  \ 
        -DLLVM_INSTALL_UTILS=ON \ 
        -DCMAKE_INSTALL_PREFIX=/opt/llvm/run && \
     ninja && \ 
     ninja install 
     
WORKDIR /opt
