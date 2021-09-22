# An out of tree MLIR dialect

Based on [mlir example](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone). Adapted compilation instructions. Disabled tests.

## Building LLVM
Commands to configure and compile LLVM

```sh
$ mkdir debug && cd debug 
$ cmake -G Ninja ../llvm \
    -DCMAKE_C_COMPILER=clang-12 -DCMAKE_CXX_COMPILER=clang++-12 \
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON  \ 
    -DCMAKE_BUILD_TYPE=Debug \ # change to RelWithDebInfo for release build
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"  \ # maybe only X86 is needed
    -DLLVM_ENABLE_LLD=ON  \ # only on Linux	
    -DLLVM_INSTALL_UTILS=ON \ # optional to install FileCheck and lit
    -DCMAKE_INSTALL_PREFIX=$(pwd)/run  # install location in `run` under build
$ ninja
$ ninja install
```

The above installs a debug version of llvm under `LLVM_PROJECT/debug/run`, 
where `LLVM_PROJECT` is the root of llvm project on your machine.

## Building
To compile this project

```sh
$ mkdir debug && cd debug 
$ cmake -G Ninja .. \
    -DMLIR_DIR=/ag/llvm-gh-mlir/debug/run/lib/cmake/mlir \
    -DLLVM_DIR=/ag/llvm-gh-mlir/debug/run/lib/cmake/llvm \
    -DLLVM_EXTERNAL_LIT=$(which lit) \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/run \
```

The above assumes that `LLVM_PROJECT=/ag/llvm-gh-mlir`, and that `lit`
command is installed and is globally available



## Contributors
Arie Gurfinkel <arie.gurfinkel@uwaterloo.ca> \
Joseph Tafese <jetafese@uwaterloo.ca>
