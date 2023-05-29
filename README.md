# Bᴛᴏʀ2ᴍʟɪʀ: A Format for Hardware Verification
![os](https://img.shields.io/badge/os-linux-orange?logo=linux)
![os](https://img.shields.io/badge/os-macos-silver?logo=apple)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][data]

## Results
Detailed analysis of run-times is available in an accompanying [Jupyter
Notebook][data] in Google Collab.

[data]: https://colab.research.google.com/drive/1wau9yTHvsWdBdMjF0TbvHTHEKW6rFHiQ?usp=sharing

## Demo

Consider a simple counter represented in Bᴛᴏʀ2 below:

```btor
1 sort bitvec 4
2 zero 1
3 state 1 out
4 init 1 3 2
5 one 1
6 add 1 3 5
7 next 1 3 6
8 ones 1
9 sort bitvec 1
10 eq 9 3 8
11 bad 10
```

Using the command `build/bin/btor2mlir-translate --import-btor counter.btor2 > counter.mlir`, where `counter.btor2` is the file shown above, we get the equivalent representation of our circuit in the BTOR Dialect of MLIR below (counter.mlir):

```mlir
module {
  func @main() {
    %0 = btor.constant 0 : i4
    br ^bb1(%0 : i4)
  ^bb1(%1: i4):  // 2 preds: ^bb0, ^bb1
    %2 = btor.constant 1 : i4
    %3 = btor.add %1, %2 : i4
    %4 = btor.constant -1 : i4
    %5 = btor.cmp eq, %1, %4 : i4
    btor.assert_not(%5)
    br ^bb1(%3 : i4)
  }
}
```

Then, using the command  `build/bin/btor2mlir-opt  --convert-std-to-llvm --convert-btor-to-llvm counter.mlir > counter.mlir.opt` we get the file below which represents the original circuit in the LLVM Dialect of MLIR. 

```mlir
module attributes {llvm.data_layout = ""} {
  llvm.func @__VERIFIER_error()
  llvm.func @main() {
    %0 = llvm.mlir.constant(0 : i4) : i4
    llvm.br ^bb1(%0 : i4)
  ^bb1(%1: i4):  // 2 preds: ^bb0, ^bb2
    %2 = llvm.mlir.constant(1 : i4) : i4
    %3 = llvm.add %1, %2  : i4
    %4 = llvm.mlir.constant(-1 : i4) : i4
    %5 = llvm.icmp "eq" %1, %4 : i4
    %6 = llvm.mlir.constant(true) : i1
    %7 = llvm.xor %5, %6  : i1
    llvm.cond_br %7, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.br ^bb1(%3 : i4)
  ^bb3:  // pred: ^bb1
    llvm.call @__VERIFIER_error() : () -> ()
    llvm.unreachable
  }
}
```

Finally, using the command `build/bin/btor2mlir-translate --mlir-to-llvmir counter.mlir.opt > counter.ll` we generate the circuit as an LLVM-IR program below (counter.ll): 

```llvm
declare void @__VERIFIER_error()
define void @main() !dbg !3 {
  br label %1, !dbg !7
1:                                                ; preds = %6, %0
  %2 = phi i4 [ %3, %6 ], [ 0, %0 ]
  %3 = add i4 %2, 1, !dbg !9
  %4 = icmp eq i4 %2, -1, !dbg !10
  %5 = xor i1 %4, true, !dbg !11
  br i1 %5, label %6, label %7, !dbg !12
6:                                                ; preds = %1
  br label %1, !dbg !13
7:                                                ; preds = %1
  call void @__VERIFIER_error(), !dbg !14
  unreachable, !dbg !15
}
```
## Docker

Dockerfile: [`docker/btor2mlir.Dockerfile`](docker/btor2mlir-builder.Dockerfile).

## Building Locally

The instructions assume that `cmake`, `clang/clang++` and `ninja` are installed on your machine,  `LLVM_PROJECT=/ag/llvm-gh-mlir`, and that `lit`
command is installed and is globally available

### Building LLVM
Commands to configure and compile LLVM

```sh
$ mkdir debug && cd debug 
$ cmake -G Ninja ../llvm \
    -DCMAKE_C_COMPILER=clang-12 -DCMAKE_CXX_COMPILER=clang++-12 \
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON  \ 
    -DCMAKE_BUILD_TYPE=Debug \ # change to RelWithDebInfo for release build
    -DLLVM_TARGETS_TO_BUILD="X86"  \
    -DLLVM_ENABLE_LLD=ON  \ # only on Linux	
    -DLLVM_INSTALL_UTILS=ON \ # optional to install FileCheck and lit
    -DCMAKE_INSTALL_PREFIX=$(pwd)/run  # install location in `run` under build
$ ninja
$ ninja install
```

The above installs a debug version of llvm under `LLVM_PROJECT/debug/run`, 
where `LLVM_PROJECT` is the root of llvm project on your machine.

### Building
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


## Contributors
Arie Gurfinkel <arie.gurfinkel@uwaterloo.ca> \
Joseph Tafese <jetafese@uwaterloo.ca>
