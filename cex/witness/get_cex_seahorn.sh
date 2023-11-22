#!/bin/bash

echo "btor2mlir-translate --import-btor $1 > $1.mlir" ; \
btor2mlir-translate --import-btor $1 > $1.mlir
echo "btor2mlir-translate --export-btor $1.mlir > $1.export.btor"
btor2mlir-translate --export-btor $1.mlir > $1.export.btor ; \
echo "btor2mlir-opt $1.mlir \
        --convert-btornd-to-llvm \
        --convert-btor-to-vector \
        --convert-arith-to-llvm \
        --convert-std-to-llvm \
        --convert-btor-to-llvm \
        --convert-vector-to-llvm > $1.mlir.opt" ; \
btor2mlir-opt $1.mlir \
        --convert-btornd-to-llvm \
        --convert-btor-to-vector \
        --convert-arith-to-llvm \
        --convert-std-to-llvm \
        --convert-btor-to-llvm \
        --convert-vector-to-llvm > $1.mlir.opt ; \
echo "btor2mlir-translate --mlir-to-llvmir $1.mlir.opt > $1.mlir.opt.ll"; \
btor2mlir-translate --mlir-to-llvmir $1.mlir.opt > $1.mlir.opt.ll ; \

# exe-cex
# --oll=$1.mlir.opt.ll.final.ll
echo "time timeout 300 sea yama -y configs/sea-cex.yaml bpf --verbose=2 -m64 $1.mlir.opt.ll -o$1.mlir.opt.ll.smt2"
time timeout 300 sea yama -y configs/sea-cex.yaml bpf --verbose=2 -m64 $1.mlir.opt.ll -o$1.mlir.opt.ll.smt2

echo "clang++-14 $1.mlir.opt.ll /tmp/h2.ll ../../build/run/lib/libcex.a -o h2.out"
clang++-14 $1.mlir.opt.ll /tmp/h2.ll ../../build/run/lib/libcex.a -o h2.out 

echo "./h2.out > /tmp/h2.txt"
env ./h2.out > /tmp/h2.txt

echo "python3 witness_generator.py /tmp/h2.txt"
python3 witness_generator.py /tmp/h2.txt

echo "btorsim -v $1 cex.txt"
btorsim -v $1 cex.txt
