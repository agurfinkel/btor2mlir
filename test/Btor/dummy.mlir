// RUN: btor2mlir-opt %s | btor2mlir-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = btor.const 5 : i32
        %i = arith.index_cast %0 : i32 to index
        // CHECK: %{{.*}} = addi %{{.*}}, %{{.*}} : i32
        %1 = btor.array : memref<8xi32>
        %2 = btor.read %1[%i] : memref<8xi32> to i32
        %3 = btor.write %2, %1[%i] : memref<8xi32> to memref<8xi32>
        %res = btor.redand %0 : i32
        %4 = btor.read %1[%i] : memref<8xi32> to i32
        btor.assume( %res )
        %res1 = btor.redxor %0 : i32
        return
    }
}
