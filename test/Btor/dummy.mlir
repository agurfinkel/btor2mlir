// RUN: btor2mlir-opt %s | btor2mlir-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = addi %{{.*}}, %{{.*}} : i32
        %res = addi %0, %0 : i32
        return
    }
}
