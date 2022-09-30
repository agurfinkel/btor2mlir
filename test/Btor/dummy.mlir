// RUN: btor2mlir-opt %s | btor2mlir-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = btor.constant 1 : i32
        %1 = btor.constant 2 : i3
        // CHECK: %{{.*}} = addi %{{.*}}, %{{.*}} : i32
        %res = btor.not %0 : i32
        %t = btor.concat %0, %1 : i32, i3, i35
        %05 = btor.constant 15 : i35
        %slice = btor.slice %05, %t, %t : i35, i1
        %ext = btor.uext %res : i32 to i35
        %cmp = btor.cmp "eq", %t, %ext : i35
        %ite = btor.ite %cmp, %t, %05 : i35
        // %res = btor.cmp "eq", %0, %0 : i32
        return
    }
}
