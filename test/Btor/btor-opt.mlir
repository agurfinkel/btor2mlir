// RUN: btor-opt %s | btor-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = btor.add %{{.*}} %{{.*}} : i32
        %1 = constant 42 : i32
        // CHECK: %{{.*}} = btor.add %{{.*}} %{{.*}} : i32
        %2 = btor.add %0 %1: i32
        return
    }
}