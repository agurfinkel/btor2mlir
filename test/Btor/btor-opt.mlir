// RUN: btor2mlir-opt %s | btor2mlir-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 7 : i3
        // CHECK: %{{.*}} = constant {{.*}} : i3
        %1 = constant 3 : i3
        // CHECK: %{{.*}} = btor.add %{{.*}}, %{{.*}} : i3
        %2 = btor.add %0, %1 : i3
        // CHECK: %{{.*}} = btor.mul %{{.*}}, %{{.*}} : i3
        %3 = btor.mul %0, %2 : i3
        // CHECK: %{{.*}} = btor.eq %{{.*}} %{{.*}}
        %4 = btor.cmp "ne", %3, %2 : i3
        // CHECK: %{{.*}} = btor.bad %{{.*}}
        btor.bad %4
        return
    }
}
