// RUN: btor2mlir-opt %s | btor2mlir-opt | FileCheck %s

// can we generate integer and index types when dealing with arrays? 
//  Keep track of 2 vars so that we don't lose out?

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %t = btor.constant 1 : i35
        %array = btor.array %t : !btor.array<i35,i35>
        %a2 = btor.read %array [%t] : !btor.array<i35,i35>, i35
        %slice = btor.slice %a2, %t, %t : i35, i1
        %ext = btor.sext %slice : i1 to i35
        %wrt = btor.write %ext, %array [%t] : i35, !btor.array<i35, i35>
        %cmp = btor.cmp "eq", %t, %ext : i35
        return
    }
}
