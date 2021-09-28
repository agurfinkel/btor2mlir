//===- BtorOps.td - Btor dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_OPS
#define BTOR_OPS

include "BtorDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AddOp : Btor_Op<"add", [NoSideEffect, SameOperandsAndResultType, Commutative]> {
    let summary = "integer addition operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %0 = constant 2 : i32
        %1 = constant 4 : i32
        // Apply the foo operation to %0
        %2 = btor.add %0 %1 : i32
        ```
    }];

    let arguments = (ins I32:$lhs, I32:$rhs);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $lhs $rhs attr-dict `:` type($lhs)
    }];
}

#endif // BTOR_OPS
