//===- CrabOps.td - Crab dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRAB_OPS
#define CRAB_OPS

include "CrabDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


//===----------------------------------------------------------------------===//
// crab constant ops 
//===----------------------------------------------------------------------===//

def ConstantOp : Crab_Op<"const", [ConstantLike, NoSideEffect]> {
  let summary = "integer constant";
  let description = [{
    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute. 

    Example:

    ```mlir
    // Integer constant
    %1 = constant 42 : i32
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let builders = [
    OpBuilder<(ins "Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>,
    OpBuilder<(ins "Attribute":$value, "Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
  }];

  let parser = [{
    return parseConstantOp(parser, result);
  }];

  let printer = [{
    return printConstantOp(p, *this);
  }];

  let hasFolder = 1;
}

#endif // CRAB_OPS
