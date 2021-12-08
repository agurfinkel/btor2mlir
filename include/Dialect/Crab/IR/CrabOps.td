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

//===----------------------------------------------------------------------===//
// crab havoc ops 
//===----------------------------------------------------------------------===//

def HavocOp : Crab_Op<"havoc"> {
    let summary = "crab havoc";
    let description = [{
        This operation takes no input and returns a SignlessIntegerLike

        Example:

        ```mlir
        // invoke the havoc operation
        %0 = crab.havoc() : i32
        ```
    }];

    let results = (outs SignlessIntegerLike:$result);

    let assemblyFormat = "`(` `)` attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// crab nd_br ops 
//===----------------------------------------------------------------------===//

def NDBranchOp : Crab_Op<"nd_br", [AttrSizedOperandSegments, NoSideEffect, Terminator]> {
  let summary = "nondeterministic branch operation";
  let description = [{
    The `nd_br` terminator operation represents a nondeterministic branch. We 
    nondeterministically select the first destination or the second 
    destination

    This branch operation is not allowed to target the entry block for a region. 
    The two destinations of the conditional branch operation are allowed to be the same.

    Example:

    ```mlir
    func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
      // Both targets are the same, operands differ
      nd_br ^bb1(%a : i32), ^bb1(%b : i32)

    ^bb1(%x : i32) :
      return %x : i32
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$lhsDestOperands,
                    Variadic<AnyType>:$rhsDestOperands);
  let successors = (successor AnySuccessor:$lhsDest, AnySuccessor:$rhsDest);

  let builders = [
    OpBuilder<(ins "Block *":$lhsDest, "ValueRange":$lhsOperands, 
        "Block *":$rhsDest, "ValueRange":$rhsOperands), 
    [{
        build($_builder, $_state, lhsOperands, rhsOperands, lhsDest, rhsDest);
    }]>];

  let assemblyFormat = [{
    $lhsDest (`(` $lhsDestOperands^ `:` type($lhsDestOperands) `)`)? `,`
    $rhsDest (`(` $rhsDestOperands^ `:` type($rhsDestOperands) `)`)?
    attr-dict
  }];
}

#endif // CRAB_OPS
