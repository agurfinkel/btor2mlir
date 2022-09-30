//===- CrabOps.td - Crab dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRAB_OPS
#define CRAB_OPS

include "CrabBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/InferIntRangeInterface.td"

//===----------------------------------------------------------------------===//
// Base crab operation definition.
//===----------------------------------------------------------------------===//

class Crab_Op<string mnemonic, list<Trait> traits = []> :
        Op<Crab_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// crab binary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for crab arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.
class CrabArithmeticOp<string mnemonic, list<Trait> traits = []> :
    Op<Crab_Dialect, mnemonic, traits #[NoSideEffect]
    #ElementwiseMappable.traits> { }

// This operation takes two operands and returns one result,
// each of these is required to be of the same type.
//  The custom assembly form of the operation is as follows
//
//     <op> %0, %1 : i32
class CrabBinaryOp<string mnemonic, list<Trait> traits = []> :
    CrabArithmeticOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)> {

  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";
}

def AddOp : CrabBinaryOp<"add", [Commutative]> {
    let summary = "integer addition operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.add(%lhs, %rhs) : i32
        ```
    }];
}

def SubOp : CrabBinaryOp<"sub"> {
    let summary = "integer subtraction operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.sub(%lhs, %rhs) : i32
        ```
    }];
}

def MulOp : CrabBinaryOp<"mul", [Commutative]> {
    let summary = "integer multiplication operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.mul(%lhs, %rhs) : i32
        ```
    }];
}

def SModOp : CrabBinaryOp<"smod"> {
    let summary = "integer signed modulus operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.smod(%lhs, %rhs) : i32
        ```
    }];
}

def SRemOp : CrabBinaryOp<"srem"> {
    let summary = "integer signed remainder operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.srem(%lhs, %rhs) : i32
        ```
    }];
}

def URemOp : CrabBinaryOp<"urem"> {
    let summary = "integer unsigned remainder operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.urem(%lhs, %rhs) : i32
        ```
    }];
}

def SDivOp : CrabBinaryOp<"sdiv"> {
    let summary = "integer signed division operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.sdiv(%lhs, %rhs) : i32
        ```
    }];
}

def UDivOp : CrabBinaryOp<"udiv"> {
    let summary = "integer unsigned division operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.udiv(%lhs, %rhs) : i32
        ```
    }];
}

def OrOp : CrabBinaryOp<"or", [Commutative]> {
    let summary = "integer binary and operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.or(%lhs, %rhs) : i32
        ```
    }];
}

def NorOp : CrabBinaryOp<"nor", [Commutative]> {
    let summary = "integer binary nor operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.nor(%lhs, %rhs) : i32
        ```
    }];
}

def AndOp : CrabBinaryOp<"and", [Commutative]> {
    let summary = "integer binary and operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.and(%lhs, %rhs) : i32
        ```
    }];
}

def NandOp : CrabBinaryOp<"nand", [Commutative]> {
    let summary = "integer binary nand operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.nand(%lhs, %rhs) : i32
        ```
    }];
}

def XOrOp : CrabBinaryOp<"xor", [Commutative]> {
    let summary = "integer binary xor operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.xor(%lhs, %rhs) : i32
        ```
    }];
}

def XnorOp : CrabBinaryOp<"xnor", [Commutative]> {
    let summary = "integer binary xnor operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = crab.xnor(%lhs, %rhs) : i32
        ```
    }];
}

//===----------------------------------------------------------------------===//
// crab constraint ops definitions
//===----------------------------------------------------------------------===//

// Base class for crab constraint operations.  Requires operands to be
// of the same type, but does not constrain them to specific
// types.
//  The custom assembly form of the operation is as follows
//
//     <op> "predicate"(%0, %1)
class CrabConstraintOp<string mnemonic, list<Trait> traits = [SameTypeOperands]> :
    Crab_Op<mnemonic, traits>,
    Arguments<(ins CrabPredicateAttr:$predicate, 
            SignlessIntegerLike:$lhs, 
            SignlessIntegerLike:$rhs)> {
    
    let description = [{
        The first argument is an attribute that defines which type of comparison is
        performed. The following comparisons are supported:

        -   equal (mnemonic: `"eq"`; integer value: `0`)
        -   not equal (mnemonic: `"ne"`; integer value: `1`)
        -   signed less than (mnemonic: `"slt"`; integer value: `2`)
        -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
        -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
        -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
        -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
        -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
        -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
        -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
    }];

    let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";

    let extraClassDeclaration = [{
        static StringRef getPredicateAttrName() { return "predicate"; }
        static CrabPredicate getPredicateByName(StringRef name);

        CrabPredicate getPredicate() {
        return (CrabPredicate)(*this)->getAttrOfType<IntegerAttr>(
            getPredicateAttrName()).getInt();
        }
    }];

    let assemblyFormat = "$predicate `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";
}

def AssertOp : CrabConstraintOp<"assert"> {
  let summary = "crab assertion";
  let description = [{
    The `assert` operation is an assertion tied to a  generic comparison
    for two arguments that need to have their types matching.

    Example:

    ```mlir
    // Custom form of scalar "signed less than" comparison.
    %x = crab.assert ne(%lhs, %rhs) : i32
    ```
  }];
} 

def AssumeOp : CrabConstraintOp<"assume"> {
  let summary = "crab assertion";
  let description = [{
    The `assume` operation is a constraint tied to a  generic comparison
    for two arguments that need to have their types matching.

    Example:

    ```mlir
    // Custom form of scalar "signed less than" comparison.
    %x = crab.assume sle(%lhs, %rhs) : i32
    ```
  }];
} 

//===----------------------------------------------------------------------===//
// crab constant ops 
//===----------------------------------------------------------------------===//

def ConstantOp : Crab_Op<"const", [ConstantLike, NoSideEffect,
    AllTypesMatch<["value", "result"]>]> {
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
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Attribute":$value, "Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let assemblyFormat = "attr-dict $value";
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
