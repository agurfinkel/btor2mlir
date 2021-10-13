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

//===----------------------------------------------------------------------===//
// btor binary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for btor arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.
class BtorArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Btor_Dialect, mnemonic, traits 
    #[NoSideEffect, SameOperandsAndResultType] #ElementwiseMappable.traits> {

  let results = (outs AnyType:$result);

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printBtorBinaryOp(this->getOperation(), p);
  }];
}

// This operation takes two operands and returns one result,
// each of these is required to be of the same type.
//  The custom assembly form of the operation is as follows
//
//     <op>i %0, %1 : i32
class BtorBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    BtorArithmeticOp<mnemonic, traits>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)> {

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printBtorBinaryOp(this->getOperation(), p);
  }];
}

def AddOp : BtorBinaryOp<"add", [Commutative]> {
    let summary = "integer addition operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.add %lhs, %rhs : i32
        ```
    }];
}

def SubOp : BtorBinaryOp<"sub"> {
    let summary = "integer subtraction operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.sub %lhs, %rhs : i32
        ```
    }];
}

def MulOp : BtorBinaryOp<"mul", [Commutative]> {
    let summary = "integer multiplication operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.mul %lhs, %rhs : i32
        ```
    }];
}

def OrOp : BtorBinaryOp<"or", [Commutative]> {
    let summary = "integer binary and operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.or %lhs, %rhs : i32
        ```
    }];
}

def NorOp : BtorBinaryOp<"nor", [Commutative]> {
    let summary = "integer binary nor operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.nor %lhs, %rhs : i32
        ```
    }];
}

def AndOp : BtorBinaryOp<"and", [Commutative]> {
    let summary = "integer binary and operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.and %lhs, %rhs : i32
        ```
    }];
}

def NandOp : BtorBinaryOp<"nand", [Commutative]> {
    let summary = "integer binary nand operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.nand %lhs, %rhs : i32
        ```
    }];
}

def XOrOp : BtorBinaryOp<"xor", [Commutative]> {
    let summary = "integer binary xor operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.xor %lhs, %rhs : i32
        ```
    }];
}

def XnorOp : BtorBinaryOp<"xnor", [Commutative]> {
    let summary = "integer binary xnor operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.xnor %lhs, %rhs : i32
        ```
    }];
}

def CmpOp : Btor_Op<"cmp", [NoSideEffect, SameTypeOperands,
     TypesMatchWith<"result type has i1 element type and same shape as operands",
    "lhs", "result", "getI1SameShape($_self)">] # ElementwiseMappable.traits> {
  let summary = "integer comparison operation";
  let description = [{
    The `cmp` operation is a generic comparison for two arguments 
    that need to have their types matching.

    Its first argument is an attribute that defines which type of comparison is
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

    The result is `1` if the comparison is true and `0` otherwise.

    Example:

    ```mlir
    // Custom form of scalar "signed less than" comparison.
    %x = btor.cmp "slt", %lhs, %rhs : i32
    ```
  }];

  let arguments = (ins
      BtorPredicateAttr:$predicate,
      SignlessIntegerLike:$lhs,
      SignlessIntegerLike:$rhs
  );
  let results = (outs BoolLike:$result);

  let builders = [
    OpBuilder<(ins "BtorPredicate":$predicate, "Value":$lhs,
                 "Value":$rhs), [{
      ::buildCmpOp($_builder, $_state, predicate, lhs, rhs);
    }]>];

  
  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static BtorPredicate getPredicateByName(StringRef name);

    BtorPredicate getPredicate() {
      return (BtorPredicate)(*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];

  let verifier = [{ return success(); }];

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
} 

def SRemOp : BtorBinaryOp<"srem", [Commutative]> {
    let summary = "integer signed remainder operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.srem %lhs, %rhs : i32
        ```
    }];
}

def URemOp : BtorBinaryOp<"urem", [Commutative]> {
    let summary = "integer unsigned remainder operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.urem %lhs, %rhs : i32
        ```
    }];
}

def SDivOp : BtorBinaryOp<"sdiv", [Commutative]> {
    let summary = "integer signed division operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.sdiv %lhs, %rhs : i32
        ```
    }];
}

def UDivOp : BtorBinaryOp<"udiv", [Commutative]> {
    let summary = "integer unsigned division operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.udiv %lhs, %rhs : i32
        ```
    }];
}

def ShiftLLOp : BtorBinaryOp<"sll", [Commutative]> {
    let summary = "integer left logical shift binary operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.sll %lhs, %rhs : i32
        ```
    }];
}

def ShiftRLOp : BtorBinaryOp<"srl", [Commutative]> {
    let summary = "integer right logical shift operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.srl %lhs, %rhs : i32
        ```
    }];
}

def ShiftRAOp : BtorBinaryOp<"sra", [Commutative]> {
    let summary = "integer right arithmetic shift operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.sra %lhs, %rhs : i32
        ```
    }];
}

//===----------------------------------------------------------------------===//
// btor unary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for unary ops. Requires single operand and result. Individual
// classes will have `operand` accessor.
class BtorUnaryOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Btor_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])>,
    Arguments<(ins SignlessIntegerLike:$operand)> {

  let results = (outs AnyType);

  let printer = [{
    return printBtorUnaryOp(this->getOperation(), p);
  }];

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];
}

def NotOp : BtorUnaryOp<"not", [SameOperandsAndResultType]> {
  let summary = "integer negation";
  let description = [{
    Syntax:

    The `not` operation computes the negation of a given value. It takes one
    operand and returns one result of the same type. 

    Example:

    ```mlir
    // Scalar negation value.
    %a = btor.not %b : i32
    ```
  }];
}

def IncOp : BtorUnaryOp<"inc", [SameOperandsAndResultType]> {
  let summary = "integer increment by one";
  let description = [{
    Syntax:

    The `inc` operation increments the given value by one. It takes one
    operand and returns one result of the same type. 

    Example:

    ```mlir
    // Scalar increment value.
    %a = btor.inc %b : i32
    ```
  }];
}

def DecOp : BtorUnaryOp<"dec", [SameOperandsAndResultType]> {
  let summary = "integer decrement by one";
  let description = [{
    Syntax:

    The `dec` operation decrements the given value by one. It takes one
    operand and returns one result of the same type. 

    Example:

    ```mlir
    // Scalar decrement value.
    %a = btor.dec %b : i32
    ```
  }];
}

def NegOp : BtorUnaryOp<"neg", [SameOperandsAndResultType]> {
  let summary = "integer negation";
  let description = [{
    Syntax:

    The `neg` operation flips the sign of a given value. It takes one
    operand and returns one result of the same type. 

    Example:

    ```mlir
    // Scalar decrement value.
    %a = btor.neg %b : i32
    ```
  }];
}

def RedAndOp : BtorUnaryOp<"redand"> {
  let summary = "integer reduction with and operator";
  let description = [{
    Syntax:

    The `redand` operation computes the and reduction of a given value. It takes one
    operand and returns one result of type i1. 

    Example:

    ```mlir
    // Scalar redand value.
    %a = btor.redand %b : i32
    ```
  }];
}

def BadOp : Btor_Op<"bad"> {
    let summary = "btor assertion";
    let description = [{
        This operation takes one boolean argument and terminates
        the program if the argument is false.

        Example:
        
        ```mlir
        %0 = constant 1 : i1
        // Apply the bad operation to %0
        btor.bad %0
        ```
    }];

    let arguments = (ins I1:$arg);

    let assemblyFormat = "$arg attr-dict";
}

#endif // BTOR_OPS
