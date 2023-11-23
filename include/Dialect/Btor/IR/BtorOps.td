//===- BtorOps.td - Btor dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_OPS
#define BTOR_OPS

include "BtorBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "Dialect/Btor/IR/BtorTypes.td"
include "Dialect/Btor/IR/BtorAttributes.td"


//===----------------------------------------------------------------------===//
// Base btor operation definition.
//===----------------------------------------------------------------------===//

class Btor_Op<string mnemonic, list<Trait> traits = []> :
        Op<Btor_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// btor integer bit cast ops definitions
//===----------------------------------------------------------------------===//

// Base class for Btor Cast operations Requires a single operand and result. 
class Btor_CastOp<string mnemonic, TypeConstraint From, TypeConstraint To,
                   list<Trait> traits = []> :
    Btor_Op<mnemonic, traits # [SameOperandsAndResultShape]>,
    Arguments<(ins From:$in)>,
    Results<(outs To:$out)> {
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

def SignlessFixedWidthIntegerLike : TypeConstraint<Or<[
        AnySignlessInteger.predicate,
        VectorOf<[AnySignlessInteger]>.predicate,
        TensorOf<[AnySignlessInteger]>.predicate]>,
    "signless-fixed-width-integer-like">;

// Cast from an integer type to another integer type.
class BtorCastOp<string mnemonic> :
    Btor_CastOp<mnemonic, SignlessFixedWidthIntegerLike,
                          SignlessFixedWidthIntegerLike> {
  let builders = [
    OpBuilder<(ins "Value":$in, "Type":$type),
     [{
        build($_builder, $_state, type, in);
      }]>
    ];
}


def UExtOp : BtorCastOp<"uext"> {
  let summary = "integer zero extension operation";
  let description = [{
    The integer zero extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with zeros.

    Example:

    ```mlir
      %2 = btor.uext %1 : i3 to i6
    ```
  }];
  let verifier = [{ return verifyExtOp<IntegerType>(*this); }];
}

def SExtOp : BtorCastOp<"sext"> {
  let summary = "integer zero extension operation";
  let description = [{
    The integer zero extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with
    copies of the most-significant bit of the input.

    Example:

    ```mlir
      %2 = btor.sext %1 : i3 to i6
    ```
  }];
  let verifier = [{ return verifyExtOp<IntegerType>(*this); }];
}

//===----------------------------------------------------------------------===//
// btor ternary integer ops definitions
//===----------------------------------------------------------------------===//

def IteOp : Btor_Op<"ite", [NoSideEffect,
    AllTypesMatch<["true_value", "false_value", "result"]>]
    #ElementwiseMappable.traits> {
    
    let summary = "integer if-then-else operation";
    let description = [{
        This operation takes a condition and two integer 
        arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.ite %cond, %true, %false : i32
        ```
    }];

  let arguments = (ins BoolLike:$condition,
                      AnyType:$true_value,
                      AnyType:$false_value);
  let results = (outs AnyType:$result);
  
  let extraClassDeclaration = [{
    Value getCondition() { return condition(); }
    Value getTrueValue() { return true_value(); }
    Value getFalseValue() { return false_value(); }
  }];

  let parser = [{
    return parseIteOp(parser, result);
  }];

  let printer = [{
    return printIteOp(p, this);
  }];
}

def SliceOp : Btor_Op<"slice", [NoSideEffect,
    AllTypesMatch<["upper_bound", "lower_bound"]>]
    #ElementwiseMappable.traits> {
    
    let summary = "integer slice operation";
    let description = [{
        This operation takes an input of size N and two integer 
        arguments, U and L, and returns an integer of size:
                U - L + 1, where L <= U < N.

        Example:
        
        ```mlir
        %res = btor.slice %in, %upper, %lower : iN, i(upper-lower+1)
        ```
    }];

  let arguments = (ins SignlessIntegerLike:$in,
                      SignlessIntegerLike:$upper_bound,
                      SignlessIntegerLike:$lower_bound);
  let results = (outs SignlessIntegerLike:$result);

  let builders = [
    OpBuilder<(ins "Value":$in, 
                   "Value":$upper_bound, 
                   "Value":$lower_bound),
     [{
       $_state.addOperands({in, upper_bound, lower_bound});
       //$_state.addTypes(trueValue.getType());
      }]>
    ];

  let parser = [{
    return parseSliceOp(parser, result);
  }];
  
  let printer = [{
    return printSliceOp(p, this->getOperation());
  }];

  let verifier = [{ return verifySliceOp<IntegerType>(*this); }];
}

//===----------------------------------------------------------------------===//
// btor binary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for btor arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.
class BtorArithmeticOp<string mnemonic, list<Trait> traits = []> :
    Op<Btor_Dialect, mnemonic, traits>;

// This operation takes two operands and returns one result,
// each of these is required to be of the same type.
//  The custom assembly form of the operation is as follows
//
//     <op> %0, %1 : i32
class BtorBinaryOp<string mnemonic, list<Trait> traits = []> :
    BtorArithmeticOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins Btor_BitVec:$lhs, Btor_BitVec:$rhs)>,
    Results<(outs Btor_BitVec:$result)> {
        let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
    }

// This operation takes two operands or the same type
// and returns one result of type I1
//
//     <op> %0, %1 : i1
class BtorBinaryDifferentResultTypeOp<string mnemonic, list<Trait> traits = []> :
    BtorArithmeticOp<mnemonic, traits>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)> {

  let results = (outs BoolLike:$result);
  
  let parser = [{
    return parseBinaryOverflowOp(parser, result);
  }];

  let printer = [{
    return printBinaryOverflowOp(p, this->getOperation());
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

def SAddOverflowOp : BtorBinaryDifferentResultTypeOp<"saddo", [Commutative]> {
    let summary = "signed integer addition with overflow flag";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.saddo %lhs, %rhs : i32
        ```
    }];  
}

def UAddOverflowOp : BtorBinaryDifferentResultTypeOp<"uaddo", [Commutative]> {
    let summary = "unsigned integer addition with overflow flag";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.uaddo %lhs, %rhs : i32
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

def SSubOverflowOp : BtorBinaryDifferentResultTypeOp<"ssubo"> {
    let summary = "signed integer subtraction with overflow flag";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.ssubo %lhs, %rhs : i32
        ```
    }];  
}

def USubOverflowOp : BtorBinaryDifferentResultTypeOp<"usubo"> {
    let summary = "unsigned integer subtraction with overflow flag";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.usubo %lhs, %rhs : i32
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

def SMulOverflowOp : BtorBinaryDifferentResultTypeOp<"smulo"> {
    let summary = "signed integer multiplication with overflow flag";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.smulo %lhs, %rhs : i32
        ```
    }];  
}

def UMulOverflowOp : BtorBinaryDifferentResultTypeOp<"umulo"> {
    let summary = "unsigned integer multiplication with overflow flag";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.umulo %lhs, %rhs : i32
        ```
    }];  
}

def SModOp : BtorBinaryOp<"smod"> {
    let summary = "integer signed modulus operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.smod %lhs, %rhs : i32
        ```
    }];
}

def SRemOp : BtorBinaryOp<"srem"> {
    let summary = "integer signed remainder operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.srem %lhs, %rhs : i32
        ```
    }];
}

def URemOp : BtorBinaryOp<"urem"> {
    let summary = "integer unsigned remainder operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.urem %lhs, %rhs : i32
        ```
    }];
}

def SDivOp : BtorBinaryOp<"sdiv"> {
    let summary = "integer signed division operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.sdiv %lhs, %rhs : i32
        ```
    }];
}

def SDivOverflowOp : BtorBinaryDifferentResultTypeOp<"sdivo"> {
    let summary = "signed integer division with overflow flag";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.sdivo %lhs, %rhs : i32
        ```
    }];  
}

def UDivOp : BtorBinaryOp<"udiv"> {
    let summary = "integer unsigned division operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.udiv %lhs, %rhs : i32
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
      Btor_BitVec:$lhs,
      Btor_BitVec:$rhs
  );
  let results = (outs Btor_BitVec:$result);

  let builders = [
    OpBuilder<(ins "BtorPredicate":$predicate, "Value":$lhs,
                 "Value":$rhs), [{
      build($_builder, $_state, ::getI1SameShape(lhs.getType()),
            predicate, lhs, rhs);
    }]>];

  
  let extraClassDeclaration = [{
    static StringRef getPredicateAttrName() { return "predicate"; }
    static BtorPredicate getPredicateByName(StringRef name);

    BtorPredicate getPredicate() {
      return (BtorPredicate)(*this)->getAttrOfType<IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];

  let verifier = [{ return verifyCmpOp(*this); }];

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` qualified(type($lhs))";
} 

def IffOp : BtorBinaryOp<"iff"> {
    let summary = "integer if-and-only-if operation";
    let description = [{
        This operation takes two i1 arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.ror %lhs, %rhs : i1
        ```
    }];

    let arguments = (ins I1:$lhs, I1:$rhs);
}

def ImpliesOp : BtorBinaryOp<"implies"> {
    let summary = "integer implication operation";
    let description = [{
        This operation takes two i1 arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.ror %lhs, %rhs : i1
        ```
    }];
    
    let arguments = (ins I1:$lhs, I1:$rhs);
}

def ShiftLLOp : BtorBinaryOp<"sll"> {
    let summary = "integer left logical shift binary operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.sll %lhs, %rhs : i32
        ```
    }];
}

def ShiftRLOp : BtorBinaryOp<"srl"> {
    let summary = "integer right logical shift operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.srl %lhs, %rhs : i32
        ```
    }];
}

def ShiftRAOp : BtorBinaryOp<"sra"> {
    let summary = "integer right arithmetic shift operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.sra %lhs, %rhs : i32
        ```
    }];
}

def RotateLOp : BtorBinaryOp<"rol"> {
    let summary = "integer left rotate operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.rol %lhs, %rhs : i32
        ```
    }];
}

def RotateROp : BtorBinaryOp<"ror"> {
    let summary = "integer right rotate operation";
    let description = [{
        This operation takes two integer arguments and returns an integer.

        Example:
        
        ```mlir
        %res = btor.ror %lhs, %rhs : i32
        ```
    }];
}

def ConcatOp : Btor_Op<"concat"> {
    let summary = "integer concatenation operation";
    let description = [{
        This operation takes two integer arguments of length N and M
        and returns an integer of length N+M

        Example:
        
        ```mlir
        %res = btor.concat %lhs, %rhs : i3, i2, i5
        ```
    }];  

    let arguments = (ins SignlessIntegerLike:$lhs,
                      SignlessIntegerLike:$rhs);
    let results = (outs SignlessIntegerLike:$result);

    let parser = [{
      return parseConcatOp(parser, result);
    }];
    let printer = [{
      return printConcatOp(p, *this);
    }];
    let verifier = [{ return verifyConcatOp<IntegerType>(*this); }];
}

//===----------------------------------------------------------------------===//
// btor unary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for unary ops. Requires single operand and result. Individual
// classes will have `operand` accessor.
class BtorUnaryOp<string mnemonic, list<Trait> traits = []> :
    Op<Btor_Dialect, mnemonic, !listconcat(traits, [NoSideEffect,
        SameOperandsAndResultType])>,
    Arguments<(ins SignlessIntegerLike:$operand)>,
    Results<(outs SignlessIntegerLike:$result)> {
    
    let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// This operation takes an operand
// and returns one result of type I1
//
//     <op> %0 : i1
class BtorUnaryDifferentResultTypeOp<string mnemonic, list<Trait> traits = []> :
    Op<Btor_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])>,
    Arguments<(ins SignlessIntegerLike:$operand)> {

  let results = (outs BoolLike:$result);
   let parser = [{
    return parseUnaryDifferentResultOp(parser, result);
  }];

  let printer = [{
    return printBtorUnaryOp(p, this->getOperation());
  }];
}

def NotOp : BtorUnaryOp<"not"> {
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

def IncOp : BtorUnaryOp<"inc"> {
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

def DecOp : BtorUnaryOp<"dec"> {
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

def NegOp : BtorUnaryOp<"neg"> {
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

def RedAndOp : BtorUnaryDifferentResultTypeOp<"redand"> {
  let summary = "integer reduction with and operator";
  let description = [{
    Syntax:

    The `redand` operation computes the and reduction of a given value. It takes one
    operand and returns one result of type i1. 

    Example:

    ```mlir
    // Scalar redand value.
    %a = btor.redand %b : i1
    ```
  }];
}

def RedOrOp : BtorUnaryDifferentResultTypeOp<"redor"> {
  let summary = "integer reduction with or operator";
  let description = [{
    Syntax:

    This operation computes the or reduction of a given value. It takes one
    operand and returns one result of type i1. 

    Example:

    ```mlir
    // Scalar redand value.
    %a = btor.redand %b : i1
    ```
  }];
}

def RedXorOp : BtorUnaryDifferentResultTypeOp<"redxor"> {
  let summary = "integer reduction with and operator";
  let description = [{
    Syntax:

    This operation computes the xor reduction of a given value. It takes one
    operand and returns one result of type i1. 

    Example:

    ```mlir
    // Scalar redand value.
    %a = btor.redand %b : i1
    ```
  }];
}

def AssertNotOp : Btor_Op<"assert_not"> {
    let summary = "btor assertion to mimic bad operation";
    let description = [{
        This operation takes one boolean argument and terminates
        the program if the argument is false.

        Example:
        
        ```mlir
        %0 = constant 1 : i1
        // Apply the assert operation to %0
        btor.assert_not ( %0 )
        ```
    }];

    let arguments = (ins Btor_BitVec:$arg, AnyI64Attr:$id);

    let assemblyFormat = "`(` $arg `)` attr-dict `,` $id `:` qualified(type($arg))";
}

//, AllTypesMatch<["value", "result"]>
def ConstantOp : Btor_Op<"constant", [ConstantLike, NoSideEffect]> {
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

  let arguments = (ins BitVecAttribute:$value);
  let results = (outs Btor_BitVec:$result);

  let assemblyFormat = "attr-dict qualified(type($result)) qualified($value)";

  let verifier = [{ return verifyConstantOp(*this); }];
  
  let hasFolder = 1;
}

def ConstraintOp : Btor_Op<"constraint"> {
    let summary = "btor constraint";
    let description = [{
        This operation takes one boolean argument and assumes 
        it holds for the program.

        Example:
        
        ```mlir
        %0 = constant 1 : i1
        // Apply the constraint operation to %0
        btor.constraint ( %0 )
        ```
    }];

    let arguments = (ins I1:$constraint);

    let assemblyFormat = "`(` $constraint `)` attr-dict";
}

def NDStateOp : Btor_Op<"nd_state"> {
    let summary = "btor nondet bv state";
    let description = [{
        This operation takes no input and returns
        a SignlessIntegerLike

        Example:

        ```mlir
        // invoke the nd_bv operation 
        %0 = btor.nd_state 15 : i3
        ```
        In the example above, 15 gets interpreted as an
        i64 integer to represent the 15th input of a circuit
    }];

    let arguments = (ins AnyI64Attr:$id);
    let results = (outs Btor_BitVec:$result);
    let parser = [{
      return parseNDStateOpOp(parser, result);
    }];

    let printer = [{
      return printNDStateOpOp(p, *this);
    }];    
}

def InputOp : Btor_Op<"input"> {
    let summary = "btor input";
    let description = [{
        This operation takes an input number and a
        SignlessIntegerLike, then returns the 
        received SignlessIntegerLike

        Example:

        ```mlir
        // invoke the input operation to %0
        %0 = btor.input 15 : i3
        ```

        In the example above, 15 gets interpreted as an
        i64 integer to represent the line number of the input
    }];

    let arguments = (ins AnyI64Attr:$id);
    let results = (outs SignlessIntegerLike:$result);

    let parser = [{
      return parseInputOp(parser, result);
    }];

    let printer = [{
      return printInputOp(p, *this);
    }];
}

//===----------------------------------------------------------------------===//
// Btor arrays
//===----------------------------------------------------------------------===//

def ArrayOp : Btor_Op<"nd_array"> {
  let summary = "btor array allocation operation";
  let description = [{
    The `nd_array` operation represents a btor array from 
    bitvec -> bitvec. For example:

    ```mlir
    %0 = btor.nd_array : vector<8,i32>
    ```

    This operation returns a single SSA value of a btor array type used
    by subsequent read and write operations. 
  }];

  let results = (outs AnyVector:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
  let verifier = [{ return verifyArrayOp(*this); }];
  let extraClassDeclaration = [{
    VectorType getArrayType() {
      return result().getType().cast<VectorType>();
    }
  }];
}

def InitArrayOp : Btor_Op<"array"> {
  let summary = "btor initialized array allocation operation";
  let description = [{
    The `array` operation represents a btor array from 
    bitvec -> bitvec that has been initialized with a value. 

    For example:

    ```mlir
    %4 = btor.array %1 : vector<8,i32>
    ```

    This operation returns a single SSA value of a btor array type used
    by subsequent read and write operations. 
  }];

  let arguments = (ins SignlessIntegerLike:$init);
  let results = (outs AnyVector:$result);
  let extraClassDeclaration = [{
    VectorType getArrayType() {
      return result().getType().cast<VectorType>();
    }
  }];
  let verifier = [{ return verifyInitArrayOp(*this); }];
  let parser = [{
    return parseInitArrayOp(parser, result);
  }];
  let printer = [{
    return printInitArrayOp(p, *this);
  }];
}

def ReadOp : Btor_Op<"read"> {
  let summary = "btor read operation";
  let description = [{
    The `read` op reads an element from a btor array specified by an index. 
    The output of load is a new value with the same type as the elements of the
    btor array.

    Example:

    ```mlir
    %1 = btor.read %A[%0] : vector<8,i32>, i32
    ```
  }];

  let arguments = (ins AnyVector:$base, SignlessIntegerLike:$index);
  let results = (outs SignlessIntegerLike:$result);

  let extraClassDeclaration = [{
    VectorType getArrayType() {
      return base().getType().cast<VectorType>();
    }
  }];

  let verifier = [{ return verifyReadOp(*this); }];
  let parser = [{
    return parseReadOp(parser, result);
  }];
  let printer = [{
    return printReadOp(p, *this);
  }];
}

def WriteOp : Btor_Op<"write"> {
  let summary = "btor write operation";
  let description = [{
    Copy the given array and write the given value to the provided
    location given by index. The value stored should have the same type 
    as the elemental type of the array. 
    Example:

    ```mlir
    %4 = btor.write %2, %A[%1] : vector<8, i32>
    ```
  }];

  let arguments = (ins SignlessIntegerLike:$value, AnyVector:$base,
                       SignlessIntegerLike:$index);

  let results = (outs AnyVector:$result);
    let extraClassDeclaration = [{
    VectorType getArrayType() {
      return base().getType().cast<VectorType>();
    }
  }];

  let verifier = [{ return verifyWriteOp(*this); }];
  let parser = [{
    return parseWriteOp(parser, result);
  }];
  let printer = [{
    return printWriteOp(p, *this);
  }];
}

#endif // BTOR_OPS
