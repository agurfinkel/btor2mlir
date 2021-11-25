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
// btor integer bit cast ops definitions
//===----------------------------------------------------------------------===//

// Base class for Btor Cast operations Requires a single operand and result. 
class Btor_CastOp<string mnemonic, TypeConstraint From, TypeConstraint To,
                   list<OpTrait> traits = []> :
    Btor_Op<mnemonic, traits # [SameOperandsAndResultShape]>,
    Arguments<(ins From:$in)>,
    Results<(outs To:$out)> {
  let builders = [
    OpBuilder<(ins "Value":$source, "Type":$destType), [{
      impl::buildCastOp($_builder, $_state, source, destType);
    }]>
  ];

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
                          SignlessFixedWidthIntegerLike>;


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
                      SignlessIntegerLike:$true_value,
                      SignlessIntegerLike:$false_value);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$condition, "Value":$trueValue, "Value":$falseValue),
     [{
       $_state.addOperands({condition, trueValue, falseValue});
       $_state.addTypes(trueValue.getType());
      }]>
    ];
  
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
  let results = (outs AnyType:$result);

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

  let verifier = [{ return verifySliceOp<IntegerType>(*this); }];
}

//===----------------------------------------------------------------------===//
// btor binary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for btor arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.
class BtorArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    Op<Btor_Dialect, mnemonic, traits #[NoSideEffect]
    #ElementwiseMappable.traits> { }

// This operation takes two operands and returns one result,
// each of these is required to be of the same type.
//  The custom assembly form of the operation is as follows
//
//     <op> %0, %1 : i32
class BtorBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    BtorArithmeticOp<mnemonic, !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)> {

  let results = (outs AnyType:$result);

  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
    return printBtorBinaryOp(this->getOperation(), p);
  }];
}

// This operation takes two operands or the same type
// and returns one result of type I1
//
//     <op> %0, %1 : i1
class BtorBinaryDifferentResultTypeOp<string mnemonic, list<OpTrait> traits = []> :
    BtorArithmeticOp<mnemonic, traits>,
    Arguments<(ins SignlessIntegerLike:$lhs, SignlessIntegerLike:$rhs)> {

  let results = (outs BoolLike:$result);

  let parser = [{
    return parseBinaryOverflowOp(parser, result);
  }];

  let printer = [{
    return printBtorBinaryDifferentResultTypeOp(this->getOperation(), p);
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
        and eturns an integer of length N+M

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

    let verifier = [{ return verifyConcatOp<IntegerType>(*this); }];
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
    return parseUnaryOp(parser, result);
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

  let results = (outs BoolLike);
}

def RedOrOp : BtorUnaryOp<"redor"> {
  let summary = "integer reduction with or operator";
  let description = [{
    Syntax:

    This operation computes the or reduction of a given value. It takes one
    operand and returns one result of type i1. 

    Example:

    ```mlir
    // Scalar redand value.
    %a = btor.redand %b : i32
    ```
  }];

  let results = (outs BoolLike);
}

def RedXorOp : BtorUnaryOp<"redxor"> {
  let summary = "integer reduction with and operator";
  let description = [{
    Syntax:

    This operation computes the xor reduction of a given value. It takes one
    operand and returns one result of type i1. 

    Example:

    ```mlir
    // Scalar redand value.
    %a = btor.redand %b : i32
    ```
  }];

  let results = (outs BoolLike);
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

    let arguments = (ins I1:$arg);

    let assemblyFormat = "`(` $arg `)` attr-dict";
}

def ConstantOp : Btor_Op<"const", [ConstantLike, NoSideEffect]> {
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

def AssumeOp : Btor_Op<"assume"> {
    let summary = "btor assumption";
    let description = [{
        This operation takes one boolean argument and assumes 
        it holds for the program.

        Example:
        
        ```mlir
        %0 = constant 1 : i1
        // Apply the assume operation to %0
        btor.assume ( %0 )
        ```
    }];

    let arguments = (ins I1:$arg);

    let assemblyFormat = "`(` $arg `)` attr-dict";
}

def UndefOp : Btor_Op<"undef"> {
    let summary = "btor undefined operation";
    let description = [{
        This operation takes no input and returns
        a SignlessIntegerLike

        Example:

        ```mlir
        // invoke the undef operation 
        %0 = btor.undef : i3
        ```
    }];

    let results = (outs SignlessIntegerLike:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
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
        %0 = btor.input 15, %some_i3_value : i3
        ```

        In the example above, 15 gets interpreted as an
        i64 integer, while %some_i3_value is expected 
        to be of type i3. 
    }];

    let arguments = (ins AnyI64Attr:$id, SignlessIntegerLike:$value);
    let results = (outs SignlessIntegerLike:$result);

    let builders = [
        OpBuilder<(ins "Value":$value),
        [{
            $_state.addOperands({value});
            $_state.addTypes(value.getType());
        }]>
    ];

    let printer = [{
      return printInputOp(p, *this);
    }];

    let parser = [{
      return parseInputOp(parser, result);
    }];
}

#endif // BTOR_OPS
