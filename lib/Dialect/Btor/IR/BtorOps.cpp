//===- BtorOps.cpp - Btor dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <utility>

#include "Dialect/Btor/IR/Btor.h"
#include "Dialect/Btor/IR/BtorTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallString.h"

#include "llvm/ADT/APSInt.h"

using namespace mlir;
using namespace mlir::btor;


/// A custom unary operation printer that omits the "std." prefix from the
/// operation names.
static void printBtorUnaryOp(OpAsmPrinter &p, Operation *op) {
  assert(op->getNumOperands() == 1 && "unary op should have one operand");
  assert(op->getNumResults() == 1 && "unary op should have one result");

  p << ' ' << op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0).getType();
}

/// A custom unary operation parser that ensures result has type i1
static ParseResult parseUnaryDifferentResultOp(OpAsmParser &parser,
                                  OperationState &result) {  
  Type operandType;
  SmallVector<OpAsmParser::OperandType, 1> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/1) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(operandType))
    return failure();
  
  result.addTypes(parser.getBuilder().getI1Type());
  return parser.resolveOperands(operands,
                                {operandType},
                                parser.getNameLoc(), result.operands);
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = btor::BitVecType::get(type.getContext(), 1);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

static ParseResult parseSliceOp(OpAsmParser &parser, OperationState &result) {
  Type resultType, operandType;
  SmallVector<OpAsmParser::OperandType, 3> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(operandType) || parser.parseOptionalComma() || 
      parser.parseType(resultType))
    return failure();

  result.addTypes(resultType);
  return parser.resolveOperands(operands,
                                {operandType, operandType, operandType},
                                parser.getNameLoc(), result.operands);
}

/// A custom slice operation printer
static void printSliceOp(OpAsmPrinter &p, Operation *op) {
  assert(op->getNumOperands() == 3 && "slice op should have one operand");
  assert(op->getNumResults() == 1 && "slice op should have one result");

  p << ' ' << op->getOperand(0) << ", " << op->getOperand(1) << ", " << op->getOperand(2);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0).getType() << ", " << op->getResult(0).getType();
}

template <typename ValType, typename Op>
static LogicalResult verifySliceOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.in().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() < dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be smaller or equal to the operand type " << srcType;

  return success();
}

//===----------------------------------------------------------------------===//
// IteOp
//===----------------------------------------------------------------------===//

static void printIteOp(OpAsmPrinter &p, IteOp *op) {
  p << " " << op->getOperands();
  p << " : ";
  if (ShapedType condType = op->getCondition().getType().dyn_cast<ShapedType>())
    p << condType << ", ";
  p << op->getType();
}

static ParseResult parseIteOp(OpAsmParser &parser, OperationState &result) {
  Type conditionType, resultType;
  SmallVector<OpAsmParser::OperandType, 3> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType))
    return failure();

  // Check for the explicit condition type if this is a masked tensor or vector.
  if (succeeded(parser.parseOptionalComma())) {
    conditionType = resultType;
    if (parser.parseType(resultType))
      return failure();
  } else {
    conditionType = parser.getBuilder().getI1Type();
  }

  result.addTypes(resultType);
  return parser.resolveOperands(operands,
                                {conditionType, resultType, resultType},
                                parser.getNameLoc(), result.operands);
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return value();
}

//===----------------------------------------------------------------------===//
// Overflow Operations
//===----------------------------------------------------------------------===//

static ParseResult parseBinaryOverflowOp(OpAsmParser &parser,
                                  OperationState &result) {  
  Type operandType;
  SmallVector<OpAsmParser::OperandType, 2> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(operandType))
    return failure();
  
  result.addTypes(parser.getBuilder().getI1Type());
  return parser.resolveOperands(operands,
                                {operandType, operandType},
                                parser.getNameLoc(), result.operands);
}

static void printBinaryOverflowOp(OpAsmPrinter &p, Operation *op) {
  assert(op->getNumOperands() == 2 && "binary op should have two operands");
  assert(op->getNumResults() == 1 && "binary op should have one result");

  p << ' ' << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());

  // Now we can output only one type for all operands and the result.
  p << " : " << op->getResult(0).getType();
}

//===----------------------------------------------------------------------===//
// Extension Operations
//===----------------------------------------------------------------------===//

template <typename ValType, typename Op>
static LogicalResult verifyExtOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.in().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() > dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

static ParseResult parseConcatOp(OpAsmParser &parser, OperationState &result) {
  Type resultType, firstOperandType, secondOperandType;
  SmallVector<OpAsmParser::OperandType, 1> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(firstOperandType) || parser.parseOptionalComma() || 
      parser.parseType(secondOperandType) || parser.parseOptionalComma() || 
      parser.parseType(resultType))
    return failure();

  result.addTypes(resultType);
  return parser.resolveOperands(operands,
                                {firstOperandType, secondOperandType},
                                parser.getNameLoc(), result.operands);
}

static void printConcatOp(OpAsmPrinter &p, Operation *op) {
  assert(op->getNumOperands() == 2 && "concat op should have two operands");

  p << ' ' << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());

  // Now we can output the types for all operands and the result.
  p << " : " << op->getOperand(0).getType() << ", " 
    << op->getOperand(1).getType() << ", " << op->getResult(0).getType();
}

template <typename ValType, typename Op>
static LogicalResult verifyConcatOp(Op op) {
  Type firstType = getElementTypeOrSelf(op.lhs().getType());
  Type secondType = getElementTypeOrSelf(op.rhs().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  auto sumOfTypes = firstType.cast<ValType>().getWidth() + 
              secondType.cast<ValType>().getWidth();
  if (sumOfTypes != dstType.cast<ValType>().getWidth())
    return op.emitError("sum of ") << firstType << " and "
         << secondType << " must be equal to operand type " << dstType;

  return success();
}

//===----------------------------------------------------------------------===//
// Input Operation
//===----------------------------------------------------------------------===//

static void printInputOp(OpAsmPrinter &p, mlir::btor::InputOp &op) {
  p << " " << op.id();
  p << " : " << op.result().getType();
}

static ParseResult parseInputOp(OpAsmParser &parser, OperationState &result) {  
  SmallVector<OpAsmParser::OperandType> ops;
  NamedAttrList attrs;
  Attribute idAttr;
  Type type;

  Type i64Type = parser.getBuilder().getIntegerType(64);

  if (parser.parseAttribute(idAttr, i64Type, "id", attrs) ||
      parser.parseOptionalAttrDict(attrs) || 
      parser.parseColonType(type))
      return failure();

  if (!idAttr.isa<mlir::IntegerAttr>())
      return parser.emitError(parser.getNameLoc(),
                              "expected integer id attribute");

  result.attributes = attrs;
  result.addTypes({type});
  return success();
}

//===----------------------------------------------------------------------===//
// NDStateOp Operation
//===----------------------------------------------------------------------===//

static void printNDStateOpOp(OpAsmPrinter &p, mlir::btor::NDStateOp &op) {
  p << " " << op.id();
  p << " : " << op.result().getType();
}

static ParseResult parseNDStateOpOp(OpAsmParser &parser, OperationState &result) {  
  SmallVector<OpAsmParser::OperandType> ops;
  NamedAttrList attrs;
  Attribute idAttr;
  Type type;

  Type i64Type = parser.getBuilder().getIntegerType(64);

  if (parser.parseAttribute(idAttr, i64Type, "id", attrs) ||
      parser.parseOptionalAttrDict(attrs) || 
      parser.parseColonType(type))
      return failure();

  if (!idAttr.isa<mlir::IntegerAttr>())
      return parser.emitError(parser.getNameLoc(),
                              "expected integer id attribute");

  result.attributes = attrs;
  result.addTypes({type});
  return success();
}

//===----------------------------------------------------------------------===//
// Array Operations
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult verifyArrayOp(Op op) {
  if (op.getArrayType().getShape().size() != 1) {
    return op.emitOpError() << "provide only one shape attribute ";
  }
  auto shape = op.getArrayType().getShape()[0];
  auto indicator = shape & (shape - 1);
  if (indicator != 0) {
    return op.emitOpError() << "given shape: " <<  shape 
                         << " has to be a power of two";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Initialzied Array Operations
//===----------------------------------------------------------------------===//

static void printInitArrayOp(OpAsmPrinter &p, InitArrayOp &op) {
  p << " " << op.init();
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op.result().getType();
}

template <typename Op>
static LogicalResult verifyInitArrayOp(Op op) {
  auto type = op.init().getType().getIntOrFloatBitWidth();
  // The value's type must match the array's element type.
  auto elementType = op.getArrayType().getElementType();
  if (elementType.getIntOrFloatBitWidth() != type) {
    return op.emitOpError() << "element type of the array must match "
                         << " bitwidth of given value: " << type;
  }
  if (op.getArrayType().getShape().size() != 1) {
    return op.emitOpError() << "provide only one shape attribute ";
  }
  auto shape = op.getArrayType().getShape()[0];
  auto indicator = shape & (shape - 1);
  if (indicator != 0) {
    return op.emitOpError() << "given shape: " <<  shape 
                         << " has to be a power of two";
  }
  return success();
}

static ParseResult parseInitArrayOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType init;
  VectorType resultType;
  if (parser.parseOperand(init) ||
      parser.parseOptionalAttrDict(result.attributes) || 
      parser.parseColon() || parser.parseType(resultType))
    return failure();

  result.addTypes(resultType);
  return parser.resolveOperands({init}, 
                                {resultType.getElementType()},
                                parser.getNameLoc(), result.operands);
}

//===----------------------------------------------------------------------===//
// Read Operations
//===----------------------------------------------------------------------===//

static void printReadOp(OpAsmPrinter &p, ReadOp &op) {
  p << " " << op.base() << "[" << op.index() << "]";
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op.base().getType() << ", " << op.result().getType();
}

template <typename Op>
static LogicalResult verifyReadOp(Op op) {
  auto type = op.result().getType().getIntOrFloatBitWidth();
  // The value's type must match the return type.
  if (op.getArrayType().getElementType().getIntOrFloatBitWidth() != type) {
    return op.emitOpError() << "element type of the array must match "
                         << " bitwidth of return type: " << type;
  }
  if (op.getArrayType().getShape().size() != 1) {
    return op.emitOpError() << "provide only one shape attribute ";
  }
  auto shape = op.getArrayType().getShape()[0];
  auto indicator = shape & (shape - 1);
  if (indicator != 0) {
    return op.emitOpError() << "given shape: " <<  shape 
                         << " has to be a power of two";
  }
  return success();
}

static ParseResult parseReadOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType base, index;
  VectorType baseType; IntegerType indexType;
  Type resultType;
  if (parser.parseOperand(base) || parser.parseLSquare() || 
      parser.parseOperand(index) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || 
      parser.parseColon() || parser.parseType(baseType) ||
      parser.parseOptionalComma() || parser.parseType(resultType))
    return failure();

  result.addTypes(resultType);
  indexType = parser.getBuilder().getIntegerType(log2(baseType.getShape()[0]));
  return parser.resolveOperands({base, index}, {baseType, indexType},
                                parser.getNameLoc(), result.operands);
}

//===----------------------------------------------------------------------===//
// Write Operations
//===----------------------------------------------------------------------===//

void printWriteOp(OpAsmPrinter &p, WriteOp &op) {
  p << " " << op.value() << ", " << op.base() << "[" << op.index() << "]";
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op.result().getType();
}

template <typename Op>
LogicalResult verifyWriteOp(Op op) {
  auto type = op.value().getType().getIntOrFloatBitWidth();
  // The value's type must match the array's element type.
  if (op.getArrayType().getElementType().getIntOrFloatBitWidth() != type) {
    return op.emitOpError() << "element type of the array must match "
                         << " bitwidth of given value: " << type;
  }
  return success();
}

static ParseResult parseWriteOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType value, base, index;
  VectorType resultType; IntegerType indexType;
  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(base) || parser.parseLSquare() || 
      parser.parseOperand(index) || parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) || 
      parser.parseColon() || parser.parseType(resultType))
    return failure();

  result.addTypes(resultType);
  indexType = parser.getBuilder().getIntegerType(log2(resultType.getShape()[0]));
  return parser.resolveOperands({value, base, index}, 
                                {resultType.getElementType(), resultType, indexType},
                                parser.getNameLoc(), result.operands);
}

//===----------------------------------------------------------------------===//
// Constant Operations
//===----------------------------------------------------------------------===//

template <typename Op>
LogicalResult verifyConstantOp(Op op) {
  Type resType = op.result().getType();
  btor::BitVecType resultType = resType.dyn_cast<btor::BitVecType>();
  Type attrType = op.valueAttr().getType();
  btor::BitVecType attributeType = attrType.dyn_cast<btor::BitVecType>();
  if (resultType && attributeType && attributeType == resultType &&
     resultType.getLength() == attributeType.getLength()) return success();
  else return failure();
}

//===----------------------------------------------------------------------===//
// Compare Operations
//===----------------------------------------------------------------------===//

template <typename Op>
LogicalResult verifyCmpOp(Op op) {
  Type resultType = op.result().getType();
  unsigned resultLength = resultType.dyn_cast<btor::BitVecType>().getLength();
  if(resultLength != 1){
    return op.emitOpError() << "result must be bit vector of length 1 instead got length of "
                         << resultLength;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AssertNot Operations
//===----------------------------------------------------------------------===//

template <typename Op>
LogicalResult verifyAssertNotOp(Op op) {
  Type resultType = op.arg().getType();
  unsigned resultLength = resultType.dyn_cast<btor::BitVecType>().getLength();
  if(resultLength != 1){
    return op.emitOpError() << "result must be bit vector of length 1 instead got length of "
                         << resultLength;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/Btor/IR/BtorOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorOpsEnums.cpp.inc"