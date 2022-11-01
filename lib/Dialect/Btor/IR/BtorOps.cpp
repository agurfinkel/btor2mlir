//===- BtorOps.cpp - Btor dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <utility>

#include "Dialect/Btor/IR/Btor.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
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
  SmallVector<OpAsmParser::UnresolvedOperand, 1> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/1) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(operandType))
    return failure();

  result.addTypes(parser.getBuilder().getI1Type());
  return parser.resolveOperands(operands, {operandType}, parser.getNameLoc(),
                                result.operands);
}

ParseResult RedAndOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryDifferentResultOp(parser, result);
}

ParseResult RedOrOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryDifferentResultOp(parser, result);
}

ParseResult RedXorOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseUnaryDifferentResultOp(parser, result);
}

void RedAndOp::print(OpAsmPrinter &p) { printBtorUnaryOp(p, *this); }

void RedOrOp::print(OpAsmPrinter &p) { printBtorUnaryOp(p, *this); }

void RedXorOp::print(OpAsmPrinter &p) { printBtorUnaryOp(p, *this); }

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

ParseResult SliceOp::parse(OpAsmParser &parser, OperationState &result) {

  Type resultType, operandType;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operands;
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
void SliceOp::print(OpAsmPrinter &p) {
  assert(getNumOperands() == 3 && "slice op should have one operand");
  assert((*this)->getNumResults() == 1 && "slice op should have one result");

  p << ' ' << getOperand(0) << ", " << getOperand(1) << ", " << getOperand(2);
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getOperand(0).getType() << ", " << getType();
}

LogicalResult SliceOp::verify() {
  Type srcType = getElementTypeOrSelf(in().getType());
  Type dstType = getElementTypeOrSelf(getType());

  if (srcType.cast<IntegerType>().getWidth() <=
      dstType.cast<IntegerType>().getWidth())
    return emitError("result type ")
           << dstType << " must be less than operand type " << srcType;

  return success();
}

//===----------------------------------------------------------------------===//
// IteOp
//===----------------------------------------------------------------------===//

void IteOp::print(OpAsmPrinter &p) {
  p << " " << getOperands();
  p << " : ";
  if (ShapedType condType = getCondition().getType().dyn_cast<ShapedType>())
    p << condType << ", ";
  p << getType();
}

ParseResult IteOp::parse(OpAsmParser &parser, OperationState &result) {
  Type conditionType, resultType;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operands;
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

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  auto type = getType();
  // The value's type must match the return type.
  if (value().getType() != type) {
    return emitOpError() << "value type " << value().getType()
                         << " must match return type: " << type;
  }
  // Only integer attribute are acceptable.
  if (!value().isa<IntegerAttr>()) {
    return emitOpError("value must be an integer attribute");
  }
  return success();
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
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(operandType))
    return failure();

  result.addTypes(parser.getBuilder().getI1Type());
  return parser.resolveOperands(operands, {operandType, operandType},
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

ParseResult SAddOverflowOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  return parseBinaryOverflowOp(parser, result);
}

ParseResult SDivOverflowOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  return parseBinaryOverflowOp(parser, result);
}

ParseResult SMulOverflowOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  return parseBinaryOverflowOp(parser, result);
}

ParseResult SSubOverflowOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  return parseBinaryOverflowOp(parser, result);
}

ParseResult UAddOverflowOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  return parseBinaryOverflowOp(parser, result);
}

ParseResult UMulOverflowOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  return parseBinaryOverflowOp(parser, result);
}

ParseResult USubOverflowOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  return parseBinaryOverflowOp(parser, result);
}

void SAddOverflowOp::print(OpAsmPrinter &printer) {
  printBinaryOverflowOp(printer, *this);
}

void SDivOverflowOp::print(OpAsmPrinter &printer) {
  printBinaryOverflowOp(printer, *this);
}

void SMulOverflowOp::print(OpAsmPrinter &printer) {
  printBinaryOverflowOp(printer, *this);
}

void SSubOverflowOp::print(OpAsmPrinter &printer) {
  printBinaryOverflowOp(printer, *this);
}

void UAddOverflowOp::print(OpAsmPrinter &printer) {
  printBinaryOverflowOp(printer, *this);
}

void UMulOverflowOp::print(OpAsmPrinter &printer) {
  printBinaryOverflowOp(printer, *this);
}

void USubOverflowOp::print(OpAsmPrinter &printer) {
  printBinaryOverflowOp(printer, *this);
}

//===----------------------------------------------------------------------===//
// Extension Operations
//===----------------------------------------------------------------------===//

template <typename ValType, typename Op>
static LogicalResult verifyExtOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.in().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() >= dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be wider than operand type " << srcType;

  return success();
}

LogicalResult UExtOp::verify() { return verifyExtOp<IntegerType>(*this); }

LogicalResult SExtOp::verify() { return verifyExtOp<IntegerType>(*this); }

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

ParseResult ConcatOp::parse(OpAsmParser &parser, OperationState &result) {
  Type resultType, firstOperandType, secondOperandType;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(firstOperandType) || parser.parseOptionalComma() ||
      parser.parseType(secondOperandType) || parser.parseOptionalComma() ||
      parser.parseType(resultType))
    return failure();

  result.addTypes(resultType);
  return parser.resolveOperands(operands, {firstOperandType, secondOperandType},
                                parser.getNameLoc(), result.operands);
}

void ConcatOp::print(OpAsmPrinter &p) {
  assert(getNumOperands() == 2 && "concat op should have two operands");

  p << ' ' << getOperand(0) << ", " << getOperand(1);
  p.printOptionalAttrDict((*this)->getAttrs());

  // Now we can output the types for all operands and the result.
  p << " : " << getOperand(0).getType() << ", " << getOperand(1).getType()
    << ", " << getResult().getType();
}

LogicalResult ConcatOp::verify() {
  Type firstType = getElementTypeOrSelf(lhs().getType());
  Type secondType = getElementTypeOrSelf(rhs().getType());
  Type dstType = getElementTypeOrSelf(getType());

  auto sumOfTypes = firstType.cast<IntegerType>().getWidth() +
                    secondType.cast<IntegerType>().getWidth();
  if (sumOfTypes != dstType.cast<IntegerType>().getWidth())
    return emitError("sum of ")
           << firstType << " and " << secondType
           << " must be equal to operand type i" << sumOfTypes;

  return success();
}

//===----------------------------------------------------------------------===//
// Input Operation
//===----------------------------------------------------------------------===//

void InputOp::print(OpAsmPrinter &p) {
  p << " " << value() << " : " << getOperand().getType();
  p << " ";
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult InputOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> ops;
  NamedAttrList attrs;
  Attribute idAttr;
  Type type;

  if (parser.parseAttribute(idAttr, "id", attrs) || parser.parseComma() ||
      parser.parseOperandList(ops, 1) || parser.parseOptionalAttrDict(attrs) ||
      parser.parseColonType(type) ||
      parser.resolveOperands(ops, type, result.operands))
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

// static void printArrayOp(OpAsmPrinter &p, Operation *op) {
//   assert(op->getNumOperands() == 2 &&
//          "indexed array op should have two operands");
//   assert(op->getNumResults() == 1 && "indexed array op should have one result");

//   p << ' ' << op->getOperand(0) << ", " << op->getOperand(1);
//   p.printOptionalAttrDict(op->getAttrs());

//   // Now we can output only one type for all operands and the result.
//   p << " : " << op->getResult(0).getType();
// }

// void ReadOp::print(OpAsmPrinter &p) { printArrayOp(p, (*this)); }
// // "$array `[` $index `]` attr-dict `:` type($result)";
// ParseResult ReadOp::parse(OpAsmParser &parser, OperationState &result) {
//   Type resultType, firstOperandType;
//   SmallVector<OpAsmParser::UnresolvedOperand, 1> operands;
//   if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
//       parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
//       parser.parseType(firstOperandType) || parser.parseOptionalComma() ||
//       parser.parseType(resultType))
//     assert(true &&
//          "indexed array op should have two operands");
//     return failure();

//   result.addTypes(resultType);
//   return parser.resolveOperands(operands, {firstOperandType},
//                                 parser.getNameLoc(), result.operands);
// }
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/Btor/IR/BtorOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorOpsEnums.cpp.inc"