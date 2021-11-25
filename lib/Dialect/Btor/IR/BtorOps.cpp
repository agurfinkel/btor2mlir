//===- BtorOps.cpp - Btor dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorDialect.h"
#include "Dialect/Btor/IR/BtorOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::btor;

/// A custom unary operation printer that omits the "std." prefix from the
/// operation names.
static void printBtorUnaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 1 && "unary op should have one operand");
  assert(op->getNumResults() == 1 && "unary op should have one result");

  p << ' ' << op->getOperand(0);
  p.printOptionalAttrDict(op->getAttrs());
  p << " : " << op->getOperand(0).getType();
}

static ParseResult parseUnaryOp(OpAsmParser &parser, OperationState &result) {  
  Type operandType, resultType;
  SmallVector<OpAsmParser::OperandType, 1> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/1) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(operandType))
    return failure();
  
  result.addTypes(parser.getBuilder().getI1Type());
  return parser.resolveOperands(operands, {operandType},
                                parser.getNameLoc(), result.operands);
}

/// A custom binary operation printer that omits the "btor." prefix from the
/// operation names.
static void printBtorBinaryOp(Operation *op, OpAsmPrinter &p) {
  assert(op->getNumOperands() == 2 && "binary op should have two operands");
  assert(op->getNumResults() == 1 && "binary op should have one result");

  // If not all the operand and result types are the same, just use the
  // generic assembly form to avoid omitting information in printing.
  auto resultType = op->getResult(0).getType();
  if (op->getOperand(0).getType() != resultType ||
      op->getOperand(1).getType() != resultType) {
    p.printGenericOp(op);
    return;
  }

  p << ' ' << op->getOperand(0) << ", " << op->getOperand(1);
  p.printOptionalAttrDict(op->getAttrs());

  // Now we can output only one type for all operands and the result.
  p << " : " << op->getResult(0).getType();
}

//===----------------------------------------------------------------------===//
// General helpers for comparison ops
//===----------------------------------------------------------------------===//

// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  return i1Type;
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

static void buildCmpOp(OpBuilder &build, OperationState &result,
                        BtorPredicate predicate, Value lhs, Value rhs) {
  result.addOperands({lhs, rhs});
  result.types.push_back(getI1SameShape(lhs.getType()));
  result.addAttribute(CmpOp::getPredicateAttrName(),
                      build.getI64IntegerAttr(static_cast<int64_t>(predicate)));
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

template <typename ValType, typename Op>
static LogicalResult verifySliceOp(Op op) {
  Type srcType = getElementTypeOrSelf(op.in().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (srcType.cast<ValType>().getWidth() <= dstType.cast<ValType>().getWidth())
    return op.emitError("result type ")
           << dstType << " must be less than operand type " << srcType;

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

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static void printConstantOp(OpAsmPrinter &p, mlir::btor::ConstantOp &op) {
  p << " ";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  p << op.getValue();
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(valueAttr, "value", result.attributes))
    return failure();

  // Add the attribute type to the list.
  return parser.addTypeToList(valueAttr.getType(), result.types);
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

//===----------------------------------------------------------------------===//
// Overflow Operations
//===----------------------------------------------------------------------===//

static ParseResult parseBinaryOverflowOp(OpAsmParser &parser,
                                  OperationState &result) {  
  Type operandType, resultType;
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

static void printBtorBinaryDifferentResultTypeOp(Operation *op, OpAsmPrinter &p) {
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

  if (srcType.cast<ValType>().getWidth() >= dstType.cast<ValType>().getWidth())
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

template <typename ValType, typename Op>
static LogicalResult verifyConcatOp(Op op) {
  Type firstType = getElementTypeOrSelf(op.lhs().getType());
  Type secondType = getElementTypeOrSelf(op.rhs().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  auto sumOfTypes = firstType.cast<ValType>().getWidth() + 
              secondType.cast<ValType>().getWidth();
  if ( sumOfTypes != dstType.cast<ValType>().getWidth())
    return op.emitError("sum of ") << firstType << " and "
         << secondType << " must be equal to operand type " << dstType;

  return success();
}

//===----------------------------------------------------------------------===//
// Input Operation
//===----------------------------------------------------------------------===//

static void printInputOp(OpAsmPrinter &p, mlir::btor::InputOp &op) {
    p << " "  << op.value() << " : " << op->getOperand(0).getType();
    p << " ";
    p.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseInputOp(OpAsmParser &parser,OperationState &result) {  
    SmallVector<OpAsmParser::OperandType> ops;
    NamedAttrList attrs;
    Attribute idAttr;
    Type type;

    if (parser.parseAttribute(idAttr, "id", attrs) ||
        parser.parseComma() ||
        parser.parseOperandList(ops, 1) ||
        parser.parseOptionalAttrDict(attrs) || 
        parser.parseColonType(type) ||
        parser.resolveOperands(ops, type, result.operands)
        )
        return failure();

    if (!idAttr.isa<mlir::IntegerAttr>())
        return parser.emitError(parser.getNameLoc(),
                                "expected integer id attribute");

    result.attributes = attrs;
    result.addTypes({type});
  return success();
}

#define GET_OP_CLASSES
#include "Dialect/Btor/IR/BtorOps.cpp.inc"
