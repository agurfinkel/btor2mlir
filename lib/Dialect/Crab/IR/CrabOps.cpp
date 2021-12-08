//===- CrabOps.cpp - Crab dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Crab/IR/CrabDialect.h"
#include "Dialect/Crab/IR/CrabOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::crab;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static void printConstantOp(OpAsmPrinter &p, mlir::crab::ConstantOp &op) {
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

#define GET_OP_CLASSES
#include "Dialect/Crab/IR/CrabOps.cpp.inc"
