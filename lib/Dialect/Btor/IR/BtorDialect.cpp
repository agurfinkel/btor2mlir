//===- BtorDialect.cpp - Btor dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/Btor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::btor;
using namespace mlir::btor::detail;

//===----------------------------------------------------------------------===//
// Btor dialect
//===----------------------------------------------------------------------===//

void BtorDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Btor/IR/BtorOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Dialect/Btor/IR/BtorOps.cpp.inc"
      >();
}

#include "Dialect/Btor/IR/BtorOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Btor types
//===----------------------------------------------------------------------===//

LogicalResult ArrayType::verify(
    function_ref<InFlightDiagnostic ()> emitError,
    IntegerType shape,
    IntegerType elementType) {
    if (!isValidElementType(elementType))
      return emitError()
            << "array elements must be int type but got "
            << elementType;

    if (!isValidElementType(elementType))
      return emitError()
            << "array types must have index of int type but got "
            << shape;

    return success();
}

void ArrayType::walkImmediateSubElements(
    function_ref<void (Attribute)> walkAttrsFn,
    function_ref<void (Type)> walkTypesFn) const {
    walkTypesFn(getElementType());
}

Type ArrayType::replaceImmediateSubElements(
    ArrayRef<Attribute> replAttrs,
    ArrayRef<Type> replTypes) const {
    return get(getShape().getContext(), getShape(), replTypes.front().cast<IntegerType>());
}

ArrayType ArrayType::cloneWith(
    IntegerType shape,
    IntegerType elementType) const {
    return ArrayType::get(shape.getContext(), shape, elementType);
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/Btor/IR/BtorOpsTypes.cpp.inc"