//===- BtorAttrs.h - Btor dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_IR_BTOR_ATTRS_H
#define BTOR_IR_BTOR_ATTRS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


//===----------------------------------------------------------------------===//
// Btor Dialect Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Dialect/Btor/IR/BtorAttributes.h.inc"

using namespace mlir;
using namespace mlir::btor;

#endif // BTOR_IR_BTOR_ATTRS_H