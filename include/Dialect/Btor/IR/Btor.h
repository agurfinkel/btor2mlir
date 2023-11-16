//===- BtorOps.h - Btor dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_IR_BTOR_H
#define BTOR_IR_BTOR_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// BtorDialect
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Btor Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Btor Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/Btor/IR/BtorOps.h.inc"

//===----------------------------------------------------------------------===//
// Btor Dialect Types
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorTypes.h"

//===----------------------------------------------------------------------===//
// Btor Dialect Attributes
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorAttributes.h"

#endif // BTOR_IR_BTOR_H
