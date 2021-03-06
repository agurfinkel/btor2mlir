//===- BtorOps.h - Btor dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_BTOROPS_H
#define BTOR_BTOROPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in all enum type definitions and utility function declarations.
#include "Dialect/Btor/IR/BtorOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Btor/IR/BtorOps.h.inc"

#endif // BTOR_BTOROPS_H
