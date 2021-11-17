//===- SmtOps.h - Smt dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SMT_SMTOPS_H
#define SMT_SMTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in all enum type definitions and utility function declarations.
#include "Dialect/Smt/IR/SmtOpsEnums.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Smt/IR/SmtOps.h.inc"

#endif // SMT_SMTOPS_H
