//===- SmtOps.h - Smt dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SMT_SMTOPS_H
#define SMT_SMTOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
// #include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// SmtDialect
//===----------------------------------------------------------------------===//

#include "Dialect/Smt/IR/SmtOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Smt Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "Dialect/Smt/IR/SmtOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// Smt Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/Smt/IR/SmtOps.h.inc"

#endif // SMT_SMTOPS_H
