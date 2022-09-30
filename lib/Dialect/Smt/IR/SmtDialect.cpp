//===- SmtDialect.cpp - Smt dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Smt/IR/Smt.h"

#include "Dialect/Smt/IR/SmtOpsDialect.cpp.inc"

// Pull in all enum type definitions and utility function declarations.
#include "Dialect/Smt/IR/SmtOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::smt;

//===----------------------------------------------------------------------===//
// Smt dialect.
//===----------------------------------------------------------------------===//

void SmtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Smt/IR/SmtOps.cpp.inc"
      >();
}
