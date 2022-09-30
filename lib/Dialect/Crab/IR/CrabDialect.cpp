//===- CrabDialect.cpp - Crab dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Crab/IR/Crab.h"

#include "Dialect/Crab/IR/CrabOpsDialect.cpp.inc"

// Pull in all enum type definitions and utility function declarations.
#include "Dialect/Crab/IR/CrabOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::crab;

//===----------------------------------------------------------------------===//
// Crab dialect.
//===----------------------------------------------------------------------===//

void CrabDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Crab/IR/CrabOps.cpp.inc"
      >();
}
