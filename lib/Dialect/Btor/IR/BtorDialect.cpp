//===- BtorDialect.cpp - Btor dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorDialect.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "Dialect/Btor/IR/BtorOpsDialect.cpp.inc"

// Pull in all enum type definitions and utility function declarations.
#include "Dialect/Btor/IR/BtorOpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::btor;

//===----------------------------------------------------------------------===//
// Btor dialect.
//===----------------------------------------------------------------------===//

void BtorDialect::initialize() {
  // addTypes<ArrayType>();
  addOperations<
#define GET_OP_LIST
#include "Dialect/Btor/IR/BtorOps.cpp.inc"
      >();
}
