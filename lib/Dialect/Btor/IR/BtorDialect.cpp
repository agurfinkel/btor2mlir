//===- BtorDialect.cpp - Btor dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/BtorDialect.h"
#include "Dialect/Btor/IR/BtorOps.h"

using namespace mlir;
using namespace mlir::btor;

#include "Btor/BtorOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Btor dialect.
//===----------------------------------------------------------------------===//

void BtorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Btor/BtorOps.cpp.inc"
      >();
}
