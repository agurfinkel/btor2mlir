//===- BtorDialect.h - Btor dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_BTORDIALECT_H
#define BTOR_BTORDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "Dialect/Btor/IR/BtorOpsDialect.h.inc"

// namespace mlir {
// namespace btor {

// class BtorDialect : public Dialect {
// public:
//   /// Parse an instance of a type registered to the dialect.
//   Type parseType(DialectAsmParser &parser) const override;

//   /// Print an instance of a type registered to the dialect.
//   void printType(Type type, DialectAsmPrinter &printer) const override;
// };

// } // namespace btor
// } // namespace mlir

#endif // BTOR_BTORDIALECT_H
