//===- standalone-translate.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "Dialect/Btor/IR/Btor.h"
#include "Target/Btor/BtorToBtorIRTranslation.h"
#include "Target/Btor/BtorIRToBtorTranslation.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::btor::registerFromBtorTranslation();
  mlir::btor::registerToBtorTranslation();
  
  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
