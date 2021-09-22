//===- BtorDialect.td - Btor dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_DIALECT
#define BTOR_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Btor dialect definition.
//===----------------------------------------------------------------------===//

def Btor_Dialect : Dialect {
    let name = "btor";
    let summary = "An out-of-tree MLIR Btor dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::btor";
}

//===----------------------------------------------------------------------===//
// Base btor operation definition.
//===----------------------------------------------------------------------===//

class Btor_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Btor_Dialect, mnemonic, traits>;

#endif // BTOR_DIALECT
