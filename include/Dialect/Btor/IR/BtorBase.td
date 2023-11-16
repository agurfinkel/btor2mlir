//===- BtorBase.td - Btor dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_BASE
#define BTOR_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Btor dialect definition.
//===----------------------------------------------------------------------===//

def Btor_Dialect : Dialect {
    let name = "btor";
    let summary = "A BTOR2 MLIR dialect";
    let cppNamespace = "::mlir::btor";
    let extraClassDeclaration = [{
    private:
            // Register the custom btor Types.
            void registerTypes();

            // Register the custom btor attributes.
            void registerAttrs();
    public:
    }];
}

#endif // BTOR_DIALECT
