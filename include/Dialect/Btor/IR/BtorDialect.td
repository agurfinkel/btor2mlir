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
// btor constants definition.
//===----------------------------------------------------------------------===//

def BTOR_EQ  : I64EnumAttrCase<"eq", 0>;
def BTOR_NE  : I64EnumAttrCase<"ne", 1>;
def BTOR_SLT : I64EnumAttrCase<"slt", 2>;
def BTOR_SLE : I64EnumAttrCase<"sle", 3>;
def BTOR_SGT : I64EnumAttrCase<"sgt", 4>;
def BTOR_SGE : I64EnumAttrCase<"sge", 5>;
def BTOR_ULT : I64EnumAttrCase<"ult", 6>;
def BTOR_ULE : I64EnumAttrCase<"ule", 7>;
def BTOR_UGT : I64EnumAttrCase<"ugt", 8>;
def BTOR_UGE : I64EnumAttrCase<"uge", 9>;

def BtorPredicateAttr : I64EnumAttr<
    "BtorPredicate", "btor.cmp comparison predicate",
    [BTOR_EQ, BTOR_NE, BTOR_SLT, BTOR_SLE, BTOR_SGT,
     BTOR_SGE, BTOR_ULT, BTOR_ULE, BTOR_UGT, BTOR_UGE]> {
  let cppNamespace = "::mlir::btor";
}

//===----------------------------------------------------------------------===//
// Btor dialect definition.
//===----------------------------------------------------------------------===//

def Btor_Dialect : Dialect {
    let name = "btor";
    let summary = "A BTOR2 MLIR dialect";
    let cppNamespace = "::mlir::btor";
    let dependentDialects = ["mlir::StandardOpsDialect"];
}

//===----------------------------------------------------------------------===//
// Btor type definitions
//===----------------------------------------------------------------------===//

def BtorArrayType : TypeDef<Btor_Dialect, "BtorArrayType"> {
  let parameters = (ins SignlessIntegerLike:$a, SignlessIntegerLike:$b);
  let mnemonic = "array";
  let assemblyFormat = "`<` params `>`";
}

//===----------------------------------------------------------------------===//
// Base btor operation definition.
//===----------------------------------------------------------------------===//

class Btor_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Btor_Dialect, mnemonic, traits>;

#endif // BTOR_DIALECT
