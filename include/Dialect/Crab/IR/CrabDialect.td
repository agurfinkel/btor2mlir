//===- CrabDialect.td - Crab dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CRAB_DIALECT
#define CRAB_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// crab constants definition.
//===----------------------------------------------------------------------===//

def CRAB_EQ  : I64EnumAttrCase<"eq", 0>;
def CRAB_NE  : I64EnumAttrCase<"ne", 1>;
def CRAB_SLT : I64EnumAttrCase<"slt", 2>;
def CRAB_SLE : I64EnumAttrCase<"sle", 3>;
def CRAB_SGT : I64EnumAttrCase<"sgt", 4>;
def CRAB_SGE : I64EnumAttrCase<"sge", 5>;
def CRAB_ULT : I64EnumAttrCase<"ult", 6>;
def CRAB_ULE : I64EnumAttrCase<"ule", 7>;
def CRAB_UGT : I64EnumAttrCase<"ugt", 8>;
def CRAB_UGE : I64EnumAttrCase<"uge", 9>;

def CrabPredicateAttr : I64EnumAttr<
    "CrabPredicate", "crab.cmp comparison predicate",
    [CRAB_EQ, CRAB_NE, CRAB_SLT, CRAB_SLE, CRAB_SGT,
     CRAB_SGE, CRAB_ULT, CRAB_ULE, CRAB_UGT, CRAB_UGE]> {
  let cppNamespace = "::mlir::crab";
}

//===----------------------------------------------------------------------===//
// Crab dialect definition.
//===----------------------------------------------------------------------===//

def Crab_Dialect : Dialect {
    let name = "crab";
    let summary = "A Crab MLIR dialect";
    let cppNamespace = "::mlir::crab";
}

//===----------------------------------------------------------------------===//
// Base crab operation definition.
//===----------------------------------------------------------------------===//

class Crab_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Crab_Dialect, mnemonic, traits>;

#endif // CRAB_DIALECT
