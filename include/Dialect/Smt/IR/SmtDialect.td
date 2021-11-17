//===- SmtDialect.td - Smt dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SMT_DIALECT
#define SMT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Smt constants definition.
//===----------------------------------------------------------------------===//

def Smt_EQ  : I64EnumAttrCase<"eq", 0>;
def Smt_NE  : I64EnumAttrCase<"ne", 1>;
def Smt_SLT : I64EnumAttrCase<"slt", 2>;
def Smt_SLE : I64EnumAttrCase<"sle", 3>;
def Smt_SGT : I64EnumAttrCase<"sgt", 4>;
def Smt_SGE : I64EnumAttrCase<"sge", 5>;
def Smt_ULT : I64EnumAttrCase<"ult", 6>;
def Smt_ULE : I64EnumAttrCase<"ule", 7>;
def Smt_UGT : I64EnumAttrCase<"ugt", 8>;
def Smt_UGE : I64EnumAttrCase<"uge", 9>;

def SmtPredicateAttr : I64EnumAttr<
    "SmtPredicate", "smt.cmp comparison predicate",
    [Smt_EQ, Smt_NE, Smt_SLT, Smt_SLE, Smt_SGT,
     Smt_SGE, Smt_ULT, Smt_ULE, Smt_UGT, Smt_UGE]> {
  let cppNamespace = "::mlir::smt";
}

//===----------------------------------------------------------------------===//
// Smt dialect definition.
//===----------------------------------------------------------------------===//

def Smt_Dialect : Dialect {
    let name = "smt";
    let summary = "An SMT2 MLIR dialect";
    let cppNamespace = "::mlir::smt";
}

//===----------------------------------------------------------------------===//
// Base Smt operation definition.
//===----------------------------------------------------------------------===//

class Smt_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Smt_Dialect, mnemonic, traits>;

#endif // SMT_DIALECT
