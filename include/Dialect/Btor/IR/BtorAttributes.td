#ifndef BTOR_ATTRS
#define BTOR_ATTRS

include "mlir/IR/EnumAttr.td"
include "BtorBase.td"

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
// btor attributes definition.
//===----------------------------------------------------------------------===//

class Btor_Attr<string name, string attrMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute">
    : AttrDef<Btor_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def BitVecAttribute : Btor_Attr<"BitVec", "bvattr"> {
  let summary = "bit vector attribute";
  let description = [{
    Bit vector attribute
  }];
  let parameters = (ins "BitVecType":$type, "APInt":$value);   
}

#endif // BTOR_ATTRS