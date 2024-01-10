#ifndef BTOR_TYPES
#define BTOR_TYPES

include "BtorBase.td"

class Btor_BaseType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Btor_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Btor_BitVec : Btor_BaseType<"BitVec", "bv"> {
  let summary = "bit vector behaving similar to unsinged int";
  let description = [{Bit vectors have designated width.}];

  let parameters = (ins "unsigned" : $width);

  let assemblyFormat = "`<` $width `>`";

  let genVerifyDecl = 0;
}

def Btor_Array : Btor_BaseType<"Array", "array"> {
  let summary = "btor array";
  let description = [{A Btor array parametrized by two btor bit vectors}];

  let parameters = (ins "BitVecType" : $shape, "BitVecType" : $element);

  let assemblyFormat = "`<` $shape `,` $element `>`";

  let genVerifyDecl = 0;
}
#endif // BTOR_TYPES
