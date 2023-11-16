#ifndef BTOR_TYPES
#define BTOR_TYPES

include "BtorBase.td"

class Btor_BaseType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Btor_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Btor_BitVec : Btor_BaseType<"BitVec", "bv"> {
  let summary = "bit vector behaving similar to unsinged int";
  let description = [{
    Bit vectors have designated length.
  }];

  let parameters = (ins "unsigned":$length);

  let assemblyFormat = "`<` $length `>`";

  let genVerifyDecl = 0;
}

#endif // BTOR_TYPES