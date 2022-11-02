//===- BtorBase.td - Btor dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_TYPES
#define BTOR_TYPES

include "BtorBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Btor types
//===----------------------------------------------------------------------===//

class BtorDialect_Type<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Btor_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

//===----------------------------------------------------------------------===//
// Btor arrays
//===----------------------------------------------------------------------===//

def Btor_Array_Type : BtorDialect_Type<"Array", [
    DeclareTypeInterfaceMethods<SubElementTypeInterface>
  ], "Type"> {
  let summary = "btor array type";
  let mnemonic = "array";
  let description = [{

    Examples:

    ```mlir
    // A 2D fixed-length array of i3xi4 i32 elements.
    !btor.array<i3,i4,i32>

    // A 1D scalable-length array that contains i4 i32 elements.
    !btor.array<i4,i32>
    ```
  }];
  let parameters = (ins
    "IntegerType":$shape,
    "IntegerType":$elementType
  );
    let extraClassDeclaration = [{
    /// Returns true if the given type can be used as an element of an array
    /// type. In particular, array can consist of only integer primitives.
    static bool isValidElementType(Type t) {
      return t.isa<IntegerType>();
    }

    /// Clone this array type with the given shape and element type. If the
    /// provided shape is `None`, the current shape of the type is used.
    ArrayType cloneWith(IntegerType shape, IntegerType elementType) const;
  }];
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $shape `,` $elementType `>`";
}

#endif // BTOR_TYPES
