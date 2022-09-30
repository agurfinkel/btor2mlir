//===- SmtOps.td - Smt dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SMT_OPS
#define SMT_OPS

include "SmtBase.td"

//===----------------------------------------------------------------------===//
// Base Smt operation definition.
//===----------------------------------------------------------------------===//

class Smt_Op<string mnemonic, list<Trait> traits = []> :
        Op<Smt_Dialect, mnemonic, traits>;

#endif // SMT_OPS
