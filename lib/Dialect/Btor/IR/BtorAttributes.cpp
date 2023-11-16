//===- BtorTypes.cpp - Btor dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/IR/Btor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "Dialect/Btor/IR/BtorTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "Dialect/Btor/IR/BtorAttributes.h"

using namespace mlir;
using namespace mlir::btor;

void BitVecAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer << this->getType();
  printer << ' ' << "=";
  printer << ' ';
  printer.printStrippedAttrOrType(this->getValue());
  printer << ">";
}

::mlir::Attribute BitVecAttr::parse(::mlir::AsmParser &parser,
                                    ::mlir::Type type) {
  ::mlir::FailureOr<BitVecType> _result_type;
  ::mlir::FailureOr<unsigned> _result_value;
  ::llvm::SMLoc loc = parser.getCurrentLocation();
  (void)loc;
  // Parse literal '<'
  if (parser.parseLess())
    return {};

  // Parse variable 'type'
  _result_type = ::mlir::FieldParser<BitVecType>::parse(parser);
  if (failed(_result_type)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse BitVecAttribute parameter 'type' which "
                     "is to be a `BitVecType`");
    return {};
  }
  // Parse literal '='
  if (parser.parseEqual())
    return {};

  // Parse variable 'value'
  _result_value = ::mlir::FieldParser<unsigned>::parse(parser);
  if (failed(_result_value)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse BitVecAttribute parameter 'value' which "
                     "is to be a `APInt`");
    return {};
  }
  // Parse literal '>'
  if (parser.parseGreater())
    return {};
  return BitVecAttr::get(
      parser.getContext(), _result_type.getValue(),
      APInt(_result_type.getValue().getLength(), _result_value.getValue()));
}

#define GET_ATTRDEF_CLASSES
#include "Dialect/Btor/IR/BtorAttributes.cpp.inc"

void BtorDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Btor/IR/BtorAttributes.cpp.inc"
      >();
}