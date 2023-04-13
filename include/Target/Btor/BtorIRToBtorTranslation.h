//===----------------------------------------------------------------------===//
//
// This provides registration calls for Btor to Btor IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_BTOR_BTORIRTOBTORTRANSLATION_H
#define TARGET_BTOR_BTORIRTOBTORTRANSLATION_H

#include "Dialect/Btor/IR/Btor.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <vector>
#include <map>

#include "btor2parser/btor2parser.h"

namespace mlir {
class ModuleOp;

namespace btor {

/// Serializes the given Btor MLIR module and creates a Btor.

class Serialize {

 public:
///===----------------------------------------------------------------------===//
/// Constructors and Destructors
///===----------------------------------------------------------------------===//

  Serialize(ModuleOp module, raw_ostream &output) : m_module(module),
    m_output(output)  {}

  ~Serialize() {}

  LogicalResult translateMainFunction();

 private:
  ModuleOp m_module;
  raw_ostream &m_output;
};

/// Register the Btor translation
void registerToBtorTranslation();

} // namespace btor
} // namespace mlir

#endif // TARGET_BTOR_BTORIRTOBTORTRANSLATION_H
