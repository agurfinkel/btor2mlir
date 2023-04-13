#include "Target/Btor/BtorIRToBtorTranslation.h"
#include "Dialect/Btor/IR/Btor.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>
#include <string>

using namespace mlir;
using namespace mlir::btor;

LogicalResult Serialize::translateMainFunction() {
  m_output.write(reinterpret_cast<char *>(m_module.dump()),
               30 * sizeof(uint32_t));
  return success();
}

//===----------------------------------------------------------------------===//
// Serialization registration
//===----------------------------------------------------------------------===//

static LogicalResult serializeModule(ModuleOp module, raw_ostream &output) {
  if (!module)
    return failure();

  Serialize serialize(module, output);

  if (serialize.translateMainFunction().failed())
    return failure();

  output.write(reinterpret_cast<char *>(module.dump()),
               30 * sizeof(uint32_t));
  output.write('\n');

  return mlir::success();
}

namespace mlir {
namespace btor {
void registerToBtorTranslation() {
  TranslateFromMLIRRegistration toBtor(
      "export-btor", 
      [](ModuleOp module, raw_ostream &output) {
        return serializeModule(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<btor::BtorDialect, 
          arith::ArithmeticDialect,
          StandardOpsDialect>();
      });
}
} // namespace btor
} // namespace mlir