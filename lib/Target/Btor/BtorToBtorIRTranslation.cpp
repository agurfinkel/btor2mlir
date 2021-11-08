#include "Target/Btor/BtorToBtorIRTranslation.h"
#include "Dialect/Btor/IR/BtorDialect.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::btor;

static OwningModuleRef deserializeModule(const llvm::MemoryBuffer *input,
                                         MLIRContext *context) {
  context->loadDialect<btor::BtorDialect>();
  // To be fillied with our btor specific code
  OwningModuleRef module;
  return module;
}

namespace mlir {
namespace btor {
void registerFromBtorTranslation() {
  TranslateToMLIRRegistration fromBtor(
      "import-btor",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
        return deserializeModule(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context);
      });
}
} // namespace btor
} // namespace mlir
