//===----------------------------------------------------------------------===//
//
// This provides registration calls for Btor dialect to Btor IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
#define TARGET_BTOR_BTORTOBTORIRTRANSLATION_H

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class MLIRContext;
class ModuleOp;

namespace btor {

/// Deserializes the given Btor module and creates a MLIR ModuleOp
/// in the given `context`. Returns the ModuleOp on success; otherwise, reports
/// errors to the error handler registered with `context` and returns a null
/// module.
OwningOpRef<mlir::ModuleOp> translateBtorToMLIR(ArrayRef<uint32_t> btor,
                                         MLIRContext *context);

/// Register the Btor translation
void registerFromBtorTranslation();

} // namespace btor
} // namespace mlir

#endif // TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
