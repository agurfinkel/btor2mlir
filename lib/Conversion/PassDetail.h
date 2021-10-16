#ifndef BTORCONVERSION_PASSDETAIL_H_
#define BTORCONVERSION_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class StandardOpsDialect;

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace btor {
class BtorDialect;
} // end namespace btor

namespace LLVM {
class LLVMDialect;
} // end namespace LLVM

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

} // end namespace mlir

#endif // BTORCONVERSION_PASSDETAIL_H_