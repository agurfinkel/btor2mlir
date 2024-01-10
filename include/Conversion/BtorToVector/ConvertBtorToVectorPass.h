#ifndef BTOR_CONVERSION_BTORTOVECTOR_CONVERTBTORTOVECTORPASS_H_
#define BTOR_CONVERSION_BTORTOVECTOR_CONVERTBTORTOVECTORPASS_H_

#include "Conversion/BtorToLLVM/ConvertBtorToLLVMPass.h"
#include <memory>

namespace mlir {
class Pass;

class RewritePatternSet;

namespace btor {
/// Collect a set of patterns to lower from btor to vector dialect
void populateBtorToVectorConversionPatterns(BtorToLLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);

/// Creates a pass to convert the Btor dialect into the vector dialect.
std::unique_ptr<Pass> createLowerToVectorPass();

} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOVECTOR_CONVERTBTORTOVECTORPASS_H_
