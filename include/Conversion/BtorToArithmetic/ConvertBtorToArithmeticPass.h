#ifndef BTOR_CONVERSION_BTORTOARITHMETIC_CONVERTBTORTOARITHMETICPASS_H_
#define BTOR_CONVERSION_BTORTOARITHMETIC_CONVERTBTORTOARITHMETICPASS_H_

#include <memory>

namespace mlir {
class Pass;

class RewritePatternSet;

namespace btor {
    /// Collect a set of patterns to lower from btor to arithmetic dialect
    void populateBtorToArithmeticConversionPatterns(RewritePatternSet &patterns);

    /// Creates a pass to convert the Btor dialect into the arithmetic dialect.
    std::unique_ptr<Pass> createConvertBtorToArithmeticPass();

} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOARITHMETIC_CONVERTBTORTOARITHMETICPASS_H_