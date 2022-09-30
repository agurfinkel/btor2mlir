#ifndef BTOR_CONVERSION_BTORTOARITHMETIC_CONVERTBTORTOARITHMETICPASS_H_
#define BTOR_CONVERSION_BTORTOARITHMETIC_CONVERTBTORTOARITHMETICPASS_H_

#include <memory>

namespace mlir {
// struct LogicalResult;
class Pass;

class RewritePatternSet;
// using OwningRewritePatternList = RewritePatternSet;

namespace btor {
    /// Collect a set of patterns to lower from btor.add to Math dialect
    void populateBtorToArithmeticConversionPatterns(RewritePatternSet &patterns);

    /// Creates a pass to convert the Btor dialect into the Math dialect.
    std::unique_ptr<Pass> createConvertBtorToArithmeticPass();

} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOARITHMETIC_CONVERTBTORTOARITHMETICPASS_H_