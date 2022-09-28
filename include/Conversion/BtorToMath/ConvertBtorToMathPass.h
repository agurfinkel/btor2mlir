#ifndef BTOR_CONVERSION_BTORTOMATH_CONVERTBTORTOMATHPASS_H_
#define BTOR_CONVERSION_BTORTOMATH_CONVERTBTORTOMATHPASS_H_

#include <memory>

namespace mlir {
struct LogicalResult;
class Pass;

class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

namespace btor {
    /// Collect a set of patterns to lower from btor.add to Math dialect
    void populateBtorToMathConversionPatterns(RewritePatternSet &patterns);

    /// Creates a pass to convert the Btor dialect into the Math dialect.
    std::unique_ptr<mlir::Pass> createLowerToMathPass();

} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOMATH_CONVERTBTORTOMATHPASS_H_