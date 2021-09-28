#ifndef BTOR_CONVERSION_BTORTOSTANDARD_CONVERTBTORTOSTANDARDPASS_H_
#define BTOR_CONVERSION_BTORTOSTANDARD_CONVERTBTORTOSTANDARDPASS_H_

#include <memory>

namespace mlir {
class Pass;

namespace btor {
    /// Creates a pass to convert the Btor dialect into the Standard dialect.
    std::unique_ptr<mlir::Pass> createLowerToStandardPass();
    /// Registers said pass
    void registerBtorToStandardPass();

} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOSTANDARD_CONVERTBTORTOSTANDARDPASS_H_