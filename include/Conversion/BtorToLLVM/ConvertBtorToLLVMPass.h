#ifndef BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_
#define BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace btor {
    
    /// Collect a set of patterns to lower from btor to LLVM dialect
    void populateBtorToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns);

    /// Creates a pass to convert the Btor dialect into the LLVM dialect.
    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_