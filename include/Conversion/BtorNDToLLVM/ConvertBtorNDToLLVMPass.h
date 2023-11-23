#ifndef BTOR_CONVERSION_BTORNDTOLLVM_CONVERTBTORNDTOLLVMPASS_H_
#define BTOR_CONVERSION_BTORNDTOLLVM_CONVERTBTORNDTOLLVMPASS_H_

#include <memory>

#include "Conversion/BtorToLLVM/ConvertBtorToLLVMPass.h"

namespace mlir {
class BtorToLLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace btor {
    
    /// Collect a set of patterns to lower from btor nd operations to LLVM dialect
    void populateBTORNDTOLLVMConversionPatterns(BtorToLLVMTypeConverter &converter,
                                                RewritePatternSet &patterns);

    /// Creates a pass to convert the Btor dialect nd bitvector operations into the LLVM dialect.
    std::unique_ptr<mlir::Pass> createLowerBtorNDToLLVMPass();

} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_BTORNDTOLLVM_CONVERTBTORNDTOLLVMPASS_H_