#ifndef BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_
#define BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_

#include <memory>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "Dialect/Btor/IR/Btor.h"

namespace mlir {
class BtorToLLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace btor {
    
    /// Collect a set of patterns to lower from btor to LLVM dialect
    void populateBtorToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns);

    /// Creates a pass to convert the Btor dialect into the LLVM dialect.
    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace btor
    class BtorToLLVMTypeConverter : public LLVMTypeConverter {
public:
        BtorToLLVMTypeConverter(MLIRContext *ctx,
                                    const DataLayoutAnalysis *analysis = nullptr)
            : LLVMTypeConverter(ctx, analysis) {
            addConversion([&](btor::BitVecType type) -> llvm::Optional<Type> {
            return convertBtorBitVecType(type);
            });
        }

        Type convertBtorBitVecType(btor::BitVecType type) {
            return ::IntegerType::get(type.getContext(), type.getLength());
        }

    };

} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_