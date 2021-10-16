#include "Conversion/BtorToLLVM/ConvertBtorToLLVMPass.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "../PassDetail.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"

using namespace mlir;

#define PASS_NAME "convert-btor-to-llvm"

namespace {

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

using AddOpLowering = VectorConvertToLLVMPattern<btor::AddOp, LLVM::AddOp>;
using SubOpLowering = VectorConvertToLLVMPattern<btor::SubOp, LLVM::SubOp>;
using MulOpLowering = VectorConvertToLLVMPattern<btor::MulOp, LLVM::MulOp>;
using UDivOpLowering = VectorConvertToLLVMPattern<btor::UDivOp, LLVM::UDivOp>;
using SDivOpLowering = VectorConvertToLLVMPattern<btor::SDivOp, LLVM::SDivOp>;
using URemOpLowering = VectorConvertToLLVMPattern<btor::URemOp, LLVM::URemOp>;
using SRemOpLowering = VectorConvertToLLVMPattern<btor::SRemOp, LLVM::SRemOp>;
using AndOpLowering = VectorConvertToLLVMPattern<btor::AndOp, LLVM::AndOp>;
using OrOpLowering = VectorConvertToLLVMPattern<btor::OrOp, LLVM::OrOp>;
using XOrOpLowering = VectorConvertToLLVMPattern<btor::XOrOp, LLVM::XOrOp>;
using ShiftLLOpLowering = VectorConvertToLLVMPattern<btor::ShiftLLOp, LLVM::ShlOp>;
using ShiftRLOpLowering = VectorConvertToLLVMPattern<btor::ShiftRLOp, LLVM::LShrOp>;
using ShiftRAOpLowering = VectorConvertToLLVMPattern<btor::ShiftRAOp, LLVM::AShrOp>;

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//


} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct BtorToLLVMLoweringPass : 
    public PassWrapper<BtorToLLVMLoweringPass, OperationPass<ModuleOp>> {

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect>();
    }
    StringRef getArgument() const final { return PASS_NAME; }
    void runOnOperation() override;
};
} // end anonymous namespace

void BtorToLLVMLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());

    mlir::btor::populateBtorToLLVMConversionPatterns(converter, patterns);

    /// Configure conversion to lower out btor; Anything else is fine.
    // indexed operators

    /// unary operators
    target.addIllegalOp<btor::NotOp, btor::IncOp, btor::DecOp, btor::NegOp>();
    target.addIllegalOp<btor::BadOp, btor::ConstantOp>();

    /// binary operators
    // logical 
    target.addIllegalOp<btor::CmpOp>();
    target.addIllegalOp<btor::AndOp, btor::NandOp, btor::NorOp, btor::OrOp>();
    target.addIllegalOp<btor::XnorOp, btor::XOrOp, btor::RotateLOp, btor::RotateROp>();
    target.addIllegalOp<btor::ShiftLLOp, btor::ShiftRAOp, btor::ShiftRLOp>();
    // arithmetic
    target.addIllegalOp<btor::AddOp, btor::MulOp, btor::SDivOp, btor::UDivOp>();
    target.addIllegalOp<btor::SModOp, btor::SRemOp, btor::URemOp, btor::SubOp>();
    target.addIllegalOp<btor::SAddOverflowOp, btor::UAddOverflowOp, btor::SDivOverflowOp>();
    target.addIllegalOp<btor::SMulOverflowOp, btor::UMulOverflowOp>();
    target.addIllegalOp<btor::SSubOverflowOp, btor::USubOverflowOp>();

    /// ternary operators
    target.addIllegalOp<btor::IteOp>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<
    AddOpLowering,
    SubOpLowering,
    MulOpLowering,
    UDivOpLowering,
    SDivOpLowering,
    URemOpLowering,
    SRemOpLowering,
    AndOpLowering,
    OrOpLowering,
    XOrOpLowering,
    ShiftLLOpLowering,
    ShiftRLOpLowering,
    ShiftRAOpLowering
  >(converter);       
}

/// Create a pass for lowering operations the remaining `Btor` operations
// to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToLLVMPass() {
    return std::make_unique<BtorToLLVMLoweringPass>(); 
}
