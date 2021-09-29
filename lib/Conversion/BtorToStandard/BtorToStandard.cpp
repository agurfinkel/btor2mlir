#include "Conversion/BtorToStandard/ConvertBtorToStandardPass.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::btor;

#define PASS_NAME "convert-btor-to-std"

namespace {
struct BtorToStandardLoweringPass : public PassWrapper<BtorToStandardLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<StandardOpsDialect>();
    }
    StringRef getArgument() const final { return PASS_NAME; }
    void runOnOperation() override;
};
} // end anonymous namespace

struct AddLowering : public OpRewritePattern<AddOp> {
    using OpRewritePattern<AddOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(AddOp addOp,PatternRewriter &rewriter) const override;
};

LogicalResult AddLowering::matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const {
    Location loc = addOp.getLoc();
    Value lhs = addOp.lhs();
    Value rhs = addOp.rhs();

    Value addIOp = rewriter.create<mlir::AddIOp>(loc, lhs, rhs);
    rewriter.replaceOp(addOp, addIOp);
    
    return success();
}

void mlir::btor::populateBtorToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AddLowering>(patterns.getContext());
}

void BtorToStandardLoweringPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateBtorToStdConversionPatterns(patterns);
    /// Configure conversion to lower out btor.add; Anything else is fine.
    ConversionTarget target(getContext());
    target.addIllegalOp<AddOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

/// Create a pass for lowering operations the remaining `Btor` operations
// to the Standard dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToStandardPass() {
    return std::make_unique<BtorToStandardLoweringPass>(); 
}

void mlir::btor::registerBtorToStandardPass() {
    PassRegistration<BtorToStandardLoweringPass>();
} 