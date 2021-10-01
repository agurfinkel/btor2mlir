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

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct AddLowering : public OpRewritePattern<AddOp> {
    using OpRewritePattern<AddOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const override;
};

struct MulLowering : public OpRewritePattern<MulOp> {
    using OpRewritePattern<MulOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(MulOp mulOp, PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult AddLowering::matchAndRewrite(AddOp addOp, PatternRewriter &rewriter) const {
    Location loc = addOp.getLoc();
    Value lhs = addOp.lhs();
    Value rhs = addOp.rhs();

    Value addIOp = rewriter.create<mlir::AddIOp>(loc, lhs, rhs);
    rewriter.replaceOp(addOp, addIOp);
    
    return success();
}

LogicalResult MulLowering::matchAndRewrite(MulOp mulOp, PatternRewriter &rewriter) const {
    Location loc = mulOp.getLoc();
    Value lhs = mulOp.lhs();
    Value rhs = mulOp.rhs();

    Value mulIOp = rewriter.create<mlir::MulIOp>(loc, lhs, rhs);
    rewriter.replaceOp(mulOp, mulIOp);
    
    return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AddLowering, MulLowering>(patterns.getContext());
}

void BtorToStandardLoweringPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateBtorToStdConversionPatterns(patterns);
    /// Configure conversion to lower out btor.add; Anything else is fine.
    ConversionTarget target(getContext());
    target.addIllegalOp<AddOp, MulOp>();
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