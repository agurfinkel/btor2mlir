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

struct AddLowering : public OpRewritePattern<mlir::btor::AddOp> {
    using OpRewritePattern<mlir::btor::AddOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::AddOp addOp, PatternRewriter &rewriter) const override;
};

struct MulLowering : public OpRewritePattern<mlir::btor::MulOp> {
    using OpRewritePattern<mlir::btor::MulOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::MulOp mulOp, PatternRewriter &rewriter) const override;
};

struct AndLowering : public OpRewritePattern<mlir::btor::AndOp> {
    using OpRewritePattern<mlir::btor::AndOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::AndOp mulOp, PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult AddLowering::matchAndRewrite(mlir::btor::AddOp addOp, PatternRewriter &rewriter) const {
    Value addIOp = rewriter.create<mlir::AddIOp>(addOp.getLoc(), addOp.lhs(), addOp.rhs());
    rewriter.replaceOp(addOp, addIOp);
    return success();
}

LogicalResult MulLowering::matchAndRewrite(mlir::btor::MulOp mulOp, PatternRewriter &rewriter) const {
    Value mulIOp = rewriter.create<mlir::MulIOp>(mulOp.getLoc(), mulOp.lhs(), mulOp.rhs());
    rewriter.replaceOp(mulOp, mulIOp);
    return success();
}

LogicalResult AndLowering::matchAndRewrite(mlir::btor::AndOp andOp, PatternRewriter &rewriter) const {
    Value andIOp = rewriter.create<mlir::AndOp>(andOp.getLoc(), andOp.lhs(), andOp.rhs());
    rewriter.replaceOp(andOp, andIOp);
    return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AddLowering, MulLowering, AndLowering>(patterns.getContext());
}

void BtorToStandardLoweringPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateBtorToStdConversionPatterns(patterns);
    /// Configure conversion to lower out btor.add; Anything else is fine.
    ConversionTarget target(getContext());
    target.addIllegalOp<mlir::btor::AddOp, mlir::btor::MulOp, mlir::btor::AndOp>();
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