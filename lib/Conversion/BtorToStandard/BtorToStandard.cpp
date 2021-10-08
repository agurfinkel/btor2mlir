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
    LogicalResult matchAndRewrite(mlir::btor::AndOp andOp, PatternRewriter &rewriter) const override;
};

struct XOrLowering : public OpRewritePattern<mlir::btor::XOrOp> {
    using OpRewritePattern<mlir::btor::XOrOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::XOrOp xorOp, PatternRewriter &rewriter) const override;
};

struct BadLowering : public OpRewritePattern<mlir::btor::BadOp> {
    using OpRewritePattern<mlir::btor::BadOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::BadOp mulOp, PatternRewriter &rewriter) const override;
};

struct CmpLowering : public OpRewritePattern<mlir::btor::CmpOp> {
    using OpRewritePattern<mlir::btor::CmpOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::CmpOp cmpOp, PatternRewriter &rewriter) const override;
};

struct NotLowering : public OpRewritePattern<mlir::btor::NotOp> {
    using OpRewritePattern<mlir::btor::NotOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::NotOp notOp, PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult AddLowering::matchAndRewrite(mlir::btor::AddOp addOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::AddIOp>(addOp, addOp.lhs(), addOp.rhs());
    return success();
}

LogicalResult MulLowering::matchAndRewrite(mlir::btor::MulOp mulOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::MulIOp>(mulOp, mulOp.lhs(), mulOp.rhs());
    return success();
}

LogicalResult AndLowering::matchAndRewrite(mlir::btor::AndOp andOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::AndOp>(andOp, andOp.lhs(), andOp.rhs());
    return success();
}

LogicalResult XOrLowering::matchAndRewrite(mlir::btor::XOrOp xorOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::XOrOp>(xorOp, xorOp.lhs(), xorOp.rhs());
    return success();
}

LogicalResult BadLowering::matchAndRewrite(mlir::btor::BadOp badOp, PatternRewriter &rewriter) const {
    Value notBad = rewriter.create<NotOp>(badOp.getLoc(), badOp.arg());
    rewriter.replaceOpWithNewOp<mlir::AssertOp>(badOp, notBad, "Expects argument to be true");
    return success();
}

// Convert btor.cmp predicate into the Standard dialect CmpIOpPredicate.  The two
// enums share the numerical values so we just need to cast.
template <typename StdPredType, typename BtorPredType>
static StdPredType convertBtorCmpPredicate(BtorPredType pred) {
  return static_cast<StdPredType>(pred);
}

LogicalResult CmpLowering::matchAndRewrite(mlir::btor::CmpOp cmpOp, PatternRewriter &rewriter) const {
    auto btorPred = convertBtorCmpPredicate<mlir::CmpIPredicate>(cmpOp.getPredicate());
    rewriter.replaceOpWithNewOp<mlir::CmpIOp>(cmpOp, btorPred, cmpOp.lhs(), cmpOp.rhs());
    return success();
}

LogicalResult NotLowering::matchAndRewrite(mlir::btor::NotOp notOp, PatternRewriter &rewriter) const {
    Value operand = notOp.operand(); 
    Type opType = operand.getType(); 

    int width = opType.getIntOrFloatBitWidth();
    int trueVal = pow(2, width) - 1;
    Value trueConst = rewriter.create<ConstantOp>(notOp.getLoc(), opType, rewriter.getIntegerAttr(opType, trueVal));
    rewriter.replaceOpWithNewOp<mlir::btor::XOrOp>(notOp, operand, trueConst);
    return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AddLowering, MulLowering, AndLowering>(patterns.getContext());
  patterns.add<CmpLowering, BadLowering, NotLowering>(patterns.getContext());
  patterns.add<XOrLowering>(patterns.getContext());
}

void BtorToStandardLoweringPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateBtorToStdConversionPatterns(patterns);
    /// Configure conversion to lower out btor.add; Anything else is fine.
    ConversionTarget target(getContext());
    target.addIllegalOp<mlir::btor::AddOp, mlir::btor::MulOp, mlir::btor::AndOp>();
    target.addIllegalOp<mlir::btor::CmpOp, mlir::btor::BadOp, mlir::btor::NotOp>();
    target.addIllegalOp<mlir::btor::XOrOp>();
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