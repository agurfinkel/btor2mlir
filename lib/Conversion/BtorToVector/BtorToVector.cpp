#include "Conversion/BtorToVector/ConvertBtorToVectorPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::btor;

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct ArrayOpLowering : public OpRewritePattern<mlir::btor::ArrayOp> {
  using OpRewritePattern<mlir::btor::ArrayOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::ArrayOp arrayOp,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult ArrayOpLowering::matchAndRewrite(mlir::btor::ArrayOp arrayOp,
                                           PatternRewriter &rewriter) const {
  VectorType resVectorType = arrayOp.getArrayType();
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(arrayOp, resVectorType, rewriter.getZeroAttr(resVectorType));
  return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToVectorConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ArrayOpLowering>(patterns.getContext());
}

namespace {
struct ConvertBtorToVectorPass
    : public ConvertBtorToVectorBase<ConvertBtorToVectorPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBtorToVectorConversionPatterns(patterns);
    ConversionTarget target(getContext());

    /// Configure conversion to lower out btor; Anything else is fine.
    // init operators
    target.addIllegalOp<btor::ArrayOp, btor::InitArrayOp>();
    // target.addIllegalOp<btor::ArrayType>();

    /// indexed operators
    // target.addIllegalOp<btor::ReadOp, btor::WriteOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

/// Create a pass for lowering operations the remaining `Btor` operations
// to the Vector dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToVectorPass() {
  return std::make_unique<ConvertBtorToVectorPass>();
}
