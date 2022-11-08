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

// struct AddLowering : public OpRewritePattern<mlir::btor::AddOp> {
//   using OpRewritePattern<mlir::btor::AddOp>::OpRewritePattern;
//   LogicalResult matchAndRewrite(mlir::btor::AddOp addOp,
//                                 PatternRewriter &rewriter) const override;
// };

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

// LogicalResult AddLowering::matchAndRewrite(mlir::btor::AddOp addOp,
//                                            PatternRewriter &rewriter) const {
//   rewriter.replaceOpWithNewOp<arith::AddIOp>(addOp, addOp.lhs(), addOp.rhs());
//   return success();
// }

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToVectorConversionPatterns(
    RewritePatternSet &patterns) {
//   patterns.add<>(patterns.getContext());
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

    /// indexed operators
    target.addIllegalOp<btor::ReadOp, btor::WriteOp>();

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
