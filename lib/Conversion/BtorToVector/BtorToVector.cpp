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

struct ArrayOpLowering : public OpConversionPattern<mlir::btor::ArrayOp> {
  using OpConversionPattern<mlir::btor::ArrayOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::ArrayOp arrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct InitArrayLowering : public OpConversionPattern<mlir::btor::InitArrayOp> {
  using OpConversionPattern<mlir::btor::InitArrayOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::InitArrayOp initArrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ReadOpLowering : public OpConversionPattern<mlir::btor::ReadOp> {
  using OpConversionPattern<mlir::btor::ReadOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::ReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct WriteOpLowering : public OpConversionPattern<mlir::btor::WriteOp> {
  using OpConversionPattern<mlir::btor::WriteOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::WriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult
ArrayOpLowering::matchAndRewrite(mlir::btor::ArrayOp arrayOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Value result = rewriter.create<arith::ConstantOp>(
      arrayOp.getLoc(), arrayOp.getType(), rewriter.getZeroAttr(arrayOp.getType()));
  arrayOp.replaceAllUsesWith(result);
  rewriter.replaceOp(arrayOp, result);
  return success();
}

LogicalResult
InitArrayLowering::matchAndRewrite(mlir::btor::InitArrayOp initArrayOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        initArrayOp, initArrayOp.getType(), initArrayOp.init());
  return success();
}

LogicalResult
ReadOpLowering::matchAndRewrite(mlir::btor::ReadOp readOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto resType = readOp.result().getType();
  rewriter.replaceOpWithNewOp<vector::ExtractElementOp>(
      readOp, resType, readOp.base(), readOp.index());
  return success();
}

LogicalResult
WriteOpLowering::matchAndRewrite(mlir::btor::WriteOp writeOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::InsertElementOp>(
      writeOp, writeOp.base().getType(), writeOp.value(),
      writeOp.base(), writeOp.index());
  return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToVectorConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ArrayOpLowering, ReadOpLowering, WriteOpLowering, InitArrayLowering>(patterns.getContext());
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
