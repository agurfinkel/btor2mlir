#include "Conversion/BtorToMemRef/ConvertBtorToMemRefPass.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "../PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Analysis/Liveness.h"

using namespace mlir;
using namespace mlir::btor;

#define PASS_NAME "convert-btor-to-memref"

namespace {
struct BtorToMemRefLoweringPass
    : public PassWrapper<BtorToMemRefLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
  StringRef getArgument() const final { return PASS_NAME; }
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct ArrayOpLowering : public OpRewritePattern<btor::ArrayOp> {
  using OpRewritePattern<mlir::btor::ArrayOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(btor::ArrayOp op, 
                                PatternRewriter &rewriter) const override;
};

struct ReadOpLowering : public OpRewritePattern<btor::ReadOp> {
  using OpRewritePattern<mlir::btor::ReadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(btor::ReadOp op, 
                                PatternRewriter &rewriter) const override;
};

struct WriteOpLowering : public OpRewritePattern<btor::WriteOp> {
  using OpRewritePattern<mlir::btor::WriteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(btor::WriteOp op, 
                                PatternRewriter &rewriter) const override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult
ArrayOpLowering::matchAndRewrite(btor::ArrayOp op, 
                                 PatternRewriter &rewriter) const {
  Type type = op.getType();
  MemRefType memrefType = type.cast<MemRefType>();
  rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType);
  return success();
}

LogicalResult
ReadOpLowering::matchAndRewrite(btor::ReadOp op, 
                                 PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.array(), op.index()); 
  return success();
}

LogicalResult
WriteOpLowering::matchAndRewrite(btor::WriteOp op, 
                                 PatternRewriter &rewriter) const {
  
  Value result = op.result();
  // The btor.write operation has form: result = btor.write %val, %arr[%index]
  // if result is not live after this point, simply map to a memref store
  // else do a copy + write 
  if (result.use_empty()) {
    rewriter.create<memref::StoreOp>(op.getLoc(), op.value(), op.array(), op.index());
    rewriter.eraseOp(op);
    return success();
  }

  Type type = op.getType();
  Location loc = op.getLoc();
  MemRefType memrefType = type.cast<MemRefType>();
  auto newArray = rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefType);
  rewriter.create<memref::CopyOp>(loc, op.array(), newArray);
  rewriter.create<memref::StoreOp>(loc, op.value(), newArray, op.index());
  return success();
}
//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToMemRefConversionPatterns(
  RewritePatternSet &patterns) {
  patterns.add<
    ArrayOpLowering,
    ReadOpLowering, 
    WriteOpLowering
  >(patterns.getContext());
}

void BtorToMemRefLoweringPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateBtorToMemRefConversionPatterns(patterns);
  ConversionTarget target(getContext());
  target.addIllegalOp<btor::ArrayOp, btor::ReadOp, btor::WriteOp>();
  // >();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::btor::createLowerToMemRefPass() {
  return std::make_unique<BtorToMemRefLoweringPass>();
}