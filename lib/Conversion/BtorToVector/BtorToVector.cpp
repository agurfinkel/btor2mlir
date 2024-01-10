#include "Conversion/BtorToVector/ConvertBtorToVectorPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::btor;

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct InitArrayLowering
    : public ConvertOpToLLVMPattern<mlir::btor::InitArrayOp> {
  using ConvertOpToLLVMPattern<mlir::btor::InitArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::InitArrayOp initArrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ReadOpLowering : public ConvertOpToLLVMPattern<mlir::btor::ReadOp> {
  using ConvertOpToLLVMPattern<mlir::btor::ReadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::ReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct WriteOpLowering : public ConvertOpToLLVMPattern<mlir::btor::WriteOp> {
  using ConvertOpToLLVMPattern<mlir::btor::WriteOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::WriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorInitArrayOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::VectorInitArrayOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::VectorInitArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::VectorInitArrayOp vecInitArrayOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorReadOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::VectorReadOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::VectorReadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::VectorReadOp vecReadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct VectorWriteOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::VectorWriteOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::VectorWriteOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::VectorWriteOp vecWriteOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult
InitArrayLowering::matchAndRewrite(mlir::btor::InitArrayOp initArrayOp,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto arrayType = typeConverter->convertType(initArrayOp.getType());
  rewriter.replaceOpWithNewOp<mlir::btor::VectorInitArrayOp>(
      initArrayOp, arrayType, adaptor.init());
  return success();
}

LogicalResult
ReadOpLowering::matchAndRewrite(mlir::btor::ReadOp readOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  auto resType = typeConverter->convertType(readOp.result().getType());
  rewriter.replaceOpWithNewOp<mlir::btor::VectorReadOp>(
      readOp, resType, adaptor.base(), adaptor.index());
  return success();
}

LogicalResult
WriteOpLowering::matchAndRewrite(mlir::btor::WriteOp writeOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto resType = typeConverter->convertType(writeOp.base().getType());
  rewriter.replaceOpWithNewOp<mlir::btor::VectorWriteOp>(
    writeOp, resType, adaptor.value(), adaptor.base(), adaptor.index());
  return success();
}

LogicalResult VectorInitArrayOpLowering::matchAndRewrite(
    mlir::btor::VectorInitArrayOp vecInitArrayOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
      vecInitArrayOp, vecInitArrayOp.getType(), vecInitArrayOp.init());
  return success();
}

LogicalResult VectorReadOpLowering::matchAndRewrite(
    mlir::btor::VectorReadOp vecReadOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::ExtractElementOp>(
      vecReadOp, vecReadOp.result().getType(), adaptor.base(), adaptor.index());
  return success();
}

LogicalResult VectorWriteOpLowering::matchAndRewrite(
    mlir::btor::VectorWriteOp vecWriteOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<vector::InsertElementOp>(
      vecWriteOp, vecWriteOp.base().getType(), adaptor.value(), adaptor.base(),
      adaptor.index());
  return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToVectorConversionPatterns(
    BtorToLLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ReadOpLowering, WriteOpLowering, InitArrayLowering,
               VectorInitArrayOpLowering, VectorReadOpLowering,
               VectorWriteOpLowering>(converter);
}

namespace {
struct ConvertBtorToVectorPass
    : public ConvertBtorToVectorBase<ConvertBtorToVectorPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    BtorToLLVMTypeConverter converter(&getContext());

    mlir::btor::populateBtorToVectorConversionPatterns(converter, patterns);
    mlir::populateStdToLLVMConversionPatterns(converter, patterns);
    /// Configure conversion to lower out btor; Anything else is fine.
    // init operators
    target.addIllegalOp<btor::InitArrayOp, btor::VectorInitArrayOp>();

    /// indexed operators
    target.addIllegalOp<btor::ReadOp, btor::VectorReadOp>();
    target.addIllegalOp<btor::WriteOp, btor::VectorWriteOp>();

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
