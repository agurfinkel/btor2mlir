#include "Conversion/BtorNDToLLVM/ConvertBtorNDToLLVMPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/IR/TypeRange.h"
// #include "mlir/IR/TypeUtilities.h"

#include <string>

using namespace mlir;

#define PASS_NAME "convert-btornd-to-llvm"

namespace {

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct NdBitvectorOpLowering : public ConvertOpToLLVMPattern<btor::NdBitvectorOp> {
  using ConvertOpToLLVMPattern<btor::NdBitvectorOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::NdBitvectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct InputOpLowering : public ConvertOpToLLVMPattern<btor::InputOp> {
  using ConvertOpToLLVMPattern<btor::InputOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::InputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// NdBitvectorOpLowering
//===----------------------------------------------------------------------===//
LogicalResult
NdBitvectorOpLowering::matchAndRewrite(btor::NdBitvectorOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto opType = op.result().getType();
  auto functionType = rewriter.getIntegerType(8);
  // Insert the `havoc` declaration if necessary.
  auto module = op->getParentOfType<ModuleOp>();
  std::string havoc;
  havoc.append("nd_bv");
  auto bvWidth = opType.getIntOrFloatBitWidth();
  if (bvWidth <= 8) {
    havoc.append(std::to_string(8));
  } else if (bvWidth <= 16) {
    havoc.append(std::to_string(16));
    functionType = rewriter.getIntegerType(16);
  } else if (bvWidth <= 32) {
    havoc.append(std::to_string(32));
    functionType = rewriter.getIntegerType(32);
  } else if (bvWidth <= 64) {
    havoc.append(std::to_string(64));
    functionType = rewriter.getIntegerType(64);
  } else if (bvWidth <= 128) {
    havoc.append(std::to_string(128));
    functionType = rewriter.getIntegerType(128);
  } else {
    havoc.append(std::to_string(bvWidth));
    functionType = rewriter.getIntegerType(bvWidth);
  }
  auto havocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(havoc);
  if (!havocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto havocFuncTy =
        LLVM::LLVMFunctionType::get(functionType, {});
    havocFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), havoc, havocFuncTy);
  }

  // don't do the truncation if we have a perfect fit
  if (bvWidth == 8 || bvWidth == 16 || bvWidth == 32 || 
      bvWidth == 64 || bvWidth == 128) {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, havocFunc, llvm::None);
    return success();
  }

  auto callND = rewriter.create<LLVM::CallOp>(op.getLoc(), havocFunc, llvm::None).getResult(0);
  rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, TypeRange({opType}), callND);
  return success();
}

//===----------------------------------------------------------------------===//
// InputOpLowering
//===----------------------------------------------------------------------===//
LogicalResult
InputOpLowering::matchAndRewrite(btor::InputOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<btor::NdBitvectorOp>(op, op.result().getType());
  return success();            
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

struct BtorNDToLLVMLoweringPass
    : public ConvertBtorNDToLLVMBase<BtorNDToLLVMLoweringPass> {

  BtorNDToLLVMLoweringPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  StringRef getArgument() const final { return PASS_NAME; }
  void runOnOperation() override;
};
} // end anonymous namespace

void BtorNDToLLVMLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());

  mlir::btor::populateBTORNDTOLLVMConversionPatterns(converter, patterns);

  target.addIllegalOp<btor::NdBitvectorOp, btor::InputOp>();

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBTORNDTOLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<NdBitvectorOpLowering, InputOpLowering>(converter);
}

/// Create a pass for lowering operations the remaining `Btor` operations
// to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerBtorNDToLLVMPass() {
  return std::make_unique<BtorNDToLLVMLoweringPass>();
}
