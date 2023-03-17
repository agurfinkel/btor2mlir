#include "Conversion/BtorNDToLLVM/ConvertBtorNDToLLVMPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <string>

using namespace mlir;

#define PASS_NAME "convert-btornd-to-llvm"

namespace {

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct NDStateOpLowering : public ConvertOpToLLVMPattern<btor::NDStateOp> {
  using ConvertOpToLLVMPattern<btor::NDStateOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::NDStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct InputOpLowering : public ConvertOpToLLVMPattern<btor::InputOp> {
  using ConvertOpToLLVMPattern<btor::InputOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::InputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

template <typename Op>
std::string createNDFunctionHelper(Op op, std::string suffix, mlir::ConversionPatternRewriter &rewriter) {
  auto opType = op.result().getType();
  auto functionType = rewriter.getIntegerType(8);
  // Insert the `havoc` declaration if necessary.
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
  havoc.append(suffix);
  havoc.append(std::to_string(op.idAttr().getInt()));
  return havoc;
}

template <typename Op>
IntegerType createFunctionTypeHelper(Op op, mlir::ConversionPatternRewriter &rewriter) {
  mlir::IntegerType functionType;

  auto opType = op.result().getType();
  auto bvWidth = opType.getIntOrFloatBitWidth();

  if (bvWidth <= 8) {
    functionType = rewriter.getIntegerType(8);
  } else if (bvWidth <= 16) {
    functionType = rewriter.getIntegerType(16);
  } else if (bvWidth <= 32) {
    functionType = rewriter.getIntegerType(32);
  } else if (bvWidth <= 64) {
    functionType = rewriter.getIntegerType(64);
  } else if (bvWidth <= 128) {
    functionType = rewriter.getIntegerType(128);
  } else {
    functionType = rewriter.getIntegerType(bvWidth);
  }
  return functionType;
}

template <typename Op>
void createPrintFunctionHelper(
          Op op, std::string printHelper,
          mlir::ConversionPatternRewriter &rewriter,
          ModuleOp module) {
  auto printFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(printHelper);
  assert(!printFunc);
  if (!printFunc) {
    OpBuilder::InsertionGuard printerGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    assert(insert == rewriter.getInsertionPoint());
    auto printFuncTy = LLVM::LLVMFunctionType::get(
          LLVM::LLVMVoidType::get(rewriter.getContext()), rewriter.getI64Type());
    printFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), printHelper, printFuncTy);
  }
  Value inputId = rewriter.create<LLVM::ConstantOp>(
      op.getLoc(), rewriter.getI64Type(), op.idAttr());
  rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), printFunc, inputId);
}
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// NDStateOpLowering
//===----------------------------------------------------------------------===//
LogicalResult
NDStateOpLowering::matchAndRewrite(btor::NDStateOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto opType = op.result().getType();
  auto module = op->getParentOfType<ModuleOp>();
  auto bvWidth = opType.getIntOrFloatBitWidth();

  std::string havoc = createNDFunctionHelper(op, std::string("_st"), rewriter);
  auto havocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(havoc);
  if (!havocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto havocFuncTy =
        LLVM::LLVMFunctionType::get(createFunctionTypeHelper(op, rewriter), {});
    havocFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), havoc, havocFuncTy);
  }
  // add helper function for printing
  std::string printHelper = "btor2mlir_print_state_num";
  createPrintFunctionHelper(op, printHelper, rewriter, module);
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
  auto opType = op.result().getType();
  auto module = op->getParentOfType<ModuleOp>();
  auto bvWidth = opType.getIntOrFloatBitWidth();
  // create the nd function of name: nd_bvX_inputY
  std::string havoc = createNDFunctionHelper(op, std::string("_in"), rewriter);
  auto havocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(havoc);
  if (!havocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    assert(insert == rewriter.getInsertionPoint());
    auto havocFuncTy =
        LLVM::LLVMFunctionType::get(createFunctionTypeHelper(op, rewriter), {});
    havocFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), havoc, havocFuncTy);
  }
  // add helper function for printing
  std::string printHelper = "btor2mlir_print_input_num";
  createPrintFunctionHelper(op, printHelper, rewriter, module);
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

  target.addIllegalOp<btor::NDStateOp, btor::InputOp>();

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
  patterns.add<NDStateOpLowering, InputOpLowering>(converter);
}

/// Create a pass for lowering operations the remaining `Btor` operations
// to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerBtorNDToLLVMPass() {
  return std::make_unique<BtorNDToLLVMLoweringPass>();
}
