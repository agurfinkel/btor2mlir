#include "Conversion/BtorToLLVM/ConvertBtorToLLVMPass.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "../PassDetail.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"

using namespace mlir;

#define PASS_NAME "convert-btor-to-llvm"

namespace {

template <typename SourceOp, typename BaseOp>
class ConvertNotOpToBtorPattern : public ConvertOpToLLVMPattern<SourceOp> {
    public:
        using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
        // using Super = VectorConvertToLLVMPattern<SourceOp, TargetOp>;

        LogicalResult matchAndRewrite(SourceOp op, 
                    typename SourceOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
            static_assert(
                std::is_base_of<
                        OpTrait::OneResult<SourceOp>, SourceOp>::value,
                        "expected single result op");

            Value baseOp = rewriter.create<BaseOp>(op.getLoc(),
                                     adaptor.lhs(), adaptor.rhs());
            rewriter.replaceOpWithNewOp<btor::NotOp>(op, baseOp);

            return success();
    // return LLVM::detail::vectorOneToOneRewrite(
    //     op, TargetOp::getOperationName(), adaptor.getOperands(),
    //     *this->getTypeConverter(), rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

using AddOpLowering = VectorConvertToLLVMPattern<btor::AddOp, LLVM::AddOp>;
using SubOpLowering = VectorConvertToLLVMPattern<btor::SubOp, LLVM::SubOp>;
using MulOpLowering = VectorConvertToLLVMPattern<btor::MulOp, LLVM::MulOp>;
using UDivOpLowering = VectorConvertToLLVMPattern<btor::UDivOp, LLVM::UDivOp>;
using SDivOpLowering = VectorConvertToLLVMPattern<btor::SDivOp, LLVM::SDivOp>;
using URemOpLowering = VectorConvertToLLVMPattern<btor::URemOp, LLVM::URemOp>;
using SRemOpLowering = VectorConvertToLLVMPattern<btor::SRemOp, LLVM::SRemOp>;
using AndOpLowering = VectorConvertToLLVMPattern<btor::AndOp, LLVM::AndOp>;
using OrOpLowering = VectorConvertToLLVMPattern<btor::OrOp, LLVM::OrOp>;
using XOrOpLowering = VectorConvertToLLVMPattern<btor::XOrOp, LLVM::XOrOp>;
using ShiftLLOpLowering = VectorConvertToLLVMPattern<btor::ShiftLLOp, LLVM::ShlOp>;
using ShiftRLOpLowering = VectorConvertToLLVMPattern<btor::ShiftRLOp, LLVM::LShrOp>;
using ShiftRAOpLowering = VectorConvertToLLVMPattern<btor::ShiftRAOp, LLVM::AShrOp>;
using UAddOverflowOpLowering = 
      VectorConvertToLLVMPattern<btor::UAddOverflowOp, LLVM::UAddWithOverflowOp>;
using SAddOverflowOpLowering = 
      VectorConvertToLLVMPattern<btor::SAddOverflowOp, LLVM::SAddWithOverflowOp>;
using USubOverflowOpLowering = 
      VectorConvertToLLVMPattern<btor::USubOverflowOp, LLVM::USubWithOverflowOp>;
using SSubOverflowOpLowering = 
      VectorConvertToLLVMPattern<btor::SSubOverflowOp, LLVM::SSubWithOverflowOp>;
using SMulOverflowOpLowering = 
      VectorConvertToLLVMPattern<btor::SMulOverflowOp, LLVM::SMulWithOverflowOp>;
using UMulOverflowOpLowering = 
      VectorConvertToLLVMPattern<btor::UMulOverflowOp, LLVM::UMulWithOverflowOp>;
// using SDivOverflowOpLowering = 
//       VectorConvertToLLVMPattern<btor::SDivOverflowOp, LLVM::SDivWithOverflowOp>;
using UExtOpLowering = VectorConvertToLLVMPattern<btor::UExtOp, LLVM::ZExtOp>;
using SExtOpLowering = VectorConvertToLLVMPattern<btor::SExtOp, LLVM::SExtOp>;
using IteOpLowering = VectorConvertToLLVMPattern<btor::IteOp, LLVM::SelectOp>;
using XnorOpLowering = ConvertNotOpToBtorPattern<btor::XnorOp, btor::XOrOp>;
using NandOpLowering = ConvertNotOpToBtorPattern<btor::NandOp, btor::AndOp>;
using NorOpLowering = ConvertNotOpToBtorPattern<btor::NorOp, btor::OrOp>;

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public ConvertOpToLLVMPattern<btor::ConstantOp> {
    using ConvertOpToLLVMPattern<btor::ConstantOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};

struct CmpOpLowering : public ConvertOpToLLVMPattern<btor::CmpOp> {
    using ConvertOpToLLVMPattern<btor::CmpOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct NotOpLowering : public ConvertOpToLLVMPattern<btor::NotOp> {
    using ConvertOpToLLVMPattern<btor::NotOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct BadOpLowering : public ConvertOpToLLVMPattern<btor::BadOp> {
    using ConvertOpToLLVMPattern<btor::BadOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::BadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct IffOpLowering : public ConvertOpToLLVMPattern<btor::IffOp> {
    using ConvertOpToLLVMPattern<btor::IffOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::IffOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ImpliesOpLowering : public ConvertOpToLLVMPattern<btor::ImpliesOp> {
    using ConvertOpToLLVMPattern<btor::ImpliesOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::ImpliesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct RotateLOpLowering : public ConvertOpToLLVMPattern<btor::RotateLOp> {
    using ConvertOpToLLVMPattern<btor::RotateLOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::RotateLOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct RotateROpLowering : public ConvertOpToLLVMPattern<btor::RotateROp> {
    using ConvertOpToLLVMPattern<btor::RotateROp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::RotateROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct IncOpLowering : public ConvertOpToLLVMPattern<btor::IncOp> {
    using ConvertOpToLLVMPattern<btor::IncOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::IncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};

struct DecOpLowering : public ConvertOpToLLVMPattern<btor::DecOp> {
    using ConvertOpToLLVMPattern<btor::DecOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::DecOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};

struct NegOpLowering : public ConvertOpToLLVMPattern<btor::NegOp> {
    using ConvertOpToLLVMPattern<btor::NegOp>::ConvertOpToLLVMPattern;
    LogicalResult matchAndRewrite(btor::NegOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConstantOpLowering
//===----------------------------------------------------------------------===//

LogicalResult ConstantOpLowering::matchAndRewrite(btor::ConstantOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    return LLVM::detail::oneToOneRewrite(op, LLVM::ConstantOp::getOperationName(),
                                       adaptor.getOperands(),
                                       *getTypeConverter(), rewriter);
}

//===----------------------------------------------------------------------===//
// CmpOpLowering
//===----------------------------------------------------------------------===//

// Convert btor.cmp predicate into the LLVM dialect CmpPredicate. The two enums
// share numerical values so just cast.
template <typename LLVMPredType, typename PredType>
static LLVMPredType convertBtorCmpPredicate(PredType pred) {
  return static_cast<LLVMPredType>(pred);
}

LogicalResult CmpOpLowering::matchAndRewrite(btor::CmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto resultType = op.getResult().getType();

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, 
                            typeConverter->convertType(resultType),
                            convertBtorCmpPredicate<LLVM::ICmpPredicate>(op.getPredicate()),
                            adaptor.lhs(), adaptor.rhs());

  return success();
}

//===----------------------------------------------------------------------===//
// NotOpLowering
//===----------------------------------------------------------------------===//

LogicalResult NotOpLowering::matchAndRewrite(btor::NotOp notOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    Value operand = adaptor.operand(); 
    Type opType = operand.getType(); 

    int width = opType.getIntOrFloatBitWidth();
    int trueVal = pow(2, width) - 1;
    Value trueConst = rewriter.create<LLVM::ConstantOp>(notOp.getLoc(), opType, rewriter.getIntegerAttr(opType, trueVal));
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(notOp, operand, trueConst);
    return success();
}

//===----------------------------------------------------------------------===//
// BadOpLowering
//===----------------------------------------------------------------------===//

LogicalResult BadOpLowering::matchAndRewrite(btor::BadOp badOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {

    auto loc = badOp.getLoc();

    Value notBad = rewriter.create<btor::NotOp>(loc, adaptor.arg());

    // Insert the `abort` declaration if necessary.
    auto module = badOp->getParentOfType<ModuleOp>();
    auto abortFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("abort");
    if (!abortFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto abortFuncTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()), {});
      abortFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                    "abort", abortFuncTy);
    }

    // Split block at `assert` operation.
    Block *opBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

    // Generate IR to call `abort`.
    Block *failureBlock = rewriter.createBlock(opBlock->getParent());
    rewriter.create<LLVM::CallOp>(loc, abortFunc, llvm::None);
    rewriter.create<LLVM::UnreachableOp>(loc);

    // Generate assertion test.
    rewriter.setInsertionPointToEnd(opBlock);
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        badOp, notBad, continuationBlock, failureBlock);

    return success();
}

//===----------------------------------------------------------------------===//
// IffOpLowering
//===----------------------------------------------------------------------===//

LogicalResult IffOpLowering::matchAndRewrite(btor::IffOp iffOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    auto loc = iffOp.getLoc();

    Value notLHS = rewriter.create<btor::NotOp>(loc, adaptor.lhs());
    Value notRHS = rewriter.create<btor::NotOp>(loc, adaptor.rhs());
    
    Value notLHSorRHS = rewriter.create<LLVM::OrOp>(loc, notLHS, adaptor.rhs());
    Value notRHSorLHS = rewriter.create<LLVM::OrOp>(loc, notRHS, adaptor.lhs());
    rewriter.replaceOpWithNewOp<LLVM::AndOp>(iffOp, notLHSorRHS, notRHSorLHS);
    return success();
}

//===----------------------------------------------------------------------===//
// ImpliesOpLowering
//===----------------------------------------------------------------------===//

LogicalResult ImpliesOpLowering::matchAndRewrite(btor::ImpliesOp impOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    auto loc = impOp.getLoc();

    Value notLHS = rewriter.create<btor::NotOp>(loc, adaptor.lhs());
    rewriter.replaceOpWithNewOp<LLVM::OrOp>(impOp, notLHS, adaptor.rhs());
    return success();
}

//===----------------------------------------------------------------------===//
// RotateLOpLowering
//===----------------------------------------------------------------------===//

LogicalResult RotateLOpLowering::matchAndRewrite(btor::RotateLOp rolOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    // We convert using the following paradigm: given lhs, rhs, width
    // shiftBy = rhs % width
    // (lhs << shiftBy) or (lhs >> (width - shiftBy))   
    auto loc = rolOp.getLoc();
    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type opType = lhs.getType(); 

    int width = opType.getIntOrFloatBitWidth();
    Value widthVal = rewriter.create<LLVM::ConstantOp>(loc, opType, rewriter.getIntegerAttr(opType, width));
    Value shiftBy = rewriter.create<LLVM::URemOp>(loc, rhs, widthVal);
    Value shiftRightBy = rewriter.create<LLVM::SubOp>(loc, widthVal, shiftBy);

    Value leftValue = rewriter.create<LLVM::ShlOp>(loc, lhs, shiftBy);
    Value rightValue = rewriter.create<LLVM::LShrOp>(loc, lhs, shiftRightBy);

    rewriter.replaceOpWithNewOp<LLVM::OrOp>(rolOp, leftValue, rightValue);
    return success();
}

//===----------------------------------------------------------------------===//
// RotateROpLowering
//===----------------------------------------------------------------------===//

LogicalResult RotateROpLowering::matchAndRewrite(btor::RotateROp rorOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    // We convert using the following paradigm: given lhs, rhs, width
    // shiftBy = rhs % width
    // (lhs >> shiftBy) or (lhs << (width - shiftBy))   
    Location loc = rorOp.getLoc();
    Value lhs = adaptor.lhs();
    Value rhs = adaptor.rhs();
    Type opType = lhs.getType(); 

    int width = opType.getIntOrFloatBitWidth();
    Value widthVal = rewriter.create<LLVM::ConstantOp>(loc, opType, rewriter.getIntegerAttr(opType, width));
    Value shiftBy = rewriter.create<LLVM::URemOp>(loc, rhs, widthVal);
    Value shiftLeftBy = rewriter.create<LLVM::SubOp>(loc, widthVal, shiftBy);

    Value leftValue = rewriter.create<LLVM::LShrOp>(loc, lhs, shiftBy);
    Value rightValue = rewriter.create<LLVM::ShlOp>(loc, lhs, shiftLeftBy);

    rewriter.replaceOpWithNewOp<LLVM::OrOp>(rorOp, leftValue, rightValue);
    return success();
}

//===----------------------------------------------------------------------===//
// IncOpLowering
//===----------------------------------------------------------------------===//

LogicalResult IncOpLowering::matchAndRewrite(mlir::btor::IncOp incOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    Value operand = adaptor.operand(); 
    Type opType = operand.getType(); 

    Value oneConst = rewriter.create<LLVM::ConstantOp>(incOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(incOp, operand, oneConst);
    return success();
}

//===----------------------------------------------------------------------===//
// DecOpLowering
//===----------------------------------------------------------------------===//

LogicalResult DecOpLowering::matchAndRewrite(mlir::btor::DecOp decOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    Value operand = adaptor.operand(); 
    Type opType = operand.getType(); 

    Value oneConst = rewriter.create<LLVM::ConstantOp>(decOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
    rewriter.replaceOpWithNewOp<LLVM::SubOp>(decOp, operand, oneConst);
    return success();
}

//===----------------------------------------------------------------------===//
// NegOpLowering
//===----------------------------------------------------------------------===//

LogicalResult NegOpLowering::matchAndRewrite(mlir::btor::NegOp negOp, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
    Value operand = adaptor.operand(); 
    Type opType = operand.getType(); 

    Value zeroConst = rewriter.create<LLVM::ConstantOp>(negOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 0));
    rewriter.replaceOpWithNewOp<LLVM::SubOp>(negOp, zeroConst, operand);
    return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

// struct BtorToLLVMLoweringPass : 
//     public PassWrapper<BtorToLLVMLoweringPass, OperationPass<ModuleOp>> {
struct BtorToLLVMLoweringPass
    : public ConvertBtorToLLVMBase<BtorToLLVMLoweringPass> {
        
    BtorToLLVMLoweringPass() = default;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect>();
    }
    StringRef getArgument() const final { return PASS_NAME; }
    void runOnOperation() override;
};
} // end anonymous namespace

void BtorToLLVMLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());

    mlir::btor::populateBtorToLLVMConversionPatterns(converter, patterns);

    /// Configure conversion to lower out btor; Anything else is fine.
    // indexed operators
    target.addIllegalOp<btor::UExtOp, btor::SExtOp>();

    /// unary operators
    target.addIllegalOp<btor::NotOp, btor::IncOp, btor::DecOp, btor::NegOp>();
    target.addIllegalOp<btor::BadOp, btor::ConstantOp>();

    /// binary operators
    // logical 
    target.addIllegalOp<btor::IffOp, btor::ImpliesOp, btor::CmpOp>();
    target.addIllegalOp<btor::AndOp, btor::NandOp, btor::NorOp, btor::OrOp>();
    target.addIllegalOp<btor::XnorOp, btor::XOrOp, btor::RotateLOp, btor::RotateROp>();
    target.addIllegalOp<btor::ShiftLLOp, btor::ShiftRAOp, btor::ShiftRLOp>();
    // arithmetic
    target.addIllegalOp<btor::AddOp, btor::MulOp, btor::SDivOp, btor::UDivOp>();
    target.addIllegalOp<btor::SModOp, btor::SRemOp, btor::URemOp, btor::SubOp>(); // srem, urem, sub
    target.addIllegalOp<btor::SAddOverflowOp, btor::UAddOverflowOp, btor::SDivOverflowOp>(); // saddo, uaddo
    target.addIllegalOp<btor::SMulOverflowOp, btor::UMulOverflowOp>();
    target.addIllegalOp<btor::SSubOverflowOp, btor::USubOverflowOp>();

    /// ternary operators
    target.addIllegalOp<btor::IteOp>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<
    ConstantOpLowering,
    AddOpLowering,
    SubOpLowering,
    MulOpLowering,
    UDivOpLowering,
    SDivOpLowering,
    URemOpLowering,
    SRemOpLowering,
    AndOpLowering,
    OrOpLowering,
    XOrOpLowering,
    ShiftLLOpLowering,
    ShiftRLOpLowering,
    ShiftRAOpLowering,
    RotateLOpLowering,
    RotateROpLowering,
    CmpOpLowering,
    SAddOverflowOpLowering,
    UAddOverflowOpLowering,
    SSubOverflowOpLowering,
    USubOverflowOpLowering,
    SMulOverflowOpLowering,
    UMulOverflowOpLowering,
    NotOpLowering,
    BadOpLowering,
    IteOpLowering,
    IffOpLowering,
    ImpliesOpLowering,
    XnorOpLowering,
    NandOpLowering,
    NorOpLowering,
    IncOpLowering,
    DecOpLowering,
    NegOpLowering,
    UExtOpLowering,
    SExtOpLowering
  >(converter);       
}

/// Create a pass for lowering operations the remaining `Btor` operations
// to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToLLVMPass() {
    return std::make_unique<BtorToLLVMLoweringPass>(); 
}
