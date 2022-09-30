#include "Conversion/BtorToArithmetic/ConvertBtorToArithmeticPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::btor;

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct ConstantLowering : public OpRewritePattern<mlir::btor::ConstantOp> {
    using OpRewritePattern<mlir::btor::ConstantOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::ConstantOp constantOp, PatternRewriter &rewriter) const override;
};

struct AddLowering : public OpRewritePattern<mlir::btor::AddOp> {
    using OpRewritePattern<mlir::btor::AddOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::AddOp addOp, PatternRewriter &rewriter) const override;
};

struct SubLowering : public OpRewritePattern<mlir::btor::SubOp> {
    using OpRewritePattern<mlir::btor::SubOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::SubOp subOp, PatternRewriter &rewriter) const override;
};

struct MulLowering : public OpRewritePattern<mlir::btor::MulOp> {
    using OpRewritePattern<mlir::btor::MulOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::MulOp mulOp, PatternRewriter &rewriter) const override;
};

struct SDivLowering : public OpRewritePattern<mlir::btor::SDivOp> {
    using OpRewritePattern<mlir::btor::SDivOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::SDivOp sdivOp, PatternRewriter &rewriter) const override;
};

struct UDivLowering : public OpRewritePattern<mlir::btor::UDivOp> {
    using OpRewritePattern<mlir::btor::UDivOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::UDivOp udivOp, PatternRewriter &rewriter) const override;
};

struct SRemLowering : public OpRewritePattern<mlir::btor::SRemOp> {
    using OpRewritePattern<mlir::btor::SRemOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::SRemOp sremOp, PatternRewriter &rewriter) const override;
};

struct URemLowering : public OpRewritePattern<mlir::btor::URemOp> {
    using OpRewritePattern<mlir::btor::URemOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::URemOp uremOp, PatternRewriter &rewriter) const override;
};

struct AndLowering : public OpRewritePattern<mlir::btor::AndOp> {
    using OpRewritePattern<mlir::btor::AndOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::AndOp andOp, PatternRewriter &rewriter) const override;
};

struct NandLowering : public OpRewritePattern<mlir::btor::NandOp> {
    using OpRewritePattern<mlir::btor::NandOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::NandOp nandOp, PatternRewriter &rewriter) const override;
};

struct OrLowering : public OpRewritePattern<mlir::btor::OrOp> {
    using OpRewritePattern<mlir::btor::OrOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::OrOp orOp, PatternRewriter &rewriter) const override;
};

struct NorLowering : public OpRewritePattern<mlir::btor::NorOp> {
    using OpRewritePattern<mlir::btor::NorOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::NorOp norOp, PatternRewriter &rewriter) const override;
};

struct XOrLowering : public OpRewritePattern<mlir::btor::XOrOp> {
    using OpRewritePattern<mlir::btor::XOrOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::XOrOp xorOp, PatternRewriter &rewriter) const override;
};

struct XnorLowering : public OpRewritePattern<mlir::btor::XnorOp> {
    using OpRewritePattern<mlir::btor::XnorOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::XnorOp xnorOp, PatternRewriter &rewriter) const override;
};

struct ShiftLLLowering : public OpRewritePattern<mlir::btor::ShiftLLOp> {
    using OpRewritePattern<mlir::btor::ShiftLLOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::ShiftLLOp sllOp, PatternRewriter &rewriter) const override;
};

struct ShiftRLLowering : public OpRewritePattern<mlir::btor::ShiftRLOp> {
    using OpRewritePattern<mlir::btor::ShiftRLOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::ShiftRLOp srlOp, PatternRewriter &rewriter) const override;
};

struct ShiftRALowering : public OpRewritePattern<mlir::btor::ShiftRAOp> {
    using OpRewritePattern<mlir::btor::ShiftRAOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::ShiftRAOp sraOp, PatternRewriter &rewriter) const override;
};

struct RotateLLowering : public OpRewritePattern<mlir::btor::RotateLOp> {
    using OpRewritePattern<mlir::btor::RotateLOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::RotateLOp rolOp, PatternRewriter &rewriter) const override;
};

struct RotateRLowering : public OpRewritePattern<mlir::btor::RotateROp> {
    using OpRewritePattern<mlir::btor::RotateROp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::RotateROp rorOp, PatternRewriter &rewriter) const override;
};

struct AssertLowering : public OpRewritePattern<mlir::btor::AssertNotOp> {
    using OpRewritePattern<mlir::btor::AssertNotOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::AssertNotOp mulOp, PatternRewriter &rewriter) const override;
};

struct CmpLowering : public OpRewritePattern<mlir::btor::CmpOp> {
    using OpRewritePattern<mlir::btor::CmpOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::CmpOp cmpOp, PatternRewriter &rewriter) const override;
};

struct NotLowering : public OpRewritePattern<mlir::btor::NotOp> {
    using OpRewritePattern<mlir::btor::NotOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::NotOp notOp, PatternRewriter &rewriter) const override;
};

struct IncLowering : public OpRewritePattern<mlir::btor::IncOp> {
    using OpRewritePattern<mlir::btor::IncOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::IncOp incOp, PatternRewriter &rewriter) const override;
};

struct DecLowering : public OpRewritePattern<mlir::btor::DecOp> {
    using OpRewritePattern<mlir::btor::DecOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::DecOp decOp, PatternRewriter &rewriter) const override;
};

struct NegLowering : public OpRewritePattern<mlir::btor::NegOp> {
    using OpRewritePattern<mlir::btor::NegOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::NegOp negOp, PatternRewriter &rewriter) const override;
};

struct IteLowering : public OpRewritePattern<mlir::btor::IteOp> {
    using OpRewritePattern<mlir::btor::IteOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::IteOp iteOp, PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult AddLowering::matchAndRewrite(mlir::btor::AddOp addOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(addOp, addOp.lhs(), addOp.rhs());
    return success();
}

LogicalResult SubLowering::matchAndRewrite(mlir::btor::SubOp subOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::SubOp>(subOp, subOp.lhs(), subOp.rhs());
    return success();
}

LogicalResult MulLowering::matchAndRewrite(mlir::btor::MulOp mulOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::MulOp>(mulOp, mulOp.lhs(), mulOp.rhs());
    return success();
}

LogicalResult SDivLowering::matchAndRewrite(mlir::btor::SDivOp sdivOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::SDivOp>(sdivOp, sdivOp.lhs(), sdivOp.rhs());
    return success();
}

LogicalResult UDivLowering::matchAndRewrite(mlir::btor::UDivOp udivOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::UDivOp>(udivOp, udivOp.lhs(), udivOp.rhs());
    return success();
}

LogicalResult SRemLowering::matchAndRewrite(mlir::btor::SRemOp sremOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::SRemOp>(sremOp, sremOp.lhs(), sremOp.rhs());
    return success();
}

LogicalResult URemLowering::matchAndRewrite(mlir::btor::URemOp uremOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::URemOp>(uremOp, uremOp.lhs(), uremOp.rhs());
    return success();
}

LogicalResult AndLowering::matchAndRewrite(mlir::btor::AndOp andOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::AndOp>(andOp, andOp.lhs(), andOp.rhs());
    return success();
}

LogicalResult NandLowering::matchAndRewrite(mlir::btor::NandOp nandOp, PatternRewriter &rewriter) const {
    Value andOp = rewriter.create<LLVM::AndOp>(nandOp.getLoc(), nandOp.lhs(), nandOp.rhs());
    rewriter.replaceOpWithNewOp<NotOp>(nandOp, andOp);
    return success();
}

LogicalResult OrLowering::matchAndRewrite(mlir::btor::OrOp orOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::OrOp>(orOp, orOp.lhs(), orOp.rhs());
    return success();
}

LogicalResult NorLowering::matchAndRewrite(mlir::btor::NorOp norOp, PatternRewriter &rewriter) const {
    Value orOp = rewriter.create<LLVM::OrOp>(norOp.getLoc(), norOp.lhs(), norOp.rhs());
    rewriter.replaceOpWithNewOp<NotOp>(norOp, orOp);
    return success();
}

LogicalResult XOrLowering::matchAndRewrite(mlir::btor::XOrOp xorOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(xorOp, xorOp.lhs(), xorOp.rhs());
    return success();
}

LogicalResult XnorLowering::matchAndRewrite(mlir::btor::XnorOp xnorOp, PatternRewriter &rewriter) const {
    Value xorOp = rewriter.create<LLVM::XOrOp>(xnorOp.getLoc(), xnorOp.lhs(), xnorOp.rhs());
    rewriter.replaceOpWithNewOp<NotOp>(xnorOp, xorOp);
    return success();
}

LogicalResult AssertLowering::matchAndRewrite(mlir::btor::AssertNotOp assertOp, PatternRewriter &rewriter) const {
    Value notBad = rewriter.create<NotOp>(assertOp.getLoc(), assertOp.arg());

    auto loc = assertOp.getLoc();

    // Insert the `abort` declaration if necessary.
    auto module = assertOp->getParentOfType<ModuleOp>();
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
        assertOp, notBad, continuationBlock, failureBlock);

    return success();
}

// Convert btor.cmp predicate into the LLVM dialect ICmpOpPredicate.  The two
// enums share the numerical values so we just need to cast.
template <typename CmpIPredicate, typename BtorPredType>
static CmpIPredicate convertBtorCmpPredicate(BtorPredType pred) {
  return static_cast<mlir::arith::CmpIPredicate>(pred);
}

LogicalResult CmpLowering::matchAndRewrite(mlir::btor::CmpOp cmpOp, PatternRewriter &rewriter) const {
    auto btorPred = convertBtorCmpPredicate<arith::CmpIPredicate>(cmpOp.getPredicate());
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmpOp, btorPred, cmpOp.lhs(), cmpOp.rhs());
    return success();
}

LogicalResult NotLowering::matchAndRewrite(mlir::btor::NotOp notOp, PatternRewriter &rewriter) const {
    Value operand = notOp.operand(); 
    Type opType = operand.getType(); 

    int width = opType.getIntOrFloatBitWidth();
    int trueVal = pow(2, width) - 1;
    Value trueConst = rewriter.create<LLVM::ConstantOp>(notOp.getLoc(), opType, rewriter.getIntegerAttr(opType, trueVal));
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(notOp, operand, trueConst);
    return success();
}

LogicalResult IncLowering::matchAndRewrite(mlir::btor::IncOp incOp, PatternRewriter &rewriter) const {
    Value operand = incOp.operand(); 
    Type opType = operand.getType(); 

    Value oneConst = rewriter.create<LLVM::ConstantOp>(incOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(incOp, operand, oneConst);
    return success();
}

LogicalResult DecLowering::matchAndRewrite(mlir::btor::DecOp decOp, PatternRewriter &rewriter) const {
    Value operand = decOp.operand(); 
    Type opType = operand.getType(); 

    Value oneConst = rewriter.create<LLVM::ConstantOp>(decOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
    rewriter.replaceOpWithNewOp<LLVM::SubOp>(decOp, operand, oneConst);
    return success();
}

LogicalResult NegLowering::matchAndRewrite(mlir::btor::NegOp negOp, PatternRewriter &rewriter) const {
    Value operand = negOp.operand(); 
    Type opType = operand.getType(); 

    Value zeroConst = rewriter.create<LLVM::ConstantOp>(negOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 0));
    rewriter.replaceOpWithNewOp<LLVM::SubOp>(negOp, zeroConst, operand);
    return success();
}

LogicalResult ShiftLLLowering::matchAndRewrite(mlir::btor::ShiftLLOp sllOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::ShlOp>(sllOp, sllOp.lhs(), sllOp.rhs());
    return success();
}

LogicalResult ShiftRLLowering::matchAndRewrite(mlir::btor::ShiftRLOp srlOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::LShrOp>(srlOp, srlOp.lhs(), srlOp.rhs());
    return success();
}

LogicalResult ShiftRALowering::matchAndRewrite(mlir::btor::ShiftRAOp sraOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::AShrOp>(sraOp, sraOp.lhs(), sraOp.rhs());
    return success();
}

LogicalResult IteLowering::matchAndRewrite(mlir::btor::IteOp iteOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
        iteOp, 
        iteOp.condition(), iteOp.true_value(), iteOp.false_value());
    return success();
}

LogicalResult RotateLLowering::matchAndRewrite(mlir::btor::RotateLOp rolOp, PatternRewriter &rewriter) const {
    // We convert using the following paradigm: given lhs, rhs, width
    // shiftBy = rhs % width
    // (lhs << shiftBy) or (lhs >> (width - shiftBy))   
    Location loc = rolOp.getLoc();
    Value lhs = rolOp.lhs();
    Value rhs = rolOp.rhs();
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

LogicalResult RotateRLowering::matchAndRewrite(mlir::btor::RotateROp rorOp, PatternRewriter &rewriter) const {
    // We convert using the following paradigm: given lhs, rhs, width
    // shiftBy = rhs % width
    // (lhs >> shiftBy) or (lhs << (width - shiftBy))   
    Location loc = rorOp.getLoc();
    Value lhs = rorOp.lhs();
    Value rhs = rorOp.rhs();
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

LogicalResult ConstantLowering::matchAndRewrite(mlir::btor::ConstantOp constOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(constOp, constOp.getType(), constOp.value());
    return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToArithmeticConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AddLowering, MulLowering, AndLowering>(patterns.getContext());
  patterns.add<CmpLowering, AssertLowering, NotLowering>(patterns.getContext());
  patterns.add<XOrLowering, XnorLowering, NandLowering>(patterns.getContext());
  patterns.add<OrLowering, NorLowering, IncLowering>(patterns.getContext());
  patterns.add<DecLowering, DecLowering, SRemLowering>(patterns.getContext());
  patterns.add<URemLowering, ShiftLLLowering, ShiftRLLowering>(patterns.getContext());
  patterns.add<ShiftRALowering, UDivLowering, SDivLowering>(patterns.getContext());
  patterns.add<NegLowering, IteLowering, RotateRLowering>(patterns.getContext());
  patterns.add<RotateLLowering, ConstantLowering>(patterns.getContext());
}

namespace {
struct ConvertBtorToArithmeticPass : public ConvertBtorToArithmeticBase<ConvertBtorToArithmeticPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        populateBtorToArithmeticConversionPatterns(patterns);
        /// Configure conversion to lower out btor.add; Anything else is fine.
        ConversionTarget target(getContext());
        target.addIllegalOp<mlir::btor::AddOp, mlir::btor::MulOp, mlir::btor::AndOp>();
        target.addIllegalOp<mlir::btor::CmpOp, mlir::btor::AssertNotOp, mlir::btor::NotOp>();
        target.addIllegalOp<mlir::btor::XOrOp, mlir::btor::XnorOp, mlir::btor::NandOp>();
        target.addIllegalOp<mlir::btor::OrOp, mlir::btor::NorOp, mlir::btor::IncOp>();
        target.addIllegalOp<mlir::btor::DecOp, mlir::btor::SubOp, mlir::btor::SRemOp>();
        target.addIllegalOp<mlir::btor::URemOp, mlir::btor::ShiftLLOp, mlir::btor::ShiftRLOp>();
        target.addIllegalOp<mlir::btor::UDivOp, mlir::btor::SDivOp, mlir::btor::ShiftRAOp>();
        target.addIllegalOp<mlir::btor::NegOp, mlir::btor::IteOp, mlir::btor::RotateLOp>();
        target.addIllegalOp<mlir::btor::RotateROp, mlir::btor::ConstantOp>();
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
} // end anonymous namespace

/// Create a pass for lowering operations the remaining `Btor` operations
// to the Math dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createConvertBtorToArithmeticPass() {
    return std::make_unique<ConvertBtorToArithmeticPass>(); 
}
