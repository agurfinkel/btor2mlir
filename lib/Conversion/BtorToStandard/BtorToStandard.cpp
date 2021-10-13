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

struct IncLowering : public OpRewritePattern<mlir::btor::IncOp> {
    using OpRewritePattern<mlir::btor::IncOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::IncOp incOp, PatternRewriter &rewriter) const override;
};

struct DecLowering : public OpRewritePattern<mlir::btor::DecOp> {
    using OpRewritePattern<mlir::btor::DecOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(mlir::btor::DecOp decOp, PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult AddLowering::matchAndRewrite(mlir::btor::AddOp addOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::AddIOp>(addOp, addOp.lhs(), addOp.rhs());
    return success();
}

LogicalResult SubLowering::matchAndRewrite(mlir::btor::SubOp subOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::SubIOp>(subOp, subOp.lhs(), subOp.rhs());
    return success();
}

LogicalResult MulLowering::matchAndRewrite(mlir::btor::MulOp mulOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::MulIOp>(mulOp, mulOp.lhs(), mulOp.rhs());
    return success();
}

LogicalResult SDivLowering::matchAndRewrite(mlir::btor::SDivOp sdivOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::SignedDivIOp>(sdivOp, sdivOp.lhs(), sdivOp.rhs());
    return success();
}

LogicalResult UDivLowering::matchAndRewrite(mlir::btor::UDivOp udivOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::UnsignedDivIOp>(udivOp, udivOp.lhs(), udivOp.rhs());
    return success();
}

LogicalResult SRemLowering::matchAndRewrite(mlir::btor::SRemOp sremOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::SignedRemIOp>(sremOp, sremOp.lhs(), sremOp.rhs());
    return success();
}

LogicalResult URemLowering::matchAndRewrite(mlir::btor::URemOp uremOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::UnsignedRemIOp>(uremOp, uremOp.lhs(), uremOp.rhs());
    return success();
}

LogicalResult AndLowering::matchAndRewrite(mlir::btor::AndOp andOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::AndOp>(andOp, andOp.lhs(), andOp.rhs());
    return success();
}

LogicalResult NandLowering::matchAndRewrite(mlir::btor::NandOp nandOp, PatternRewriter &rewriter) const {
    Value andOp = rewriter.create<mlir::AndOp>(nandOp.getLoc(), nandOp.lhs(), nandOp.rhs());
    rewriter.replaceOpWithNewOp<NotOp>(nandOp, andOp);
    return success();
}

LogicalResult OrLowering::matchAndRewrite(mlir::btor::OrOp orOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::OrOp>(orOp, orOp.lhs(), orOp.rhs());
    return success();
}

LogicalResult NorLowering::matchAndRewrite(mlir::btor::NorOp norOp, PatternRewriter &rewriter) const {
    Value orOp = rewriter.create<mlir::OrOp>(norOp.getLoc(), norOp.lhs(), norOp.rhs());
    rewriter.replaceOpWithNewOp<NotOp>(norOp, orOp);
    return success();
}

LogicalResult XOrLowering::matchAndRewrite(mlir::btor::XOrOp xorOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::XOrOp>(xorOp, xorOp.lhs(), xorOp.rhs());
    return success();
}

LogicalResult XnorLowering::matchAndRewrite(mlir::btor::XnorOp xnorOp, PatternRewriter &rewriter) const {
    Value xorOp = rewriter.create<mlir::XOrOp>(xnorOp.getLoc(), xnorOp.lhs(), xnorOp.rhs());
    rewriter.replaceOpWithNewOp<NotOp>(xnorOp, xorOp);
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

LogicalResult IncLowering::matchAndRewrite(mlir::btor::IncOp incOp, PatternRewriter &rewriter) const {
    Value operand = incOp.operand(); 
    Type opType = operand.getType(); 

    Value oneConst = rewriter.create<ConstantOp>(incOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
    rewriter.replaceOpWithNewOp<mlir::AddIOp>(incOp, operand, oneConst);
    return success();
}

LogicalResult DecLowering::matchAndRewrite(mlir::btor::DecOp decOp, PatternRewriter &rewriter) const {
    Value operand = decOp.operand(); 
    Type opType = operand.getType(); 

    Value oneConst = rewriter.create<ConstantOp>(decOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
    rewriter.replaceOpWithNewOp<mlir::SubIOp>(decOp, operand, oneConst);
    return success();
}

LogicalResult ShiftLLLowering::matchAndRewrite(mlir::btor::ShiftLLOp sllOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::ShiftLeftOp>(sllOp, sllOp.lhs(), sllOp.rhs());
    return success();
}

LogicalResult ShiftRLLowering::matchAndRewrite(mlir::btor::ShiftRLOp srlOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::UnsignedShiftRightOp>(srlOp, srlOp.lhs(), srlOp.rhs());
    return success();
}

LogicalResult ShiftRALowering::matchAndRewrite(mlir::btor::ShiftRAOp sraOp, PatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<mlir::SignedShiftRightOp>(sraOp, sraOp.lhs(), sraOp.rhs());
    return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToStdConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AddLowering, MulLowering, AndLowering>(patterns.getContext());
  patterns.add<CmpLowering, BadLowering, NotLowering>(patterns.getContext());
  patterns.add<XOrLowering, XnorLowering, NandLowering>(patterns.getContext());
  patterns.add<OrLowering, NorLowering, IncLowering>(patterns.getContext());
  patterns.add<DecLowering, DecLowering, SRemLowering>(patterns.getContext());
  patterns.add<URemLowering, ShiftLLLowering, ShiftRLLowering>(patterns.getContext());
  patterns.add<ShiftRALowering, UDivLowering, SDivLowering>(patterns.getContext());
}

void BtorToStandardLoweringPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateBtorToStdConversionPatterns(patterns);
    /// Configure conversion to lower out btor.add; Anything else is fine.
    ConversionTarget target(getContext());
    target.addIllegalOp<mlir::btor::AddOp, mlir::btor::MulOp, mlir::btor::AndOp>();
    target.addIllegalOp<mlir::btor::CmpOp, mlir::btor::BadOp, mlir::btor::NotOp>();
    target.addIllegalOp<mlir::btor::XOrOp, mlir::btor::XnorOp, mlir::btor::NandOp>();
    target.addIllegalOp<mlir::btor::OrOp, mlir::btor::NorOp, mlir::btor::IncOp>();
    target.addIllegalOp<mlir::btor::DecOp, mlir::btor::SubOp, mlir::btor::SRemOp>();
    target.addIllegalOp<mlir::btor::URemOp, mlir::btor::ShiftLLOp, mlir::btor::ShiftRLOp>();
    target.addIllegalOp<mlir::btor::UDivOp, mlir::btor::SDivOp, mlir::btor::ShiftRAOp>();
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