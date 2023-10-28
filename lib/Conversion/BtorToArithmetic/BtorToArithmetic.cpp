#include "Conversion/BtorToArithmetic/ConvertBtorToArithmeticPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::btor;

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct ConstantLowering : public OpRewritePattern<mlir::btor::ConstantOp> {
  using OpRewritePattern<mlir::btor::ConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override;
};

struct AddLowering : public OpRewritePattern<mlir::btor::AddOp> {
  using OpRewritePattern<mlir::btor::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::AddOp addOp,
                                PatternRewriter &rewriter) const override;
};

struct SubLowering : public OpRewritePattern<mlir::btor::SubOp> {
  using OpRewritePattern<mlir::btor::SubOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::SubOp subOp,
                                PatternRewriter &rewriter) const override;
};

struct MulLowering : public OpRewritePattern<mlir::btor::MulOp> {
  using OpRewritePattern<mlir::btor::MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::MulOp mulOp,
                                PatternRewriter &rewriter) const override;
};

struct SDivLowering : public OpRewritePattern<mlir::btor::SDivOp> {
  using OpRewritePattern<mlir::btor::SDivOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::SDivOp sdivOp,
                                PatternRewriter &rewriter) const override;
};

struct UDivLowering : public OpRewritePattern<mlir::btor::UDivOp> {
  using OpRewritePattern<mlir::btor::UDivOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::UDivOp udivOp,
                                PatternRewriter &rewriter) const override;
};

struct SRemLowering : public OpRewritePattern<mlir::btor::SRemOp> {
  using OpRewritePattern<mlir::btor::SRemOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::SRemOp sremOp,
                                PatternRewriter &rewriter) const override;
};

struct URemLowering : public OpRewritePattern<mlir::btor::URemOp> {
  using OpRewritePattern<mlir::btor::URemOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::URemOp uremOp,
                                PatternRewriter &rewriter) const override;
};

struct AndLowering : public OpRewritePattern<mlir::btor::AndOp> {
  using OpRewritePattern<mlir::btor::AndOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::AndOp andOp,
                                PatternRewriter &rewriter) const override;
};

struct NandLowering : public OpRewritePattern<mlir::btor::NandOp> {
  using OpRewritePattern<mlir::btor::NandOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::NandOp nandOp,
                                PatternRewriter &rewriter) const override;
};

struct OrLowering : public OpRewritePattern<mlir::btor::OrOp> {
  using OpRewritePattern<mlir::btor::OrOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::OrOp orOp,
                                PatternRewriter &rewriter) const override;
};

struct NorLowering : public OpRewritePattern<mlir::btor::NorOp> {
  using OpRewritePattern<mlir::btor::NorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::NorOp norOp,
                                PatternRewriter &rewriter) const override;
};

struct XOrLowering : public OpRewritePattern<mlir::btor::XOrOp> {
  using OpRewritePattern<mlir::btor::XOrOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::XOrOp xorOp,
                                PatternRewriter &rewriter) const override;
};

struct XnorLowering : public OpRewritePattern<mlir::btor::XnorOp> {
  using OpRewritePattern<mlir::btor::XnorOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::XnorOp xnorOp,
                                PatternRewriter &rewriter) const override;
};

struct ShiftLLLowering : public OpRewritePattern<mlir::btor::ShiftLLOp> {
  using OpRewritePattern<mlir::btor::ShiftLLOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::ShiftLLOp sllOp,
                                PatternRewriter &rewriter) const override;
};

struct ShiftRLLowering : public OpRewritePattern<mlir::btor::ShiftRLOp> {
  using OpRewritePattern<mlir::btor::ShiftRLOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::ShiftRLOp srlOp,
                                PatternRewriter &rewriter) const override;
};

struct ShiftRALowering : public OpRewritePattern<mlir::btor::ShiftRAOp> {
  using OpRewritePattern<mlir::btor::ShiftRAOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::ShiftRAOp sraOp,
                                PatternRewriter &rewriter) const override;
};

struct RotateLLowering : public OpRewritePattern<mlir::btor::RotateLOp> {
  using OpRewritePattern<mlir::btor::RotateLOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::RotateLOp rolOp,
                                PatternRewriter &rewriter) const override;
};

struct RotateRLowering : public OpRewritePattern<mlir::btor::RotateROp> {
  using OpRewritePattern<mlir::btor::RotateROp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::RotateROp rorOp,
                                PatternRewriter &rewriter) const override;
};

struct CmpLowering : public OpRewritePattern<mlir::btor::CmpOp> {
  using OpRewritePattern<mlir::btor::CmpOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::CmpOp cmpOp,
                                PatternRewriter &rewriter) const override;
};

struct NotLowering : public OpRewritePattern<mlir::btor::NotOp> {
  using OpRewritePattern<mlir::btor::NotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::NotOp notOp,
                                PatternRewriter &rewriter) const override;
};

struct IncLowering : public OpRewritePattern<mlir::btor::IncOp> {
  using OpRewritePattern<mlir::btor::IncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::IncOp incOp,
                                PatternRewriter &rewriter) const override;
};

struct DecLowering : public OpRewritePattern<mlir::btor::DecOp> {
  using OpRewritePattern<mlir::btor::DecOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::DecOp decOp,
                                PatternRewriter &rewriter) const override;
};

struct NegLowering : public OpRewritePattern<mlir::btor::NegOp> {
  using OpRewritePattern<mlir::btor::NegOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::NegOp negOp,
                                PatternRewriter &rewriter) const override;
};

struct ConcatLowering : public OpRewritePattern<mlir::btor::ConcatOp> {
  using OpRewritePattern<mlir::btor::ConcatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override;
};

struct UExtLowering : public OpRewritePattern<mlir::btor::UExtOp> {
  using OpRewritePattern<mlir::btor::UExtOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::UExtOp uextOp,
                                PatternRewriter &rewriter) const override;
};

struct SExtLowering : public OpRewritePattern<mlir::btor::SExtOp> {
  using OpRewritePattern<mlir::btor::SExtOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::SExtOp sextOp,
                                PatternRewriter &rewriter) const override;
};

struct SliceLowering : public OpRewritePattern<mlir::btor::SliceOp> {
  using OpRewritePattern<mlir::btor::SliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::SliceOp sliceOp,
                                PatternRewriter &rewriter) const override;
};

struct IffLowering : public OpRewritePattern<mlir::btor::IffOp> {
  using OpRewritePattern<mlir::btor::IffOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::IffOp iffOp,
                                PatternRewriter &rewriter) const override;
};

struct ImpliesLowering : public OpRewritePattern<mlir::btor::ImpliesOp> {
  using OpRewritePattern<mlir::btor::ImpliesOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::btor::ImpliesOp impliesOp,
                                PatternRewriter &rewriter) const override;
};

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//

LogicalResult AddLowering::matchAndRewrite(mlir::btor::AddOp addOp,
                                           PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::AddIOp>(addOp, addOp.lhs(), addOp.rhs());
  return success();
}

LogicalResult SubLowering::matchAndRewrite(mlir::btor::SubOp subOp,
                                           PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::SubIOp>(subOp, subOp.lhs(), subOp.rhs());
  return success();
}

LogicalResult MulLowering::matchAndRewrite(mlir::btor::MulOp mulOp,
                                           PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::MulIOp>(mulOp, mulOp.lhs(), mulOp.rhs());
  return success();
}

LogicalResult SDivLowering::matchAndRewrite(mlir::btor::SDivOp sdivOp,
                                            PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::DivSIOp>(sdivOp, sdivOp.lhs(),
                                              sdivOp.rhs());
  return success();
}

LogicalResult UDivLowering::matchAndRewrite(mlir::btor::UDivOp udivOp,
                                            PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::DivUIOp>(udivOp, udivOp.lhs(),
                                              udivOp.rhs());
  return success();
}

LogicalResult SRemLowering::matchAndRewrite(mlir::btor::SRemOp sremOp,
                                            PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::RemSIOp>(sremOp, sremOp.lhs(),
                                              sremOp.rhs());
  return success();
}

LogicalResult URemLowering::matchAndRewrite(mlir::btor::URemOp uremOp,
                                            PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::RemUIOp>(uremOp, uremOp.lhs(),
                                              uremOp.rhs());
  return success();
}

LogicalResult AndLowering::matchAndRewrite(mlir::btor::AndOp andOp,
                                           PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::AndIOp>(andOp, andOp.lhs(), andOp.rhs());
  return success();
}

LogicalResult NandLowering::matchAndRewrite(mlir::btor::NandOp nandOp,
                                            PatternRewriter &rewriter) const {
  Value andOp = rewriter.create<arith::AndIOp>(nandOp.getLoc(), nandOp.lhs(),
                                               nandOp.rhs());
  rewriter.replaceOpWithNewOp<NotOp>(nandOp, andOp);
  return success();
}

LogicalResult OrLowering::matchAndRewrite(mlir::btor::OrOp orOp,
                                          PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::OrIOp>(orOp, orOp.lhs(), orOp.rhs());
  return success();
}

LogicalResult NorLowering::matchAndRewrite(mlir::btor::NorOp norOp,
                                           PatternRewriter &rewriter) const {
  Value orOp =
      rewriter.create<arith::OrIOp>(norOp.getLoc(), norOp.lhs(), norOp.rhs());
  rewriter.replaceOpWithNewOp<NotOp>(norOp, orOp);
  return success();
}

LogicalResult XOrLowering::matchAndRewrite(mlir::btor::XOrOp xorOp,
                                           PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::XOrIOp>(xorOp, xorOp.lhs(), xorOp.rhs());
  return success();
}

LogicalResult XnorLowering::matchAndRewrite(mlir::btor::XnorOp xnorOp,
                                            PatternRewriter &rewriter) const {
  Value xorOp = rewriter.create<arith::XOrIOp>(xnorOp.getLoc(), xnorOp.lhs(),
                                               xnorOp.rhs());
  rewriter.replaceOpWithNewOp<NotOp>(xnorOp, xorOp);
  return success();
}

// Convert btor.cmp predicate into the LLVM dialect ICmpOpPredicate.  The two
// enums share the numerical values so we just need to cast.
template <typename CmpIPredicate, typename BtorPredType>
static CmpIPredicate convertBtorCmpPredicate(BtorPredType pred) {
  return static_cast<mlir::arith::CmpIPredicate>(pred);
}

LogicalResult CmpLowering::matchAndRewrite(mlir::btor::CmpOp cmpOp,
                                           PatternRewriter &rewriter) const {
  auto btorPred =
      convertBtorCmpPredicate<arith::CmpIPredicate>(cmpOp.getPredicate());
  rewriter.replaceOpWithNewOp<arith::CmpIOp>(cmpOp, btorPred, cmpOp.lhs(),
                                             cmpOp.rhs());
  return success();
}

LogicalResult NotLowering::matchAndRewrite(mlir::btor::NotOp notOp,
                                           PatternRewriter &rewriter) const {
  Value operand = notOp.operand();
  Type opType = operand.getType();

  int trueVal = -1;
  Value trueConst = rewriter.create<arith::ConstantOp>(
      notOp.getLoc(), opType, rewriter.getIntegerAttr(opType, trueVal));
  rewriter.replaceOpWithNewOp<arith::XOrIOp>(notOp, operand, trueConst);
  return success();
}

LogicalResult IncLowering::matchAndRewrite(mlir::btor::IncOp incOp,
                                           PatternRewriter &rewriter) const {
  Value operand = incOp.operand();
  Type opType = operand.getType();

  Value oneConst = rewriter.create<arith::ConstantOp>(
      incOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
  rewriter.replaceOpWithNewOp<arith::AddIOp>(incOp, operand, oneConst);
  return success();
}

LogicalResult DecLowering::matchAndRewrite(mlir::btor::DecOp decOp,
                                           PatternRewriter &rewriter) const {
  Value operand = decOp.operand();
  Type opType = operand.getType();

  Value oneConst = rewriter.create<arith::ConstantOp>(
      decOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
  rewriter.replaceOpWithNewOp<arith::SubIOp>(decOp, operand, oneConst);
  return success();
}

LogicalResult NegLowering::matchAndRewrite(mlir::btor::NegOp negOp,
                                           PatternRewriter &rewriter) const {
  Value operand = negOp.operand();
  Type opType = operand.getType();

  Value zeroConst = rewriter.create<arith::ConstantOp>(
      negOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 0));
  rewriter.replaceOpWithNewOp<arith::SubIOp>(negOp, zeroConst, operand);
  return success();
}

LogicalResult
ShiftLLLowering::matchAndRewrite(mlir::btor::ShiftLLOp sllOp,
                                 PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ShLIOp>(sllOp, sllOp.lhs(), sllOp.rhs());
  return success();
}

LogicalResult
ShiftRLLowering::matchAndRewrite(mlir::btor::ShiftRLOp srlOp,
                                 PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ShRUIOp>(srlOp, srlOp.lhs(), srlOp.rhs());
  return success();
}

LogicalResult
ShiftRALowering::matchAndRewrite(mlir::btor::ShiftRAOp sraOp,
                                 PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ShRSIOp>(sraOp, sraOp.lhs(), sraOp.rhs());
  return success();
}

LogicalResult
RotateLLowering::matchAndRewrite(mlir::btor::RotateLOp rolOp,
                                 PatternRewriter &rewriter) const {
  // We convert using the following paradigm: given lhs, rhs, width
  // shiftBy = rhs % width
  // (lhs << shiftBy) or (lhs >> (width - shiftBy))
  Location loc = rolOp.getLoc();
  Value lhs = rolOp.lhs();
  Value rhs = rolOp.rhs();
  Type opType = lhs.getType();

  int width = opType.getIntOrFloatBitWidth();
  Value widthVal = rewriter.create<arith::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, width));
  Value shiftBy = rewriter.create<arith::RemUIOp>(loc, rhs, widthVal);
  Value shiftRightBy = rewriter.create<arith::SubIOp>(loc, widthVal, shiftBy);

  Value leftValue = rewriter.create<arith::ShLIOp>(loc, lhs, shiftBy);
  Value rightValue = rewriter.create<arith::ShRUIOp>(loc, lhs, shiftRightBy);

  rewriter.replaceOpWithNewOp<arith::OrIOp>(rolOp, leftValue, rightValue);
  return success();
}

LogicalResult
RotateRLowering::matchAndRewrite(mlir::btor::RotateROp rorOp,
                                 PatternRewriter &rewriter) const {
  // We convert using the following paradigm: given lhs, rhs, width
  // shiftBy = rhs % width
  // (lhs >> shiftBy) or (lhs << (width - shiftBy))
  Location loc = rorOp.getLoc();
  Value lhs = rorOp.lhs();
  Value rhs = rorOp.rhs();
  Type opType = lhs.getType();

  int width = opType.getIntOrFloatBitWidth();
  Value widthVal = rewriter.create<arith::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, width));
  Value shiftBy = rewriter.create<arith::RemUIOp>(loc, rhs, widthVal);
  Value shiftLeftBy = rewriter.create<arith::SubIOp>(loc, widthVal, shiftBy);

  Value leftValue = rewriter.create<arith::ShRUIOp>(loc, lhs, shiftBy);
  Value rightValue = rewriter.create<arith::ShLIOp>(loc, lhs, shiftLeftBy);

  rewriter.replaceOpWithNewOp<arith::OrIOp>(rorOp, leftValue, rightValue);
  return success();
}

LogicalResult
ConstantLowering::matchAndRewrite(mlir::btor::ConstantOp constOp,
                                  PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, constOp.getType(),
                                                 constOp.value());
  return success();
}

LogicalResult ConcatLowering::matchAndRewrite(mlir::btor::ConcatOp concatOp,
                                              PatternRewriter &rewriter) const {
  auto loc = concatOp.getLoc();
  Value lhs = concatOp.lhs();
  Value rhs = concatOp.rhs();

  int lhsWidth = lhs.getType().getIntOrFloatBitWidth();
  int rhsWidth = rhs.getType().getIntOrFloatBitWidth();
  auto resultWidthType = rewriter.getIntegerType(lhsWidth + rhsWidth);
  Value resultWidthVal = rewriter.create<arith::ConstantOp>(
      loc, resultWidthType,
      rewriter.getIntegerAttr(resultWidthType, lhsWidth + rhsWidth));
  Value rhsWidthVal = rewriter.create<arith::ConstantOp>(
      loc, resultWidthType, rewriter.getIntegerAttr(resultWidthType, rhsWidth));

  Value lhsZeroExtend =
      rewriter.create<arith::ExtUIOp>(loc, resultWidthVal.getType(), lhs);
  Value lhsShiftLeft =
      rewriter.create<arith::ShLIOp>(loc, lhsZeroExtend, rhsWidthVal);
  Value rhsZeroExtend =
      rewriter.create<arith::ExtUIOp>(loc, resultWidthVal.getType(), rhs);
  rewriter.replaceOpWithNewOp<arith::OrIOp>(concatOp, lhsShiftLeft,
                                            rhsZeroExtend);
  return success();
}

LogicalResult UExtLowering::matchAndRewrite(mlir::btor::UExtOp uextOp,
                                            PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ExtUIOp>(uextOp, uextOp.getType(),
                                              uextOp.in());
  return success();
}

LogicalResult SExtLowering::matchAndRewrite(mlir::btor::SExtOp sextOp,
                                            PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<arith::ExtSIOp>(sextOp, sextOp.getType(),
                                              sextOp.in());
  return success();
}

LogicalResult SliceLowering::matchAndRewrite(mlir::btor::SliceOp sliceOp,
                                             PatternRewriter &rewriter) const {
  // The idea here is to shift right until the bit indexed by the upperbound is
  // the last bit on the right. Then we truncate to the type needed
  auto loc = sliceOp.getLoc();
  Value input = sliceOp.in();
  Type opType = input.getType();

  int inputWidth = opType.getIntOrFloatBitWidth();
  Value widthVal = rewriter.create<arith::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, inputWidth));
  Value shiftRightBy =
      rewriter.create<arith::SubIOp>(loc, widthVal, sliceOp.upper_bound());

  Value valToTruncate =
      rewriter.create<arith::ShRUIOp>(loc, input, shiftRightBy);
  rewriter.replaceOpWithNewOp<arith::TruncIOp>(
      sliceOp, TypeRange({sliceOp.result().getType()}), valToTruncate);
  return success();
}

LogicalResult IffLowering::matchAndRewrite(mlir::btor::IffOp iffOp,
                                           PatternRewriter &rewriter) const {
  auto loc = iffOp.getLoc();

  Value notLHS = rewriter.create<btor::NotOp>(loc, iffOp.lhs());
  Value notRHS = rewriter.create<btor::NotOp>(loc, iffOp.rhs());

  Value notLHSorRHS = rewriter.create<arith::OrIOp>(loc, notLHS, iffOp.rhs());
  Value notRHSorLHS = rewriter.create<arith::OrIOp>(loc, notRHS, iffOp.lhs());
  rewriter.replaceOpWithNewOp<arith::AndIOp>(iffOp, notLHSorRHS, notRHSorLHS);
  return success();
}

LogicalResult
ImpliesLowering::matchAndRewrite(mlir::btor::ImpliesOp impOp,
                                 PatternRewriter &rewriter) const {
  auto loc = impOp.getLoc();

  Value notLHS = rewriter.create<btor::NotOp>(loc, impOp.lhs());
  rewriter.replaceOpWithNewOp<arith::OrIOp>(impOp, notLHS, impOp.rhs());
  return success();
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToArithmeticConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConstantLowering, AddLowering, SubLowering, MulLowering,
               UDivLowering, SDivLowering, URemLowering, SRemLowering,
               AndLowering, OrLowering, XOrLowering,
               ShiftLLLowering, ShiftRLLowering, ShiftRALowering,
               RotateLLowering, RotateRLowering, CmpLowering, NotLowering,
               IffLowering, ImpliesLowering, XnorLowering,
               NandLowering, NorLowering, IncLowering, DecLowering, NegLowering,
               UExtLowering, SExtLowering, SliceLowering, ConcatLowering>(
      patterns.getContext());
}

namespace {
struct ConvertBtorToArithmeticPass
    : public ConvertBtorToArithmeticBase<ConvertBtorToArithmeticPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBtorToArithmeticConversionPatterns(patterns);
    ConversionTarget target(getContext());

    /// Configure conversion to lower out btor; Anything else is fine.
    // indexed operators
    target.addIllegalOp<btor::UExtOp, btor::SExtOp, btor::SliceOp>();

    /// unary operators
    target.addIllegalOp<btor::NotOp, btor::IncOp, btor::DecOp, btor::NegOp>();
    target.addIllegalOp<btor::ConstantOp>();

    /// binary operators
    // logical
    target.addIllegalOp<btor::IffOp, btor::ImpliesOp, btor::CmpOp>();
    target.addIllegalOp<btor::AndOp, btor::NandOp, btor::NorOp, btor::OrOp>();
    target.addIllegalOp<btor::XnorOp, btor::XOrOp, btor::RotateLOp,
                        btor::RotateROp>();
    target.addIllegalOp<btor::ShiftLLOp, btor::ShiftRAOp, btor::ShiftRLOp,
                        btor::ConcatOp>();
    // arithmetic
    target.addIllegalOp<btor::AddOp, btor::MulOp, btor::SDivOp, btor::UDivOp>();
    target.addIllegalOp<btor::SRemOp, btor::URemOp, btor::SubOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
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
