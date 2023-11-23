#include "Conversion/BtorToLLVM/ConvertBtorToLLVMPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"

#include <string>

using namespace mlir;

#define PASS_NAME "convert-btor-to-llvm"

namespace {

template <typename SourceOp, typename BaseOp>
class ConvertNotOpToBtorPattern : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    Value baseOp =
        rewriter.create<BaseOp>(op.getLoc(), adaptor.lhs(), adaptor.rhs());
    rewriter.replaceOpWithNewOp<btor::NotOp>(op, baseOp);

    return success();
  }
};

template <typename SourceOp, typename TargetOp>
class ConvertReduceOpToLLVMPattern : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    auto operand = adaptor.operand();
    auto type = operand.getType();
    std::vector<int64_t> shape(1, type.getIntOrFloatBitWidth());
    Type vectorTtype = VectorType::get(shape, rewriter.getI1Type());
    auto vectorValue =
        rewriter.create<LLVM::BitcastOp>(op.getLoc(), vectorTtype, operand);

    rewriter.replaceOpWithNewOp<TargetOp>(op, rewriter.getI1Type(),
                                          vectorValue);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Straightforward Op Lowerings
//===----------------------------------------------------------------------===//

using AddOpLowering = VectorConvertToLLVMPattern<btor::AddOp, LLVM::AddOp>;
using SubOpLowering = VectorConvertToLLVMPattern<btor::SubOp, LLVM::SubOp>;
using MulOpLowering = VectorConvertToLLVMPattern<btor::MulOp, LLVM::MulOp>;
using AndOpLowering = VectorConvertToLLVMPattern<btor::AndOp, LLVM::AndOp>;
using OrOpLowering = VectorConvertToLLVMPattern<btor::OrOp, LLVM::OrOp>;
using XOrOpLowering = VectorConvertToLLVMPattern<btor::XOrOp, LLVM::XOrOp>;
using ShiftLLOpLowering =
    VectorConvertToLLVMPattern<btor::ShiftLLOp, LLVM::ShlOp>;
using ShiftRLOpLowering =
    VectorConvertToLLVMPattern<btor::ShiftRLOp, LLVM::LShrOp>;
using ShiftRAOpLowering =
    VectorConvertToLLVMPattern<btor::ShiftRAOp, LLVM::AShrOp>;
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
using UExtOpLowering = VectorConvertToLLVMPattern<btor::UExtOp, LLVM::ZExtOp>;
using SExtOpLowering = VectorConvertToLLVMPattern<btor::SExtOp, LLVM::SExtOp>;
using IteOpLowering = VectorConvertToLLVMPattern<btor::IteOp, LLVM::SelectOp>;
using RedOrOpLowering =
    ConvertReduceOpToLLVMPattern<btor::RedOrOp, LLVM::vector_reduce_or>;
using RedXorOpLowering =
    ConvertReduceOpToLLVMPattern<btor::RedXorOp, LLVM::vector_reduce_xor>;
using RedAndOpLowering =
    ConvertReduceOpToLLVMPattern<btor::RedAndOp, LLVM::vector_reduce_and>;
using XnorOpLowering = ConvertNotOpToBtorPattern<btor::XnorOp, btor::XOrOp>;
using NandOpLowering = ConvertNotOpToBtorPattern<btor::NandOp, btor::AndOp>;
using NorOpLowering = ConvertNotOpToBtorPattern<btor::NorOp, btor::OrOp>;

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public ConvertOpToLLVMPattern<btor::ConstantOp> {
  using ConvertOpToLLVMPattern<btor::ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct CmpOpLowering : public ConvertOpToLLVMPattern<btor::CmpOp> {
  using ConvertOpToLLVMPattern<btor::CmpOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct NotOpLowering : public ConvertOpToLLVMPattern<btor::NotOp> {
  using ConvertOpToLLVMPattern<btor::NotOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct AssertNotOpLowering : public ConvertOpToLLVMPattern<btor::AssertNotOp> {
  using ConvertOpToLLVMPattern<btor::AssertNotOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::AssertNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct IffOpLowering : public ConvertOpToLLVMPattern<btor::IffOp> {
  using ConvertOpToLLVMPattern<btor::IffOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::IffOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ImpliesOpLowering : public ConvertOpToLLVMPattern<btor::ImpliesOp> {
  using ConvertOpToLLVMPattern<btor::ImpliesOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::ImpliesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct RotateLOpLowering : public ConvertOpToLLVMPattern<btor::RotateLOp> {
  using ConvertOpToLLVMPattern<btor::RotateLOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::RotateLOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct RotateROpLowering : public ConvertOpToLLVMPattern<btor::RotateROp> {
  using ConvertOpToLLVMPattern<btor::RotateROp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::RotateROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct IncOpLowering : public ConvertOpToLLVMPattern<btor::IncOp> {
  using ConvertOpToLLVMPattern<btor::IncOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::IncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct DecOpLowering : public ConvertOpToLLVMPattern<btor::DecOp> {
  using ConvertOpToLLVMPattern<btor::DecOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::DecOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct NegOpLowering : public ConvertOpToLLVMPattern<btor::NegOp> {
  using ConvertOpToLLVMPattern<btor::NegOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct SliceOpLowering : public ConvertOpToLLVMPattern<btor::SliceOp> {
  using ConvertOpToLLVMPattern<btor::SliceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ConcatOpLowering : public ConvertOpToLLVMPattern<btor::ConcatOp> {
  using ConvertOpToLLVMPattern<btor::ConcatOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct SDivOverflowOpLowering
    : public ConvertOpToLLVMPattern<btor::SDivOverflowOp> {
  using ConvertOpToLLVMPattern<btor::SDivOverflowOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::SDivOverflowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct SModOpLowering : public ConvertOpToLLVMPattern<btor::SModOp> {
  using ConvertOpToLLVMPattern<btor::SModOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::SModOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct NDStateOpLowering : public ConvertOpToLLVMPattern<btor::NDStateOp> {
  using ConvertOpToLLVMPattern<btor::NDStateOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::NDStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ConstraintOpLowering : public ConvertOpToLLVMPattern<btor::ConstraintOp> {
  using ConvertOpToLLVMPattern<btor::ConstraintOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::ConstraintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct InputOpLowering : public ConvertOpToLLVMPattern<btor::InputOp> {
  using ConvertOpToLLVMPattern<btor::InputOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::InputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct ArrayOpLowering : public ConvertOpToLLVMPattern<mlir::btor::ArrayOp> {
  using ConvertOpToLLVMPattern<mlir::btor::ArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::ArrayOp arrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct UDivOpLowering : public ConvertOpToLLVMPattern<btor::UDivOp> {
  using ConvertOpToLLVMPattern<btor::UDivOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::UDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct SDivOpLowering : public ConvertOpToLLVMPattern<btor::SDivOp> {
  using ConvertOpToLLVMPattern<btor::SDivOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::SDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct URemOpLowering : public ConvertOpToLLVMPattern<btor::URemOp> {
  using ConvertOpToLLVMPattern<btor::URemOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::URemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

struct SRemOpLowering : public ConvertOpToLLVMPattern<btor::SRemOp> {
  using ConvertOpToLLVMPattern<btor::SRemOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(btor::SRemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// ConstantOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ConstantOpLowering::matchAndRewrite(btor::ConstantOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto resultType = op.getResult().getType();
  auto intType = typeConverter->convertType(resultType);

  unsigned val = op.valueAttr().getValue().getSExtValue();

  rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
      op, intType,
      rewriter.getIntegerAttr(intType, val));

  return success();
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

LogicalResult
CmpOpLowering::matchAndRewrite(btor::CmpOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  auto resultType = op.getResult().getType();

  rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
      op, typeConverter->convertType(resultType),
      convertBtorCmpPredicate<LLVM::ICmpPredicate>(op.getPredicate()),
      adaptor.lhs(), adaptor.rhs());

  return success();
}

//===----------------------------------------------------------------------===//
// NotOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
NotOpLowering::matchAndRewrite(btor::NotOp notOp, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value operand = adaptor.operand();
  Type opType = operand.getType();
  Value trueConst = rewriter.create<LLVM::ConstantOp>(
      notOp.getLoc(), opType, rewriter.getIntegerAttr(opType, -1));
  rewriter.replaceOpWithNewOp<LLVM::XOrOp>(notOp, operand, trueConst);
  return success();
}

//===----------------------------------------------------------------------===//
// AssertNotOpLowering
//===----------------------------------------------------------------------===//

LogicalResult AssertNotOpLowering::matchAndRewrite(
    btor::AssertNotOp assertOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {

  auto loc = assertOp.getLoc();
  Type i64Type = rewriter.getI64Type();
  Value notBad = rewriter.create<btor::NotOp>(loc, adaptor.arg());

  // Insert the `__VERIFIER_error` declaration if necessary.
  auto module = assertOp->getParentOfType<ModuleOp>();
  auto verifierError = "__VERIFIER_error";
  auto verifierErrorFunc =
      module.lookupSymbol<LLVM::LLVMFuncOp>(verifierError);
  auto verifierAssert = "__VERIFIER_assert";
  auto verifierAssertFunc =
      module.lookupSymbol<LLVM::LLVMFuncOp>(verifierAssert);
  if (!verifierErrorFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto verifierErrorFuncTy =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()), {});
    verifierErrorFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), verifierError, verifierErrorFuncTy);
    auto verifierAssertFuncTy =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(
          getContext()), {notBad.getType(), i64Type});
    verifierAssertFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), verifierAssert, verifierAssertFuncTy);
  }

  // Split block at `assert` operation.
  Block *opBlock = rewriter.getInsertionBlock();
  auto opPosition = rewriter.getInsertionPoint();
  Block *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

  // Generate IR to call `abort`.
  Block *failureBlock = rewriter.createBlock(opBlock->getParent());
  Value propertyNumber = rewriter.create<LLVM::ConstantOp>(
    loc, i64Type, rewriter.getIntegerAttr(i64Type, adaptor.id()));
  rewriter.create<LLVM::CallOp>(loc, verifierAssertFunc, ValueRange({notBad, propertyNumber}));
  rewriter.create<LLVM::CallOp>(loc, verifierErrorFunc, llvm::None);
  rewriter.create<LLVM::UnreachableOp>(loc);

  // Generate assertion test.
  rewriter.setInsertionPointToEnd(opBlock);
  rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(assertOp, notBad,
                                              continuationBlock, failureBlock);

  return success();
}

//===----------------------------------------------------------------------===//
// IffOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
IffOpLowering::matchAndRewrite(btor::IffOp iffOp, OpAdaptor adaptor,
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

LogicalResult
ImpliesOpLowering::matchAndRewrite(btor::ImpliesOp impOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  auto loc = impOp.getLoc();

  Value notLHS = rewriter.create<btor::NotOp>(loc, adaptor.lhs());
  rewriter.replaceOpWithNewOp<LLVM::OrOp>(impOp, notLHS, adaptor.rhs());
  return success();
}

//===----------------------------------------------------------------------===//
// RotateLOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
RotateLOpLowering::matchAndRewrite(btor::RotateLOp rolOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  // We convert using the following paradigm: given lhs, rhs, width
  // shiftBy = rhs % width
  // (lhs << shiftBy) or (lhs >> (width - shiftBy))
  auto loc = rolOp.getLoc();
  Value lhs = adaptor.lhs();
  Value rhs = adaptor.rhs();
  Type opType = lhs.getType();

  int width = opType.getIntOrFloatBitWidth();
  Value widthVal = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, width));
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

LogicalResult
RotateROpLowering::matchAndRewrite(btor::RotateROp rorOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  // We convert using the following paradigm: given lhs, rhs, width
  // shiftBy = rhs % width
  // (lhs >> shiftBy) or (lhs << (width - shiftBy))
  Location loc = rorOp.getLoc();
  Value lhs = adaptor.lhs();
  Value rhs = adaptor.rhs();
  Type opType = lhs.getType();

  int width = opType.getIntOrFloatBitWidth();
  Value widthVal = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, width));
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

LogicalResult
IncOpLowering::matchAndRewrite(mlir::btor::IncOp incOp, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value operand = adaptor.operand();
  Type opType = operand.getType();

  Value oneConst = rewriter.create<LLVM::ConstantOp>(
      incOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
  rewriter.replaceOpWithNewOp<LLVM::AddOp>(incOp, operand, oneConst);
  return success();
}

//===----------------------------------------------------------------------===//
// DecOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
DecOpLowering::matchAndRewrite(mlir::btor::DecOp decOp, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value operand = adaptor.operand();
  Type opType = operand.getType();

  Value oneConst = rewriter.create<LLVM::ConstantOp>(
      decOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 1));
  rewriter.replaceOpWithNewOp<LLVM::SubOp>(decOp, operand, oneConst);
  return success();
}

//===----------------------------------------------------------------------===//
// NegOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
NegOpLowering::matchAndRewrite(mlir::btor::NegOp negOp, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  Value operand = adaptor.operand();
  Type opType = operand.getType();

  Value zeroConst = rewriter.create<LLVM::ConstantOp>(
      negOp.getLoc(), opType, rewriter.getIntegerAttr(opType, 0));
  rewriter.replaceOpWithNewOp<LLVM::SubOp>(negOp, zeroConst, operand);
  return success();
}

//===----------------------------------------------------------------------===//
// SliceOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
SliceOpLowering::matchAndRewrite(mlir::btor::SliceOp sliceOp, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  // The idea here is to shift right until the bit indexed by the lowerbound is
  // the last bit on the right. Then we truncate to the size needed
  Value input = adaptor.in();
  Value valToTruncate =
      rewriter.create<LLVM::LShrOp>(sliceOp.getLoc(), input, adaptor.lower_bound());
  rewriter.replaceOpWithNewOp<LLVM::TruncOp>(
      sliceOp, TypeRange({sliceOp.result().getType()}), valToTruncate);
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ConcatOpLowering::matchAndRewrite(mlir::btor::ConcatOp concatOp,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {

  auto loc = concatOp.getLoc();
  Value lhs = adaptor.lhs();
  Value rhs = adaptor.rhs();

  int lhsWidth = lhs.getType().getIntOrFloatBitWidth();
  int rhsWidth = rhs.getType().getIntOrFloatBitWidth();
  auto resultWidthType = rewriter.getIntegerType(lhsWidth + rhsWidth);
  Value resultWidthVal = rewriter.create<LLVM::ConstantOp>(
      loc, resultWidthType,
      rewriter.getIntegerAttr(resultWidthType, lhsWidth + rhsWidth));
  Value rhsWidthVal = rewriter.create<LLVM::ConstantOp>(
      loc, resultWidthType, rewriter.getIntegerAttr(resultWidthType, rhsWidth));

  Value lhsZeroExtend =
      rewriter.create<LLVM::ZExtOp>(loc, resultWidthVal.getType(), lhs);
  Value lhsShiftLeft =
      rewriter.create<LLVM::ShlOp>(loc, lhsZeroExtend, rhsWidthVal);
  Value rhsZeroExtend =
      rewriter.create<LLVM::ZExtOp>(loc, resultWidthVal.getType(), rhs);
  rewriter.replaceOpWithNewOp<LLVM::OrOp>(concatOp, lhsShiftLeft,
                                          rhsZeroExtend);
  return success();
}

//===----------------------------------------------------------------------===//
// SDivOverflowOpLowering
//===----------------------------------------------------------------------===//

LogicalResult SDivOverflowOpLowering::matchAndRewrite(
    mlir::btor::SDivOverflowOp sdivOverflowOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = sdivOverflowOp.getLoc();
  auto rhs = adaptor.rhs(), lhs = adaptor.lhs();

  Value sdiv = rewriter.create<LLVM::SDivOp>(loc, lhs, rhs);
  Value product = rewriter.create<LLVM::MulOp>(loc, lhs, sdiv);

  rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
      sdivOverflowOp, LLVM::ICmpPredicate::ne, rhs, product);
  return success();
}

//===----------------------------------------------------------------------===//
// SModOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
SModOpLowering::matchAndRewrite(mlir::btor::SModOp smodOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  // since srem(a, b) = sign_of(a) * smod(a, b),
  // we have smod(a, b) =  sign_of(b) * |srem(a, b)|
  auto loc = smodOp.getLoc();
  auto rhs = adaptor.rhs(), lhs = adaptor.lhs();
  auto opType = rhs.getType();

  Value zeroConst = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, 0));
  Value srem = rewriter.create<btor::SRemOp>(loc, lhs, rhs);
  Value remLessThanZero = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::slt, srem, zeroConst);
  Value rhsLessThanZero = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::slt, rhs, zeroConst);
  Value rhsIsNotZero = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::ne, rhs, zeroConst);
  Value xorOp =
      rewriter.create<LLVM::XOrOp>(loc, remLessThanZero, rhsLessThanZero);
  Value needsNegationAndRhsNotZero =
    rewriter.create<LLVM::AndOp>(loc, xorOp, rhsIsNotZero);
  Value negOp = rewriter.create<btor::NegOp>(loc, srem);
  rewriter.replaceOpWithNewOp<LLVM::SelectOp>(smodOp, needsNegationAndRhsNotZero, negOp, srem);
  return success();
}

//===----------------------------------------------------------------------===//
// NDStateOpLowering
//===----------------------------------------------------------------------===//
LogicalResult
NDStateOpLowering::matchAndRewrite(btor::NDStateOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  auto bvType = op.result().getType().dyn_cast<btor::BitVecType>();
  auto opType = IntegerType::get(bvType.getContext(), bvType.getLength());
  // Insert the `havoc` declaration if necessary.
  auto module = op->getParentOfType<ModuleOp>();
  std::string havoc;
  havoc.append("nd_bv");
  havoc.append(std::to_string(opType.getIntOrFloatBitWidth()));
  auto havocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(havoc);
  if (!havocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto havocFuncTy =
        LLVM::LLVMFunctionType::get(opType, {});
    havocFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), havoc, havocFuncTy);
  }
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, havocFunc, llvm::None);
  return success();
}

//===----------------------------------------------------------------------===//
// ConstraintOpLowering
//===----------------------------------------------------------------------===//
LogicalResult
ConstraintOpLowering::matchAndRewrite(btor::ConstraintOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  auto opType = op.constraint().getType();

  // Insert the `__SEA_assume` declaration if necessary.
  auto module = op->getParentOfType<ModuleOp>();
  auto seaAssume = "__SEA_assume";
  auto seaAssumeFunc =
      module.lookupSymbol<LLVM::LLVMFuncOp>(seaAssume);
  if (!seaAssumeFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto seaAssumeFuncTy =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(getContext()), opType);
    seaAssumeFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), seaAssume, seaAssumeFuncTy);
  }

  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, seaAssumeFunc, op.constraint());
  return success();
}

//===----------------------------------------------------------------------===//
// InputOpLowering
//===----------------------------------------------------------------------===//
LogicalResult
InputOpLowering::matchAndRewrite(btor::InputOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const {
  auto opType = op.result().getType();
  // Insert the `havoc` declaration if necessary.
  auto module = op->getParentOfType<ModuleOp>();
  std::string havoc;
  havoc.append("nd_bv");
  havoc.append(std::to_string(opType.getIntOrFloatBitWidth()));
  auto havocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(havoc);
  if (!havocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto havocFuncTy =
        LLVM::LLVMFunctionType::get(opType, {});
    havocFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), havoc, havocFuncTy);
  }
  rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, havocFunc, llvm::None);
  return success();            
}

//===----------------------------------------------------------------------===//
// ArrayOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
ArrayOpLowering::matchAndRewrite(mlir::btor::ArrayOp arrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
  auto opType = arrayOp.getArrayType();
  // Insert the `havoc` declaration if necessary.
  auto module = arrayOp->getParentOfType<ModuleOp>();
  std::string havoc;
  havoc.append("nd_array");
  havoc.append(std::to_string(opType.getShape().front()));
  havoc.append("xbv");
  havoc.append(std::to_string(opType.getElementType().getIntOrFloatBitWidth()));
  auto havocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(havoc);
  if (!havocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto havocFuncTy =
        LLVM::LLVMFunctionType::get(opType, {});
    havocFunc = rewriter.create<LLVM::LLVMFuncOp>(
        rewriter.getUnknownLoc(), havoc, havocFuncTy);
  }
  auto callOp = rewriter.create<LLVM::CallOp>(arrayOp.getLoc(), havocFunc, llvm::None);
  auto result = callOp.getResult(0);
  arrayOp.replaceAllUsesWith(result);
  rewriter.replaceOp(arrayOp, result);
  return success();
}

//===----------------------------------------------------------------------===//
// SDivOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
SDivOpLowering::matchAndRewrite(mlir::btor::SDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  // ensure that if divisor is zero, result is -1
  auto loc = op.getLoc();
  auto rhs = adaptor.rhs(), lhs = adaptor.lhs();
  auto opType = rhs.getType();

  Value zeroConst = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, 0));  
  Value sdiv = rewriter.create<LLVM::SDivOp>(loc, lhs, rhs);
  Value divisorIsZero = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::eq, rhs, zeroConst);
  Value onesConst = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, -1));
  rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, divisorIsZero, onesConst, sdiv);
  return success();
}

//===----------------------------------------------------------------------===//
// UDivOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
UDivOpLowering::matchAndRewrite(mlir::btor::UDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  // ensure that if divisor is zero, result is -1
  auto loc = op.getLoc();
  auto rhs = adaptor.rhs(), lhs = adaptor.lhs();
  auto opType = rhs.getType();

  Value zeroConst = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, 0));  
  Value udiv = rewriter.create<LLVM::UDivOp>(loc, lhs, rhs);
  Value divisorIsZero = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::eq, rhs, zeroConst);
  Value onesConst = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, -1));
  rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, divisorIsZero, onesConst, udiv);
  return success();
}

//===----------------------------------------------------------------------===//
// SDivOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
SRemOpLowering::matchAndRewrite(mlir::btor::SRemOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  // ensure that if divisor is zero, result is lhs
  auto loc = op.getLoc();
  auto rhs = adaptor.rhs(), lhs = adaptor.lhs();
  auto opType = rhs.getType();

  Value zeroConst = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, 0));  
  Value srem = rewriter.create<LLVM::SRemOp>(loc, lhs, rhs);
  Value divisorIsZero = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::eq, rhs, zeroConst);
  rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, divisorIsZero, lhs, srem);
  return success();
}

//===----------------------------------------------------------------------===//
// URemOpLowering
//===----------------------------------------------------------------------===//

LogicalResult
URemOpLowering::matchAndRewrite(mlir::btor::URemOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
  // ensure that if divisor is zero, result is lhs
  auto loc = op.getLoc();
  auto rhs = adaptor.rhs(), lhs = adaptor.lhs();
  auto opType = rhs.getType();

  Value zeroConst = rewriter.create<LLVM::ConstantOp>(
      loc, opType, rewriter.getIntegerAttr(opType, 0));  
  Value urem = rewriter.create<LLVM::URemOp>(loc, lhs, rhs);
  Value divisorIsZero = rewriter.create<LLVM::ICmpOp>(
      loc, LLVM::ICmpPredicate::eq, rhs, zeroConst);
  rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, divisorIsZero, lhs, urem);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {

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
  BtorToLLVMTypeConverter converter(&getContext());

  mlir::btor::populateBtorToLLVMConversionPatterns(converter, patterns);
  mlir::populateStdToLLVMConversionPatterns(converter, patterns);

  /// Configure conversion to lower out btor; Anything else is fine.
  // indexed operators
  target.addIllegalOp<btor::UExtOp, btor::SExtOp, btor::SliceOp>();

  /// unary operators
  target.addIllegalOp<btor::NotOp, btor::IncOp, btor::DecOp, btor::NegOp>();
  target.addIllegalOp<btor::RedAndOp, btor::RedXorOp, btor::RedOrOp>();
  // target.addIllegalOp<btor::NDStateOp, btor::InputOp>();
  target.addIllegalOp<btor::AssertNotOp, btor::ConstraintOp, btor::ConstantOp>(); 

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
  target.addIllegalOp<btor::SModOp, btor::SRemOp, btor::URemOp,
                      btor::SubOp>(); // srem, urem, sub
  target.addIllegalOp<btor::SAddOverflowOp, btor::UAddOverflowOp,
                      btor::SDivOverflowOp>(); // saddo, uaddo
  target.addIllegalOp<btor::SMulOverflowOp, btor::UMulOverflowOp>();
  target.addIllegalOp<btor::SSubOverflowOp, btor::USubOverflowOp>();

  /// ternary operators
  target.addIllegalOp<btor::IteOp>();

  /// array operations
  target.addIllegalOp<btor::ArrayOp>();

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<
      ConstantOpLowering, AddOpLowering, SubOpLowering, MulOpLowering,
      UDivOpLowering, SDivOpLowering, URemOpLowering, SRemOpLowering,
      SModOpLowering, AndOpLowering, OrOpLowering, XOrOpLowering,
      ShiftLLOpLowering, ShiftRLOpLowering, ShiftRAOpLowering,
      RotateLOpLowering, RotateROpLowering, CmpOpLowering,
      SAddOverflowOpLowering, UAddOverflowOpLowering, SSubOverflowOpLowering,
      USubOverflowOpLowering, SMulOverflowOpLowering, UMulOverflowOpLowering,
      SDivOverflowOpLowering, NotOpLowering, AssertNotOpLowering, IteOpLowering,
      IffOpLowering, ImpliesOpLowering, XnorOpLowering, NandOpLowering,
      NorOpLowering, IncOpLowering, DecOpLowering, NegOpLowering,
      RedOrOpLowering, RedAndOpLowering, RedXorOpLowering, UExtOpLowering,
      SExtOpLowering, SliceOpLowering, ConcatOpLowering, 
      // NDStateOpLowering, InputOpLowering,
      ConstraintOpLowering, ArrayOpLowering>(converter);
}

/// Create a pass for lowering operations the remaining `Btor` operations
// to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToLLVMPass() {
  return std::make_unique<BtorToLLVMLoweringPass>();
}
