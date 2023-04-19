#include "Target/Btor/BtorIRToBtorTranslation.h"
#include "Dialect/Btor/IR/Btor.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/TypeSwitch.h"

#include <iostream>
#include <string>

using namespace mlir;
using namespace mlir::btor;

LogicalResult Serialize::buildBinaryOperation(const Value &lhs,
                                              const Value &rhs,
                                              const Value &res,
                                              const Type type,
                                              std::string op) {
  assert (opIsInCache(lhs));
  assert (opIsInCache(rhs));
  auto sortId = getOrCreateSort(type);

  m_output << nextLine << " ";
  m_output << op << " ";
  m_output << sortId 
    << " " << getOpFromCache(lhs) << " "
    << getOpFromCache(rhs) << '\n';

  setCacheWithOp(res, nextLine);
  nextLine += 1;
  return success();
}

void Serialize::createSort(Type type) {
  if (type.isIntOrFloat()) {
    auto bitWidth = type.getIntOrFloatBitWidth();
    assert (bitWidth > 0);
    setSortWithType(type, nextLine);
    m_output << nextLine << " sort bitvec " << bitWidth << '\n';
  } else {
    assert (type.isa<VectorType>());
    auto shape = type.cast<VectorType>().getShape().front();
    auto elementType = type.cast<VectorType>().getElementType();
    Type shapeType = IntegerType::get(type.getContext(), unsigned (log2(shape)));
    assert (elementType.getIntOrFloatBitWidth() > 0);
    assert (shapeType.getIntOrFloatBitWidth() > 0);

    setSortWithType(type, nextLine);
    m_output << nextLine << " sort array "
      << getOrCreateSort(shapeType) << " "
      << getOrCreateSort(elementType) << '\n';
  }
  nextLine += 1;
}

uint64_t Serialize::getOrCreateSort(Type opType) {
  if (!sortIsInCache(opType))
    createSort(opType);
  return getSort(opType);
}

LogicalResult Serialize::createBtorLine(btor::UExtOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SExtOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SliceOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::NotOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::IncOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::DecOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::NegOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::RedOrOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::RedXorOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::IffOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::ImpliesOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::CmpOp &op, bool isInit) {
  assert (opIsInCache(op.lhs()));
  assert (opIsInCache(op.rhs()));
  auto sortId = getOrCreateSort(op.getType());
  
  m_output << nextLine;
  switch (op.predicate())
  {
  case BtorPredicate::eq:
    m_output << " eq ";
    break;
  case BtorPredicate::ne:
    m_output << " neq ";
    break;
  case BtorPredicate::sge: 
    m_output << " sgte ";
    break;
  case BtorPredicate::sgt:
    m_output << " sgt ";
    break;
  case BtorPredicate::sle:
    m_output << " slte ";
    break;
  case BtorPredicate::slt:
    m_output << " slt ";
    break;
  case BtorPredicate::uge:
    m_output << " ugte ";
    break;
  case BtorPredicate::ugt:
    m_output << " ugt ";
    break;
  case BtorPredicate::ule:
    m_output << " ulte ";
    break;
  case BtorPredicate::ult:
    m_output << " ult ";
    break;
  }
  m_output << sortId
    << " " << getOpFromCache(op.lhs()) << " "
    << getOpFromCache(op.rhs()) << '\n';

  setCacheWithOp(op.getResult(), nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::AndOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::NandOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::NorOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::OrOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::XnorOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::XOrOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::RotateROp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::RotateLOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::ShiftLLOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::ShiftRAOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::ShiftRLOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::ConcatOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(),
                            op.getType(), "concat");
}

LogicalResult Serialize::createBtorLine(btor::AddOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(),
                              op.getType(), "add");
}

LogicalResult Serialize::createBtorLine(btor::MulOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SDivOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::UDivOp &op, bool isInit) { 
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(),
                              op.getType(), "udiv");
}

LogicalResult Serialize::createBtorLine(btor::SModOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SRemOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::URemOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SubOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SAddOverflowOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::UAddOverflowOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SDivOverflowOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SMulOverflowOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::UMulOverflowOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::SSubOverflowOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::USubOverflowOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::IteOp &op, bool isInit) { return failure(); }

LogicalResult Serialize::createBtorLine(btor::ArrayOp &op, bool isInit) {
  return success(); // no work needs to be done here
}

LogicalResult Serialize::createBtorLine(btor::InitArrayOp &op, bool isInit) {
  assert (opIsInCache(op.init()));
  setCacheWithOp(op.result(), getOpFromCache(op.init()));
  return success();
}

LogicalResult Serialize::createBtorLine(btor::ReadOp &op, bool isInit) {
  return buildBinaryOperation(op.base(), op.index(), op.result(),
                              op.getType(), "read");
}

LogicalResult Serialize::createBtorLine(btor::WriteOp &op, bool isInit) {
  assert (opIsInCache(op.index()));
  assert (opIsInCache(op.base()));
  assert (opIsInCache(op.value()));
  auto sortId = getOrCreateSort(op.getType());

  m_output << nextLine << " write " << sortId 
    << " " << getOpFromCache(op.base()) << " "
    << getOpFromCache(op.index()) << " "
    << getOpFromCache(op.value()) << '\n';

  setCacheWithOp(op.result(), nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::ConstraintOp &op, bool isInit) { return failure(); }


LogicalResult Serialize::createBtorLine(btor::ConstantOp &op, bool isInit) {
  auto sortId = getOrCreateSort(op.getType());
  m_output << nextLine << " constd " 
    << sortId << " "
    << op.value().getInt() << '\n';

  setCacheWithOp(op.getResult(), nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::InputOp &op, bool isInit) {
  auto sortId = getOrCreateSort(op.getType());
  m_output << nextLine << " input " << sortId << '\n';

  setCacheWithOp(op.getResult(), nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::RedAndOp &op, bool isInit) {
  auto sortId = getOrCreateSort(op.getType());
  if (!opIsInCache(op.operand())) 
    return failure();
  
  auto operand = getOpFromCache(op.operand());  
  m_output << nextLine << " redand " << sortId 
    << " " << operand << '\n';

  setCacheWithOp(op.getResult(), nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::AssertNotOp &op, bool isInit) {
  if (!opIsInCache(op.arg())) 
    return failure();

  m_output << nextLine << " bad " << getOpFromCache(op.arg()) << '\n';
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::NDStateOp &op, bool isInit) {
  return success(); // no work needs to be done here
}

LogicalResult Serialize::createBtorLine(mlir::BranchOp &op, bool isInit) {
  if (isInit) {
    // create the states
    for (Type opType : op.getOperandTypes()) {;
      auto sortId = getOrCreateSort(opType);
      m_states.push_back(nextLine);
      m_output << nextLine << " state " << sortId << '\n';
      nextLine += 1;
    }
  } 

  for (unsigned i = 0; i < m_states.size(); ++i) {
    Value res = op.getOperand(i);
    auto sortId = getOrCreateSort(res.getType());
    if (opIsInCache(res)) {
      auto opNextState = getOpFromCache(res);
      m_output << nextLine;
      if (isInit) { m_output << " init "; } 
      else {  m_output << " next "; }
      m_output << sortId
        << " " << m_states.at(i) << " " 
        << opNextState << "\n";
      nextLine += 1;
    }
  }
  return success();
}

LogicalResult Serialize::Serialize::createBtor(mlir::Operation &op, bool isInit) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // btor ops.
          .Case<btor::UExtOp, btor::SExtOp, btor::SliceOp,
                btor::NotOp, btor::IncOp, btor::DecOp, btor::NegOp,
                btor::RedAndOp, btor::RedXorOp, btor::RedOrOp, btor::InputOp,
                btor::AssertNotOp, btor::ConstantOp, btor::NDStateOp,
                btor::IffOp, btor::ImpliesOp, btor::CmpOp,
                btor::AndOp, btor::NandOp, btor::NorOp, btor::OrOp,
                btor::XnorOp, btor::XOrOp, btor::RotateLOp, btor::RotateROp,
                btor::ShiftLLOp, btor::ShiftRAOp, btor::ShiftRLOp, btor::ConcatOp,
                btor::AddOp, btor::MulOp, btor::SDivOp, btor::UDivOp,
                btor::SModOp, btor::SRemOp, btor::URemOp, btor::SubOp,
                btor::SAddOverflowOp, btor::UAddOverflowOp, btor::SDivOverflowOp,
                btor::SMulOverflowOp, btor::UMulOverflowOp, btor::InitArrayOp,
                btor::SSubOverflowOp, btor::USubOverflowOp, btor::ReadOp,
                btor::IteOp, btor::ArrayOp, btor::ConstraintOp, btor::WriteOp>(
              [&](auto op) { return createBtorLine(op, isInit); })
          // Standard ops.
          .Case<BranchOp>(
              [&](auto op) { return createBtorLine(op, isInit); })
          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();
  return success();
}

LogicalResult Serialize::translateInitFunction(mlir::Block &initBlock) {
  std::cerr << "initBlock with " << initBlock.getNumArguments() << " arguments, "
    << initBlock.getNumSuccessors() << " successors, and "
    // Note, this `.size()` is traversing a linked-list and is O(n).
    << initBlock.getOperations().size() << " operations\n";
  for (Operation &op : initBlock.getOperations()) {
    std::cerr << "visiting op: '";
    op.getName().dump();
    std::cerr << "' with " << op.getNumOperands() << " operands and "
          << op.getNumResults() << " results\n";
    if (failed(createBtor(op, true))) {
      std::cerr << "Init Function creation failed";
      return failure();
    }
  }
  return success();
}

LogicalResult Serialize::translateNextFunction(mlir::Block &nextBlock) {
  std::cerr << "nextBlock with " << nextBlock.getNumArguments() << " arguments, "
    << nextBlock.getNumSuccessors() << " successors, and "
    // Note, this `.size()` is traversing a linked-list and is O(n).
    << nextBlock.getOperations().size() << " operations\n";
  // set states from block arguments
  for (unsigned i = 0; i < m_states.size(); ++i) {
    setCacheWithOp(nextBlock.getArgument(i), m_states.at(i));
  }
  // compute next states
  for (Operation &op : nextBlock.getOperations()) {
    std::cerr << "visiting op: '";
    op.getName().dump();
    std::cerr << "' with " << op.getNumOperands() << " operands and "
          << op.getNumResults() << " results\n";
    if (failed(createBtor(op, false))) {
      std::cerr << "Next Function creation failed";
      return failure();
    }
  }
  return success();
}

LogicalResult Serialize::translateMainFunction() {
  // extract main function from module
  auto module_regions = m_module->getRegions();
  auto &blocks = module_regions.front().getBlocks();
  auto &funcOp = blocks.front().getOperations().front();
  // translate each block
  auto &regions = funcOp.getRegion(0);
  assert (regions.getBlocks().size() == 2);
  if (translateInitFunction(regions.getBlocks().front()).failed())
    return failure();
  if (translateNextFunction(regions.getBlocks().back()).failed())
    return failure();
  m_module.dump();
  return success();
}

//===----------------------------------------------------------------------===//
// Serialization registration
//===----------------------------------------------------------------------===//

static LogicalResult serializeModule(ModuleOp module, raw_ostream &output) {
  if (!module)
    return failure();

  Serialize serialize(module, output);

  if (serialize.translateMainFunction().failed())
    return failure();

  return mlir::success();
}

namespace mlir {
namespace btor {
void registerToBtorTranslation() {
  TranslateFromMLIRRegistration toBtor(
      "export-btor", 
      [](ModuleOp module, raw_ostream &output) {
        return serializeModule(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<btor::BtorDialect, StandardOpsDialect>();
      });
}
} // namespace btor
} // namespace mlir