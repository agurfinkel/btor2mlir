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

LogicalResult Serialize::buildTernaryOperation(const Value &first,
                                              const Value &second,
                                              const Value &third,
                                              const Value &res,
                                              const Type type,
                                              std::string op) {
  assert (opIsInCache(first));
  assert (opIsInCache(second));
  assert (opIsInCache(third));
  auto sortId = getOrCreateSort(type);

  m_output << nextLine << " ";
  m_output << op << " " << sortId << " " << getOpFromCache(first) 
    << " " << getOpFromCache(second) << " "
    << getOpFromCache(third) << '\n';

  setCacheWithOp(res, nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::buildBinaryOperation(const Value &lhs,
                                              const Value &rhs,
                                              const Value &res,
                                              const Type type,
                                              std::string op) {
  assert (opIsInCache(lhs));
  assert (opIsInCache(rhs));
  auto sortId = getOrCreateSort(type);

  m_output << nextLine << " ";
  m_output << op << " " << sortId << " " << getOpFromCache(lhs) 
    << " " << getOpFromCache(rhs) << '\n';

  setCacheWithOp(res, nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::buildUnaryOperation(const Value &value,
              const Value &res, const Type type, std::string op) {
  assert (opIsInCache(value));
  auto sortId = getOrCreateSort(type);

  m_output << nextLine << " ";
  m_output << op << " ";
  m_output << sortId 
    << " " << getOpFromCache(value) << "\n";

  setCacheWithOp(res, nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::buildCastOperation(const Value &value,
              const Value &res, const Type type, std::string op) {
  assert (opIsInCache(value));
  auto sortId = getOrCreateSort(type);
  auto opType = type.dyn_cast<btor::BitVecType>();
  auto valueType = value.getType().dyn_cast<btor::BitVecType>();
  auto width = opType.getWidth() - valueType.getWidth();

  m_output << nextLine << " ";
  m_output << op << " " << sortId << " " 
    << getOpFromCache(value) << " " << width << "\n";

  setCacheWithOp(res, nextLine);
  nextLine += 1;
  return success();
}

void Serialize::createSort(Type type) {
  if (type.isa<btor::BitVecType>()) {
    auto bitWidth = type.dyn_cast<btor::BitVecType>().getWidth();
    assert (bitWidth > 0);
    setSortWithType(type, nextLine);
    m_output << nextLine << " sort bitvec " << bitWidth << '\n';
  } else {
    assert (type.isa<btor::ArrayType>());
    auto shapeType = type.cast<btor::ArrayType>().getShape();
    auto elementType = type.cast<btor::ArrayType>().getElement();
    assert (elementType.getWidth() > 0);
    assert (shapeType.getWidth() > 0);

    auto shapeSort = getOrCreateSort(shapeType);
    auto elementSort = getOrCreateSort(elementType);
    setSortWithType(type, nextLine);
    m_output << nextLine << " sort array "
      << shapeSort << " " << elementSort << '\n';
  }
  nextLine += 1;
}

uint64_t Serialize::getOrCreateSort(Type opType) {
  if (!sortIsInCache(opType))
    createSort(opType);
  return getSort(opType);
}

LogicalResult Serialize::createBtorLine(btor::UExtOp &op, bool isInit) {
  return buildCastOperation(op.in(), op.out(), op.getType(), "uext");
}

LogicalResult Serialize::createBtorLine(btor::SExtOp &op, bool isInit) {
  return buildCastOperation(op.in(), op.out(), op.getType(), "sext");
}

LogicalResult Serialize::createBtorLine(btor::SliceOp &op, bool isInit) {
  assert (opIsInCache(op.lower_bound()));
  assert (opIsInCache(op.upper_bound()));
  assert (opIsInCache(op.in()));
  btor::ConstantOp lower = op.lower_bound().getDefiningOp<btor::ConstantOp>();
  btor::ConstantOp upper = op.upper_bound().getDefiningOp<btor::ConstantOp>();
  assert (lower);
  assert (upper);
  auto sortId = getOrCreateSort(op.getType());

  m_output << nextLine << " slice " << sortId 
    << " " << getOpFromCache(op.in()) << " "
    << upper.value().getInt() << " "
    << lower.value().getInt() << "\n"; 

  setCacheWithOp(op.result(), nextLine);
  nextLine += 1;  
  return success();
}

LogicalResult Serialize::createBtorLine(btor::NotOp &op, bool isInit) {
  return buildUnaryOperation(op.operand(), op.result(), op.getType(), "not");
}

LogicalResult Serialize::createBtorLine(btor::IncOp &op, bool isInit) {
  return buildUnaryOperation(op.operand(), op.result(), op.getType(), "inc");
}

LogicalResult Serialize::createBtorLine(btor::DecOp &op, bool isInit) {
  return buildUnaryOperation(op.operand(), op.result(), op.getType(), "dec");
}

LogicalResult Serialize::createBtorLine(btor::NegOp &op, bool isInit) {
  return buildUnaryOperation(op.operand(), op.result(), op.getType(), "neg");
}

LogicalResult Serialize::createBtorLine(btor::RedOrOp &op, bool isInit) {
  return buildUnaryOperation(op.operand(), op.result(), op.getType(), "redor");
}

LogicalResult Serialize::createBtorLine(btor::RedXorOp &op, bool isInit) {
  return buildUnaryOperation(op.operand(), op.result(), op.getType(), "redxor");
}

LogicalResult Serialize::createBtorLine(btor::RedAndOp &op, bool isInit) {
  return buildUnaryOperation(op.operand(), op.result(), op.getType(), "redand");
}

LogicalResult Serialize::createBtorLine(btor::IffOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "iff");
}

LogicalResult Serialize::createBtorLine(btor::ImpliesOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "implies");
}

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

LogicalResult Serialize::createBtorLine(btor::AndOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "and");
}

LogicalResult Serialize::createBtorLine(btor::NandOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "nand");
}

LogicalResult Serialize::createBtorLine(btor::NorOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "nor");
}

LogicalResult Serialize::createBtorLine(btor::OrOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "or");
}

LogicalResult Serialize::createBtorLine(btor::XnorOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "xnor");
}

LogicalResult Serialize::createBtorLine(btor::XOrOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "xor");
}

LogicalResult Serialize::createBtorLine(btor::RotateROp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "ror");
}

LogicalResult Serialize::createBtorLine(btor::RotateLOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "rol");
}

LogicalResult Serialize::createBtorLine(btor::ShiftLLOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "sll");
}

LogicalResult Serialize::createBtorLine(btor::ShiftRAOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "sra");
}

LogicalResult Serialize::createBtorLine(btor::ShiftRLOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "srl");
}

LogicalResult Serialize::createBtorLine(btor::ConcatOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "concat");
}

LogicalResult Serialize::createBtorLine(btor::AddOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "add");
}

LogicalResult Serialize::createBtorLine(btor::MulOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "mul");
}

LogicalResult Serialize::createBtorLine(btor::SDivOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "sdiv");
}

LogicalResult Serialize::createBtorLine(btor::UDivOp &op, bool isInit) { 
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "udiv");
}

LogicalResult Serialize::createBtorLine(btor::SModOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "smod");
}

LogicalResult Serialize::createBtorLine(btor::SRemOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "srem");
}

LogicalResult Serialize::createBtorLine(btor::URemOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "urem");
}

LogicalResult Serialize::createBtorLine(btor::SubOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "sub");
}

LogicalResult Serialize::createBtorLine(btor::SAddOverflowOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "saddo");
}

LogicalResult Serialize::createBtorLine(btor::UAddOverflowOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "uaddo");
}

LogicalResult Serialize::createBtorLine(btor::SDivOverflowOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "sdivo");
}

LogicalResult Serialize::createBtorLine(btor::SMulOverflowOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "smulo");
}

LogicalResult Serialize::createBtorLine(btor::UMulOverflowOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "umulo");
}

LogicalResult Serialize::createBtorLine(btor::SSubOverflowOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "ssubo");
}

LogicalResult Serialize::createBtorLine(btor::USubOverflowOp &op, bool isInit) {
  return buildBinaryOperation(op.lhs(), op.rhs(), op.result(), op.getType(), "usubo");
}

LogicalResult Serialize::createBtorLine(btor::IteOp &op, bool isInit) {
  return buildTernaryOperation(op.condition(), op.true_value(), 
    op.false_value(), op.result(), op.getType(), "ite");
}

LogicalResult Serialize::createBtorLine(btor::ArrayOp &op, bool isInit) {
  return success(); // no work needs to be done here
}

LogicalResult Serialize::createBtorLine(btor::InitArrayOp &op, bool isInit) {
  assert (opIsInCache(op.init()));
  setCacheWithOp(op.result(), getOpFromCache(op.init()));
  return success();
}

LogicalResult Serialize::createBtorLine(btor::ReadOp &op, bool isInit) {
  return buildBinaryOperation(op.base(), op.index(), op.result(), op.getType(), "read");
}

LogicalResult Serialize::createBtorLine(btor::WriteOp &op, bool isInit) {
  return buildTernaryOperation(op.base(), op.index(), 
    op.value(), op.result(), op.getType(), "write");
}

LogicalResult Serialize::createBtorLine(btor::ConstraintOp &op, bool isInit) {
  assert (opIsInCache(op.constraint()));

  m_output << nextLine << " constraint " 
    << getOpFromCache(op.constraint()) << '\n';
  nextLine += 1;
  return success();
}


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

LogicalResult Serialize::createBtorLine(btor::AssertNotOp &op, bool isInit) {
  assert (opIsInCache(op.arg()));

  m_output << nextLine << " bad " << getOpFromCache(op.arg()) << '\n';
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::NDStateOp &op, bool isInit) {
  if (isInit) {
    auto sortId = getOrCreateSort(op.getType());
    m_output << nextLine << " state " << sortId << '\n';
    setCacheWithOp(op.getResult(), nextLine);
    nextLine += 1;
  }
  return success();
}

LogicalResult Serialize::createBtorLine(mlir::BranchOp &op, bool isInit) {
  if (isInit) {
    // create the states
    for (Value blockOperand : op.getOperands()) {
      auto parOp = blockOperand.getDefiningOp();
      if (auto parentOp = isa<NDStateOp>(parOp)) {
        // add the nd_state to the vector of states to match order
        assert(parOp->getNumResults() == 1);
        m_states.push_back(getOpFromCache(parOp->getResult(0)));
        continue;
      }
      auto opType = blockOperand.getType();
      auto sortId = getOrCreateSort(opType);
      m_states.push_back(nextLine);
      m_output << nextLine << " state " << sortId << '\n';
      nextLine += 1;
    }
  }
  // populate init/next operations for states
  for (unsigned i = 0; i < m_states.size(); ++i) {
    Value res = op.getOperand(i);
    auto sortId = getOrCreateSort(res.getType());
    if (opIsInCache(res)) {
      auto opNextState = getOpFromCache(res);
      if ((opNextState == m_states.at(i)) && isInit) { continue; }
      m_output << nextLine;
      if (isInit) { m_output << " init "; } 
      else {  m_output << " next "; }
      m_output << sortId
        << " " << m_states.at(i) << " " 
        << opNextState << "\n";
      nextLine += 1;
    }
  }
  // handle nd states that copy another state
  std::map<::llvm::hash_code, uint64_t> trackCopiedStates;
  using llvm::hash_value;
  for (unsigned i = 0; i < m_states.size(); ++i) {
    Value res = op.getOperand(i);
    if (opIsInCache(res)) {
      continue;
    } 
    auto sortId = getOrCreateSort(res.getType());
    auto resCode = hash_value(res);
    if (trackCopiedStates.count(resCode) != 0) {
      m_output << nextLine;
      if (isInit) { m_output << " init "; } 
      else {  m_output << " next "; }
      m_output << sortId
        << " " << m_states.at(i) << " " 
        << trackCopiedStates.at(resCode) << "\n";
      nextLine += 1;
    } else {
      trackCopiedStates[resCode] = m_states.at(i);
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
  for (Operation &op : initBlock.getOperations()) {
    if (failed(createBtor(op, true))) {
      return failure();
    }
  }
  return success();
}

LogicalResult Serialize::translateNextFunction(mlir::Block &nextBlock) {
  // set states from block arguments
  for (unsigned i = 0; i < m_states.size(); ++i) {
    setCacheWithOp(nextBlock.getArgument(i), m_states.at(i));
  }
  // compute next states
  for (Operation &op : nextBlock.getOperations()) {
    if (failed(createBtor(op, false))) {
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