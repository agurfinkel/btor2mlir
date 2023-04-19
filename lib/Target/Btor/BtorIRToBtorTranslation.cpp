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
using llvm::hash_value;

void Serialize::createSort(int bitWidth) {
  assert (bitWidth > 0);
  setSortWithType(bitWidth, nextLine);
  m_output << nextLine << " sort bitvec " << bitWidth << '\n';
  nextLine += 1;
}

LogicalResult Serialize::createBtorLine(btor::UExtOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SExtOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SliceOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::NotOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::IncOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::DecOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::NegOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::RedOrOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::RedXorOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::IffOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::ImpliesOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::CmpOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::AndOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::NandOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::NorOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::OrOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::XnorOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::XOrOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::RotateROp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::RotateLOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::ShiftLLOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::ShiftRAOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::ShiftRLOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::ConcatOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::AddOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::MulOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SDivOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::UDivOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SModOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SRemOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::URemOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SubOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SAddOverflowOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::UAddOverflowOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SDivOverflowOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SMulOverflowOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::UMulOverflowOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::SSubOverflowOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::USubOverflowOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::IteOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::ArrayOp &op, bool isInit) { return success(); }

LogicalResult Serialize::createBtorLine(btor::ConstraintOp &op, bool isInit) { return success(); }


LogicalResult Serialize::createBtorLine(btor::ConstantOp &op, bool isInit) {
  auto opType = op.getType().getIntOrFloatBitWidth();
  if (!sortIsInCache(opType))
    createSort(opType);
  m_output << nextLine << " constd " 
    << getSort(opType) << " "
    << op.value() << '\n';

  llvm::hash_code code = hash_value(op.getResult());
  setCacheWithOp(code, nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::InputOp &op, bool isInit) {
  auto opType = op.getType().getIntOrFloatBitWidth();
  if (!sortIsInCache(opType))
    createSort(opType);
  m_output << nextLine << " input " << getSort(opType) << '\n';

  llvm::hash_code code = hash_value(op.getResult());
  setCacheWithOp(code, nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::RedAndOp &op, bool isInit) {
  auto opType = op.getType().getIntOrFloatBitWidth();
  if (!sortIsInCache(opType))
    createSort(opType);
  llvm::hash_code opCode = hash_value(op.operand());
  if (!opIsInCache(opCode)) 
    return failure();
  
  auto operand = getOpFromCache(opCode);  
  m_output << nextLine << " redand " << getSort(opType) 
    << " " << operand << '\n';

  llvm::hash_code code = hash_value(op.getResult());
  setCacheWithOp(code, nextLine);
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::AssertNotOp &op, bool isInit) {
  llvm::hash_code argCode = hash_value(op.arg());
  if (!opIsInCache(argCode)) 
    return failure();

  m_output << nextLine << " bad " << getOpFromCache(argCode) << '\n';
  nextLine += 1;
  return success();
}

LogicalResult Serialize::createBtorLine(btor::NDStateOp &op, bool isInit) {
  return success(); // no work needs to be done here
}

LogicalResult Serialize::createBtorLine(mlir::BranchOp &op, bool isInit) {
  if (isInit) {
    // create the states
    for (Type type : op.getOperandTypes()) {
      auto bitWidth = type.getIntOrFloatBitWidth();
      if (!sortIsInCache(bitWidth))
        createSort(bitWidth);
      m_states.push_back(nextLine);
      m_output << nextLine << " state " << getSort(bitWidth) << '\n';
      nextLine += 1;
    }
  } 

  for (unsigned i = 0; i < m_states.size(); ++i) {
    Value res = op.getOperand(i);
    auto sort = res.getType().getIntOrFloatBitWidth();
    llvm::hash_code code = hash_value(res);
    if (opIsInCache(code)) {
      auto opNextState = getOpFromCache(code);
      m_output << nextLine;
      if (isInit) { m_output << " init "; } 
      else {  m_output << " next "; }
      m_output << getSort(sort)
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
                btor::SMulOverflowOp, btor::UMulOverflowOp,
                btor::SSubOverflowOp, btor::USubOverflowOp,
                btor::IteOp, btor::ArrayOp, btor::ConstraintOp>(
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
    llvm::hash_code blockArgCode = hash_value(nextBlock.getArgument(i));
    setCacheWithOp(blockArgCode, m_states.at(i));
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

  // output.write(reinterpret_cast<char *>(module.dump()),
  //              30 * sizeof(uint32_t));
  output << '\n';

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