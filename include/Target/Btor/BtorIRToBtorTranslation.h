//===----------------------------------------------------------------------===//
//
// This provides registration calls for Btor to Btor IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_BTOR_BTORIRTOBTORTRANSLATION_H
#define TARGET_BTOR_BTORIRTOBTORTRANSLATION_H

#include "Dialect/Btor/IR/Btor.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"

#include <map>
#include <vector>

#include "btor2parser/btor2parser.h"

using llvm::hash_value;

namespace mlir {
class ModuleOp;

namespace btor {

/// Serializes the given Btor MLIR module and creates a Btor.

class Serialize {

public:
  ///===----------------------------------------------------------------------===//
  /// Constructors and Destructors
  ///===----------------------------------------------------------------------===//

  Serialize(ModuleOp module, raw_ostream &output)
      : m_module(module), m_output(output) {}

  ~Serialize() {}

  LogicalResult translateMainFunction();

  uint64_t getSort(const Type type) {
    assert(sortIsInCache(type));
    ::llvm::hash_code code = hash_value(type);
    return m_sorts.at(code);
  }

  void setSortWithType(const Type type, const uint64_t id) {
    assert(!sortIsInCache(type));
    ::llvm::hash_code code = hash_value(type);
    m_sorts[code] = id;
    assert(sortIsInCache(type));
  }

  bool sortIsInCache(const Type type) {
    ::llvm::hash_code code = hash_value(type);
    return m_sorts.count(code) != 0;
  }

  uint64_t getOpFromCache(const Value &value) {
    assert(opIsInCache(value));
    ::llvm::hash_code code = hash_value(value);
    return m_cache.at(code);
  }

  void setCacheWithOp(const Value &value, const uint64_t id) {
    assert(!opIsInCache(value));
    ::llvm::hash_code code = hash_value(value);
    m_cache[code] = id;
    assert(opIsInCache(value));
  }

  bool opIsInCache(const Value &value) {
    ::llvm::hash_code code = hash_value(value);
    return m_cache.count(code) != 0;
  }

private:
  ModuleOp m_module;
  raw_ostream &m_output;
  uint64_t nextLine = 1;

  std::vector<uint64_t> m_states;

  std::map<::llvm::hash_code, uint64_t> m_cache;
  std::map<::llvm::hash_code, uint64_t> m_sorts;

  void createSort(Type type);
  uint64_t getOrCreateSort(Type type);

  LogicalResult translateInitFunction(mlir::Block &initBlock);
  LogicalResult translateNextFunction(mlir::Block &nextBlock);
  LogicalResult createBtor(mlir::Operation &op, bool isInit);

  LogicalResult createBtorLine(btor::ConstantOp &op, bool isInit);
  LogicalResult createBtorLine(btor::InputOp &op, bool isInit);
  LogicalResult createBtorLine(btor::RedAndOp &op, bool isInit);
  LogicalResult createBtorLine(btor::AssertNotOp &op, bool isInit);
  LogicalResult createBtorLine(btor::NDStateOp &op, bool isInit);
  LogicalResult createBtorLine(btor::UExtOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SExtOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SliceOp &op, bool isInit);
  LogicalResult createBtorLine(btor::NotOp &op, bool isInit);
  LogicalResult createBtorLine(btor::IncOp &op, bool isInit);
  LogicalResult createBtorLine(btor::DecOp &op, bool isInit);
  LogicalResult createBtorLine(btor::NegOp &op, bool isInit);
  LogicalResult createBtorLine(btor::RedOrOp &op, bool isInit);
  LogicalResult createBtorLine(btor::RedXorOp &op, bool isInit);
  LogicalResult createBtorLine(btor::IffOp &op, bool isInit);
  LogicalResult createBtorLine(btor::ImpliesOp &op, bool isInit);
  LogicalResult createBtorLine(btor::CmpOp &op, bool isInit);
  LogicalResult createBtorLine(btor::AndOp &op, bool isInit);
  LogicalResult createBtorLine(btor::NandOp &op, bool isInit);
  LogicalResult createBtorLine(btor::NorOp &op, bool isInit);
  LogicalResult createBtorLine(btor::OrOp &op, bool isInit);
  LogicalResult createBtorLine(btor::XnorOp &op, bool isInit);
  LogicalResult createBtorLine(btor::XOrOp &op, bool isInit);
  LogicalResult createBtorLine(btor::RotateROp &op, bool isInit);
  LogicalResult createBtorLine(btor::RotateLOp &op, bool isInit);
  LogicalResult createBtorLine(btor::ShiftLLOp &op, bool isInit);
  LogicalResult createBtorLine(btor::ShiftRAOp &op, bool isInit);
  LogicalResult createBtorLine(btor::ShiftRLOp &op, bool isInit);
  LogicalResult createBtorLine(btor::ConcatOp &op, bool isInit);
  LogicalResult createBtorLine(btor::AddOp &op, bool isInit);
  LogicalResult createBtorLine(btor::MulOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SDivOp &op, bool isInit);
  LogicalResult createBtorLine(btor::UDivOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SModOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SRemOp &op, bool isInit);
  LogicalResult createBtorLine(btor::URemOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SubOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SAddOverflowOp &op, bool isInit);
  LogicalResult createBtorLine(btor::UAddOverflowOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SDivOverflowOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SMulOverflowOp &op, bool isInit);
  LogicalResult createBtorLine(btor::UMulOverflowOp &op, bool isInit);
  LogicalResult createBtorLine(btor::SSubOverflowOp &op, bool isInit);
  LogicalResult createBtorLine(btor::USubOverflowOp &op, bool isInit);
  LogicalResult createBtorLine(btor::IteOp &op, bool isInit);
  LogicalResult createBtorLine(btor::ArrayOp &op, bool isInit);
  LogicalResult createBtorLine(btor::ConstraintOp &op, bool isInit);
  LogicalResult createBtorLine(btor::InitArrayOp &op, bool isInit);
  LogicalResult createBtorLine(btor::ReadOp &op, bool isInit);
  LogicalResult createBtorLine(btor::WriteOp &op, bool isInit);

  LogicalResult createBtorLine(mlir::BranchOp &op, bool isInit);

  LogicalResult buildTernaryOperation(const Value &first, const Value &second,
                                      const Value &third, const Value &res,
                                      Type type, std::string op);
  LogicalResult buildBinaryOperation(const Value &lhs, const Value &rhs,
                                     const Value &res, Type type,
                                     std::string op);
  LogicalResult buildUnaryOperation(const Value &lhs, const Value &res,
                                    Type type, std::string op);
  LogicalResult buildCastOperation(const Value &in, const Value &res, Type type,
                                   std::string op);
};

/// Register the Btor translation
void registerToBtorTranslation();

} // namespace btor
} // namespace mlir

#endif // TARGET_BTOR_BTORIRTOBTORTRANSLATION_H
