//===----------------------------------------------------------------------===//
//
// This provides registration calls for Btor dialect to Btor IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
#define TARGET_BTOR_BTORTOBTORIRTRANSLATION_H

#include "Dialect/Btor/IR/BtorDialect.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <vector>
#include <map>

#include "btor2parser/btor2parser.h"

namespace mlir {
class MLIRContext;
class ModuleOp;

namespace btor {

/// Deserializes the given Btor module and creates a MLIR ModuleOp
/// in the given `context`. Makes use of btor2parser.

class Deserialize {

 public:
///===----------------------------------------------------------------------===//
/// Constructors and Destructors
///===----------------------------------------------------------------------===//

  Deserialize(MLIRContext *context, const std::string &s) : m_context(context), 
    m_builder(OpBuilder(m_context)), m_unknownLoc(UnknownLoc::get(m_context)) {
        m_modelFile = fopen(s.c_str(), "r");
    }

  ~Deserialize() {
      if (m_model) {
        btor2parser_delete(m_model);
      }
      if (m_modelFile) {
        fclose(m_modelFile);
      }
  }

///===----------------------------------------------------------------------===//
/// Parse btor2 file
///===----------------------------------------------------------------------===//
  
  bool parseModelIsSuccessful();

///===----------------------------------------------------------------------===//
/// Create MLIR module
///===----------------------------------------------------------------------===//
  
  std::map<int64_t, Value> cache;

  OwningOpRef<FuncOp> buildInitFunction();
  OwningOpRef<FuncOp> buildNextFunction();
  
 private: 
///===----------------------------------------------------------------------===//
/// Parse btor2 file
///===----------------------------------------------------------------------===//
  
  Btor2Parser *m_model = nullptr;
  FILE *m_modelFile = nullptr;

  std::vector<Btor2Line *> m_inputs;
  std::vector<Btor2Line *> m_states;
  std::vector<Btor2Line *> m_bads;
  std::vector<Btor2Line *> m_inits;
  std::vector<Btor2Line *> m_nexts;
  std::vector<Btor2Line *> m_constraints;
 
  std::vector<Btor2Line *> m_lines;

  void parseModelLine(Btor2Line *l);

  Btor2Line * getLineById(unsigned id) {
      assert(id < m_lines.size());
      return m_lines.at(id);
  }

  void setLineWithId(unsigned id, Btor2Line * line) {
      assert(id < m_lines.size());
      assert(!m_lines.at(id));
      m_lines[id] = line;
  }
///===----------------------------------------------------------------------===//
/// Create MLIR module
///===----------------------------------------------------------------------===//

  MLIRContext *m_context;
  OpBuilder m_builder;
  Location m_unknownLoc;
  
  void toOp(Btor2Line *line);
  bool isValidChild(Btor2Line * line);
  void createNegateLine(int64_t curAt, const Value &child);
  Operation * createMLIR(const Btor2Line *line, const int64_t *kids);

  template <typename btorOp>
  Operation * buildBinaryOp(const Value &lhs, const Value &rhs) {
    auto res = m_builder.create<btorOp>(m_unknownLoc, lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildComparisonOp(btor::BtorPredicate pred,
                                const Value &lhs, const Value &rhs) {
    auto res = m_builder.create<btorOp>(m_unknownLoc, pred, lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildOverflowOp(const Value &lhs, const Value &rhs) {
    auto res = m_builder.create<btorOp>(m_unknownLoc, 
                                    m_builder.getIntegerType(1), 
                                    lhs, rhs);
    return res;
  }

  Operation * buildConstantOp(unsigned width, const std::string &str, unsigned radix) {
    Type type = m_builder.getIntegerType(width);
    mlir::APInt value(width, 0, radix);
    if (str.compare("ones") == 0) {
      value.setAllBits();
    } else if (str.compare("one") == 0) {
      value = mlir::APInt(width, 1, radix);
    } else if (str.compare("zero") != 0) {
      value = mlir::APInt(width, str, radix);
    }
    auto res = m_builder.create<btor::ConstantOp>(m_unknownLoc, type,
                        m_builder.getIntegerAttr(type, value.getLimitedValue()));
    return res;
  }

  Operation * buildConcatOp(const Value &lhs, const Value &rhs) {
    auto newWidth = lhs.getType().getIntOrFloatBitWidth() +
               rhs.getType().getIntOrFloatBitWidth();
    Type resType = m_builder.getIntegerType(newWidth);
    auto res = m_builder.create<btor::ConcatOp>(m_unknownLoc, resType, lhs, rhs);
    return res;
  }

  Operation * buildSliceOp(const Value &val, int64_t upper, int64_t lower) {
    auto opType = val.getType();
    auto operandWidth = opType.getIntOrFloatBitWidth();
    assert(operandWidth > upper && upper >= lower);

    auto resType = m_builder.getIntegerType(upper - lower + 1);
    auto u = m_builder.create<btor::ConstantOp>(
        m_unknownLoc, opType, m_builder.getIntegerAttr(opType, upper));
    assert(u && u->getNumResults() == 1);
    auto l = m_builder.create<btor::ConstantOp>(
        m_unknownLoc, opType, m_builder.getIntegerAttr(opType, lower));
    assert(l && l->getNumResults() == 1);

    auto res = m_builder.create<btor::SliceOp>(m_unknownLoc, resType, val,
                                        u->getResult(0), l->getResult(0));
    return res;
  }
};

/// Register the Btor translation
void registerFromBtorTranslation();

} // namespace btor
} // namespace mlir

#endif // TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
