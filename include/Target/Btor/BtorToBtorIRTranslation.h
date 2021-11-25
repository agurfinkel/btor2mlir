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
  
  Value getFromCacheById(const int64_t id) {
    assert(valueAtIdIsInCache(id));
    return m_cache.at(id);
  }

  void setCacheWithId(const int64_t id, const Value &value) {
    assert(!valueAtIdIsInCache(id));
    m_cache[id] = value;
    assert(valueAtIdIsInCache(id));
  }

  void setCacheWithId(const int64_t id, Operation * op) {
    // We never have to use the result of a btor line with no 
    // return values since btor2 doesn't allow it
    if (hasReturnValue(getLineById(id))) {
      assert(op);
      assert(op->getNumResults() == 1);
      assert(op->getResult(0));
      setCacheWithId(id, op->getResult(0));
    }
  }

  bool valueAtIdIsInCache(const int64_t id) {
    return m_cache.count(id) != 0;
  }

  OwningOpRef<FuncOp> buildInitFunction();
  OwningOpRef<FuncOp> buildNextFunction();
  
 private: 
///===----------------------------------------------------------------------===//
/// Parse btor2 file
///===----------------------------------------------------------------------===//
  
  Btor2Parser *m_model = nullptr;
  FILE *m_modelFile = nullptr;

  std::vector<Btor2Line *> m_states;
  std::vector<Btor2Line *> m_bads;
  std::vector<Btor2Line *> m_inits;
  std::vector<Btor2Line *> m_nexts;
  std::vector<Btor2Line *> m_constraints;
  std::vector<Btor2Line *> m_lines;

  std::map<int64_t, Value> m_cache;

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
  bool needsMLIROp(Btor2Line * line);
  bool hasReturnValue(Btor2Line * line);
  void createNegateLine(int64_t curAt, const Value &child);
  Operation * createMLIR(const Btor2Line *line, 
                        const SmallVector<Value> &kids,
                        const SmallVector<unsigned> &arguments);

  // Builder wrappers
  Type getIntegerTypeOf(Btor2Line * line) {
    return m_builder.getIntegerType(line->sort.bitvec.width);
  }

  std::vector<Value> collectReturnValuesForInit(
                    const std::vector<Type> &returnTypes) {
    std::vector<Value> results(m_states.size(), nullptr);
    std::map<unsigned, Value> undefOpsBySort;
    for (unsigned i = 0, sz = m_states.size(); i < sz; ++i) {
      if (unsigned initLine = m_states.at(i)->init) {
        results[i] = getFromCacheById(initLine);
      } else {
        auto sort = returnTypes.at(i).getIntOrFloatBitWidth();
        if (undefOpsBySort.count(sort) == 0) {
            auto res = m_builder.create<btor::UndefOp>(m_unknownLoc,
                                        returnTypes.at(i));
            assert(res); 
            assert(res->getNumResults() == 1);
            undefOpsBySort[sort] = res->getResult(0);
        }
        results[i] = undefOpsBySort.at(sort);
        }
        assert(results[i]);
    }
    return results;
  }

  // Binary Operations
  template <typename btorOp>
  Operation * buildBinaryOp(const Value &lhs, const Value &rhs) {
    auto res = m_builder.create<btorOp>(m_unknownLoc, lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildComparisonOp(const btor::BtorPredicate pred,
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

  Operation * buildConcatOp(const Value &lhs, const Value &rhs) {
    auto newWidth = lhs.getType().getIntOrFloatBitWidth() +
               rhs.getType().getIntOrFloatBitWidth();
    Type resType = m_builder.getIntegerType(newWidth);
    auto res = m_builder.create<btor::ConcatOp>(m_unknownLoc, resType, lhs, rhs);
    return res;
  }

  // Unary Operations
  Operation * buildConstantOp(const unsigned width, 
                            const std::string &str,
                            const unsigned radix) {
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
                        m_builder.getIntegerAttr(type, value));
    return res;
  }

  template <typename btorOp>
  Operation * buildUnaryOp(const Value &val) {
    auto res = m_builder.create<btorOp>(m_unknownLoc, val);
    return res;
  }

  template <typename btorOp>
  Operation * buildReductionOp(const Value &val) {
    auto res = m_builder.create<btor::RedAndOp>(m_unknownLoc, 
                                m_builder.getIntegerType(1), val);
    return res;
  }

  void buildReturnOp(const std::vector<Value> &results) {
    m_builder.create<ReturnOp>(m_unknownLoc, results);
  }

   Operation * buildInputOp(const unsigned width) {
    Type type = m_builder.getIntegerType(width);
    mlir::APInt value(width, 0, 10);
    auto op = m_builder.create<btor::ConstantOp>(m_unknownLoc, type,
                        m_builder.getIntegerAttr(type, value));
    assert(op);
    assert(op->getNumResults() == 1);
    assert(op->getResult(0));
    Value constValue = op->getResult(0);
    auto res = m_builder.create<btor::InputOp>(m_unknownLoc, type,
                                m_builder.getI64IntegerAttr(0), 
                                constValue);
    return res;
  }

  // Indexed Operations
  Operation * buildSliceOp(const Value &val, 
                        const int64_t upper, 
                        const int64_t lower) {
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

  template <typename btorOp>
  Operation * buildExtOp(const Value &val,
                        const unsigned width) {
    auto res = m_builder.create<btorOp>(m_unknownLoc, val,
                            m_builder.getIntegerType(width));
    return res;
  }

  // Ternary Operations
  Operation * buildIteOp(const Value &condition, 
                        const Value &lhs, 
                        const Value &rhs) {
    auto res = m_builder.create<btor::IteOp>(m_unknownLoc, 
                                        condition, lhs, rhs);
    return res;
  }
};

/// Register the Btor translation
void registerFromBtorTranslation();

} // namespace btor
} // namespace mlir

#endif // TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
