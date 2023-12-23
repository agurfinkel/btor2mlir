//===----------------------------------------------------------------------===//
//
// This provides registration calls for Btor to Btor IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
#define TARGET_BTOR_BTORTOBTORIRTRANSLATION_H

#include "Dialect/Btor/IR/Btor.h"
#include "Dialect/Btor/IR/BtorTypes.h"
#include "Dialect/Btor/IR/BtorAttributes.h"

#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <vector>
#include <map>
#include <utility>

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
        m_sourceFile = m_builder.getStringAttr(s);
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
    m_cache[id] = value;
    assert(valueAtIdIsInCache(id));
  }

  void setCacheWithId(const int64_t id, Operation * op) {
    assert(hasReturnValue(getLineById(id)));
    assert(op);
    assert(op->getNumResults() == 1);
    assert(op->getResult(0));
    setCacheWithId(id, op->getResult(0));
  }

  bool valueAtIdIsInCache(const int64_t id) {
    return m_cache.count(id) != 0;
  }

  OwningOpRef<mlir::FuncOp> buildMainFunction();

  btor::BitVecType getBVType(Type opType) {
    return opType.dyn_cast<btor::BitVecType>();
  }
  
 private: 
///===----------------------------------------------------------------------===//
/// Parse btor2 file
///===----------------------------------------------------------------------===//
  
  Btor2Parser *m_model = nullptr;
  FILE *m_modelFile = nullptr;
  StringAttr m_sourceFile = nullptr;

  std::vector<Btor2Line *> m_states;
  std::vector<Btor2Line *> m_bads;
  std::vector<Btor2Line *> m_nexts;
  std::vector<Btor2Line *> m_constraints;
  std::vector<Btor2Line *> m_lines;

  std::map<int64_t, Btor2Line *> m_inits; // lets us quickly find the initLine from state->init
  std::map<int64_t, Value> m_cache;
  std::map<int64_t, Btor2Line *> m_sorts;
  std::map<unsigned, unsigned> m_inputs; // lineId -> input #
  std::map<unsigned, unsigned> map_bads; // lineId -> bad #

  std::pair <int,int> parseModelLine(Btor2Line *l, std::pair <int,int> numberedCommands);

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
  bool isStateArgumentOfInitOp(Btor2Line * cur, unsigned argumentId);
  void createNegateLine(int64_t curAt, const unsigned lineId, const Value &child);
  Operation * createMLIR(const Btor2Line *line, 
                        const SmallVector<Value> &kids,
                        const SmallVector<unsigned> &arguments);
  std::vector<Value> buildInitFunction(const std::vector<Type> &returnTypes);
  std::vector<Value> buildNextFunction(const std::vector<Type> &returnTypes, 
                                    Block *body);
  std::vector<Value> collectReturnValuesForInit(const std::vector<Type> &returnTypes);

  // Builder wrappers
  Type getTypeOf(const Btor2Line *line) {
    if (line->sort.tag == BTOR2_TAG_SORT_array) {
      auto shape = btor::BitVecType::get(m_context, m_sorts.at(line->sort.array.index)->sort.bitvec.width);
      auto elementType = btor::BitVecType::get(m_context,
        m_sorts.at(line->sort.array.element)->sort.bitvec.width);
      return btor::ArrayType::get(m_context, shape, elementType);
      ;
    }
    return btor::BitVecType::get(m_context, line->sort.bitvec.width);
  }

  // Binary Operations
  template <typename btorOp>
  Operation * buildBinaryOp(const Value &lhs, const Value &rhs, const unsigned  lineId) {
    auto res = m_builder.create<btorOp>(FileLineColLoc::get(m_sourceFile, lineId, 0),
                                      lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildComparisonOp(const btor::BtorPredicate pred,
                                const Value &lhs,
                                const Value &rhs,
                                const unsigned  lineId) {
    auto res = m_builder.create<btorOp>(FileLineColLoc::get(m_sourceFile, lineId, 0), 
                                       pred, lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildOverflowOp(const Value &lhs, const Value &rhs, const unsigned  lineId) {
    auto res = m_builder.create<btorOp>(FileLineColLoc::get(m_sourceFile, lineId, 0), 
                                    btor::BitVecType::get(m_context, 1),
                                    lhs, rhs);
    return res;
  }

  Operation * buildConcatOp(const Value &lhs, const Value &rhs, const unsigned  lineId) {
    auto newWidth = getBVType(lhs.getType()).getWidth() + getBVType(rhs.getType()).getWidth();
    Type resType = btor::BitVecType::get(m_context, newWidth);
    auto res = m_builder.create<btor::ConcatOp>(FileLineColLoc::get(m_sourceFile, lineId, 0),
                                                resType, lhs, rhs);
    return res;
  }

  // Unary Operations
  Operation * buildConstantOp(const unsigned width, 
                            const std::string &str,
                            const unsigned radix,
                            const unsigned  lineId) {
    Type type = btor::BitVecType::get(m_context, width);
    mlir::APInt value(width, 0, radix);
    if (str.compare("ones") == 0) {
      value.setAllBits();
    } else if (str.compare("one") == 0) {
      value = mlir::APInt(width, 1, radix);
    } else if (str.compare("zero") != 0) {
      value = mlir::APInt(width, str, radix);
    }
    auto res = m_builder.create<btor::ConstantOp>(
                        FileLineColLoc::get(m_sourceFile, lineId, 0),
                        type, m_builder.getIntegerAttr(m_builder.getIntegerType(width), value));
    return res;
  }

  template <typename btorOp>
  Operation * buildUnaryOp(const Value &val, const unsigned  lineId) {
    auto res = m_builder.create<btorOp>(
                    FileLineColLoc::get(m_sourceFile, lineId, 0),
                    val);
    return res;
  }

  template <typename btorOp>
  Operation * buildReductionOp(const Value &val, const unsigned  lineId) {
    auto res = m_builder.create<btorOp>(
                                FileLineColLoc::get(m_sourceFile, lineId, 0), 
                                btor::BitVecType::get(m_context, 1), val);
    return res;
  }

  void buildReturnOp(const std::vector<Value> &results) {
    m_builder.create<mlir::ReturnOp>(m_unknownLoc, results);
  }

  Operation * buildInputOp(const unsigned width, const unsigned lineId) {
    Type type = btor::BitVecType::get(m_context, width);
    auto res = m_builder.create<btor::InputOp>(
        FileLineColLoc::get(m_sourceFile, lineId, 0),
        type, 
        m_builder.getIntegerAttr(m_builder.getIntegerType(64), m_inputs.at(lineId)));
    return res;
  }

  Operation * buildAssertNotOp(const Value &val, const unsigned  lineId) {
    auto res = m_builder.create<btor::AssertNotOp>(
        FileLineColLoc::get(m_sourceFile, lineId, 0),
        val,
        m_builder.getIntegerAttr(m_builder.getIntegerType(64), map_bads.at(lineId)));
    return res;
  }

  // Indexed Operations
  Operation * buildSliceOp(const Value &val, 
                        const int64_t upper, 
                        const int64_t lower,
                        const unsigned  lineId) {
    btor::BitVecType opType = getBVType(val.getType());
    assert(opType.getWidth() > upper && upper >= lower);
    auto loc = FileLineColLoc::get(m_sourceFile, lineId, 0);

    auto resType = btor::BitVecType::get(m_context, upper - lower + 1);
    auto u = m_builder.create<btor::ConstantOp>(
        loc, opType,
        m_builder.getIntegerAttr(m_builder.getIntegerType(opType.getWidth()), upper));
    assert(u && u->getNumResults() == 1);
    auto l = m_builder.create<btor::ConstantOp>(
        loc, opType, 
        m_builder.getIntegerAttr(m_builder.getIntegerType(opType.getWidth()), lower));
    assert(l && l->getNumResults() == 1);

    auto res = m_builder.create<btor::SliceOp>(
                            loc, resType, val,
                            u->getResult(0), l->getResult(0));
    return res;
  }

  template <typename btorOp>
  Operation * buildExtOp(const Value &val,
                        const unsigned width,
                        const unsigned  lineId) {
    auto res = m_builder.create<btorOp>(
                            FileLineColLoc::get(m_sourceFile, lineId, 0), 
                            val, btor::BitVecType::get(m_context, width));
    return res;
  }

  // Ternary Operations
  Operation * buildIteOp(const Value &condition, 
                        const Value &lhs, 
                        const Value &rhs,
                        const unsigned  lineId) {
    auto res = m_builder.create<btor::IteOp>(
                          FileLineColLoc::get(m_sourceFile, lineId, 0),
                          condition, lhs, rhs);
    return res;
  }

  // Array Operations
  Operation *buildReadOp(const Value &array,
                        const Value &index,
                        const unsigned  lineId) {
    auto elementType = array.getType().cast<btor::ArrayType>().getElement();
    auto res =
        m_builder.create<btor::ReadOp>(
                        FileLineColLoc::get(m_sourceFile, lineId, 0),
                        elementType, array, index);
    return res;
  }

  Operation *buildWriteOp(const Value &array,
                          const Value &index,
                          const Value &value,
                          const unsigned  lineId) {
    auto res = m_builder.create<btor::WriteOp>(
                            FileLineColLoc::get(m_sourceFile, lineId, 0),
                            array.getType(),
                            value, array, index);
    return res;
  }

  // State Operations
  Operation *buildNDStateOp(const Btor2Line *line, const unsigned lineId) {
    if (auto initLine = line->init)
      return nullptr;
    auto it = std::find(m_states.begin(), m_states.end(), line);
    assert (it != m_states.end());
    auto index = it - m_states.begin();
    if (line->sort.tag == BTOR2_TAG_SORT_array) {
      auto res = m_builder.create<btor::ArrayOp>(FileLineColLoc::get(m_sourceFile, lineId, 0),
                    getTypeOf(line));
      return res;
    } else {
      auto res = m_builder.create<btor::NDStateOp>(FileLineColLoc::get(m_sourceFile, lineId, 0),
                    getTypeOf(line),
                    m_builder.getIntegerAttr(m_builder.getIntegerType(64), index));
      return res;
    }
  }

  void buildInitOp(const Btor2Line *line, const Value &initValue, const unsigned lineId) {
    auto stateId = line->args[0];
    if (line->sort.tag == BTOR2_TAG_SORT_array) {
      auto initValueType = initValue.getType();
      if (initValueType.isa<btor::ArrayType>()) {
        setCacheWithId(stateId, initValue);
        return;
      }
      auto res = m_builder.create<btor::InitArrayOp>(FileLineColLoc::get(m_sourceFile, lineId, 0),
                    getTypeOf(line), initValue);
      assert(res);
      assert(res->getNumResults() == 1);
      setCacheWithId(stateId, res->getResult(0));
      return;
    }
    setCacheWithId(stateId, initValue);
  }
};

/// Register the Btor translation
void registerFromBtorTranslation();

} // namespace btor
} // namespace mlir

#endif // TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
