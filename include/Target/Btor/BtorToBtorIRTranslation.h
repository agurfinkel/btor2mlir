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

  Deserialize(MLIRContext *context, const std::string &s) : context(context), 
    builder(OpBuilder(context)), unknownLoc(UnknownLoc::get(context)) {
        modelFile = fopen(s.c_str(), "r");
    }

  ~Deserialize() {
      if (model) {
        btor2parser_delete(model);
      }
      if (modelFile) {
        fclose(modelFile);
      }
  }

///===----------------------------------------------------------------------===//
/// Parse btor2 file
///===----------------------------------------------------------------------===//

  std::vector<Btor2Line *> inputs;
  std::vector<Btor2Line *> states;
  std::vector<Btor2Line *> bads;
  std::vector<Btor2Line *> inits;
  std::vector<Btor2Line *> nexts;
  std::vector<Btor2Line *> constraints;
 
  std::map<int64_t, Btor2Line *> reachedLines;
  
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
  
  Btor2Parser *model = nullptr;
  FILE *modelFile = nullptr;

  void parseModelLine(Btor2Line *l);

///===----------------------------------------------------------------------===//
/// Create MLIR module
///===----------------------------------------------------------------------===//

  MLIRContext *context;
  OpBuilder builder;
  Location unknownLoc;
  
  void toOp(Btor2Line *line);
  bool isValidChild(Btor2Line * line);
  void createNegateLine(int64_t curAt, Value &child);
  Operation * createMLIR(const Btor2Line *line, const int64_t *kids);

  template <typename btorOp>
  Operation * buildBinaryOp(Value &lhs, Value &rhs) {
    auto res = builder.create<btorOp>(unknownLoc, lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildComparisonOp(btor::BtorPredicate pred,
                                Value &lhs, Value &rhs) {
    auto res = builder.create<btorOp>(unknownLoc, pred, lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildOverflowOp(Value &lhs, Value &rhs) {
    auto res = builder.create<btorOp>(unknownLoc, 
                                    builder.getIntegerType(1), 
                                    lhs, rhs);
    return res;
  }

  Operation * buildConstantOp(uint32_t width, std::string str, uint32_t radix) {
    Type type = builder.getIntegerType(width);
    mlir::APInt value(width, 0, radix);
    if (str.compare("ones") == 0) {
      value.setAllBits();
    } else if (str.compare("one") == 0) {
      value = mlir::APInt(width, 1, radix);
    } else if (str.compare("zero") != 0) {
      value = mlir::APInt(width, str, radix);
    }
    auto res = builder.create<btor::ConstantOp>(unknownLoc, type,
                        builder.getIntegerAttr(type, value.getLimitedValue()));
    return res;
  }

  Operation * buildConcatOp(Value &lhs, Value &rhs) {
    auto newWidth = lhs.getType().getIntOrFloatBitWidth() +
               rhs.getType().getIntOrFloatBitWidth();
    Type resType = builder.getIntegerType(newWidth);
    auto res = builder.create<btor::ConcatOp>(unknownLoc, resType, lhs, rhs);
    return res;
  }

  Operation * buildSliceOp(Value &val, int64_t upper, int64_t lower) {
    auto opType = val.getType();
    auto operandWidth = opType.getIntOrFloatBitWidth();
    assert(operandWidth > upper && upper >= lower);

    auto resType = builder.getIntegerType(upper - lower + 1);
    auto u = builder.create<btor::ConstantOp>(
        unknownLoc, opType, builder.getIntegerAttr(opType, upper));
    assert(u && u->getNumResults() == 1);
    auto l = builder.create<btor::ConstantOp>(
        unknownLoc, opType, builder.getIntegerAttr(opType, lower));
    assert(l && l->getNumResults() == 1);

    auto res = builder.create<btor::SliceOp>(unknownLoc, resType, val,
                                        u->getResult(0), l->getResult(0));
    return res;
  }
};

/// Register the Btor translation
void registerFromBtorTranslation();

} // namespace btor
} // namespace mlir

#endif // TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
