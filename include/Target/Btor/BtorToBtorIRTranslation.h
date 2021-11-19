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

  Deserialize(MLIRContext *context) : context(context), 
    builder(OpBuilder(context)), unknownLoc(UnknownLoc::get(context)) {}

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
  
  bool parseModel();
  void setModelFile(FILE * file) { modelFile = file; }
  void filterInits();
  void filterNexts();

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
  void createNegateLine(int64_t curAt, Value child);
  Operation * createMLIR(const Btor2Line *line, const int64_t *kids);

  template <typename btorOp>
  Operation * buildBinaryOp(Value lhs, Value rhs) {
    auto res = builder.create<btorOp>(unknownLoc, lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildComparisonOp(btor::BtorPredicate pred,
                                Value lhs, Value rhs) {
    auto res = builder.create<btorOp>(unknownLoc, pred, lhs, rhs);
    return res;
  }

  template <typename btorOp>
  Operation * buildOverflowOp(Value lhs, Value rhs) {
    auto res = builder.create<btorOp>(unknownLoc, 
                                    builder.getIntegerType(1), 
                                    lhs, rhs);
    return res;
  }
};

/// Register the Btor translation
void registerFromBtorTranslation();

} // namespace btor
} // namespace mlir

#endif // TARGET_BTOR_BTORTOBTORIRTRANSLATION_H
