#include "Target/Btor/BtorToBtorIRTranslation.h"
#include "Dialect/Btor/IR/BtorDialect.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>
#include <string>

using namespace mlir;
using namespace mlir::btor;

void Deserialize::filterInits() {
  size_t i = 0;
  for (size_t j = 0, sz = inits.size(); j < sz; ++j) {
      if (inits.at(j)) {
          inits[i++] = inits.at(j);
      }
  }
  inits.resize(i);
}

void Deserialize::filterNexts() {
  size_t i = 0;
  for (size_t j = 0, sz = nexts.size(); j < sz; ++j) {
      if (nexts.at(j)) {
        nexts[i++] = nexts.at(j);
      }
  }
  nexts.resize(i);
}

void Deserialize::parseModelLine(Btor2Line *l) {
  reachedLines[l->id] = l;
  switch (l->tag) {
  case BTOR2_TAG_bad:
    bads.push_back(l);
    break;

  case BTOR2_TAG_constraint:
    constraints.push_back(l);
    break;

  case BTOR2_TAG_init:
    inits[l->args[0]] = l;
    break;

  case BTOR2_TAG_input:
    inputs.push_back(l);
    break;

  case BTOR2_TAG_next:
    nexts[l->args[0]] = l;
    break;

  case BTOR2_TAG_state:
    states.push_back(l);
    break;

  default:
    break;
  }
}

bool Deserialize::parseModel() {
  assert(modelFile);
  model = btor2parser_new();
  if (!btor2parser_read_lines(model, modelFile)) {
    std::cerr << "parse error at: " << btor2parser_error(model) << "\n";
    return false;
  }
  auto numLines = btor2parser_max_id(model);
  inits.resize(numLines, nullptr);
  nexts.resize(numLines, nullptr);
  Btor2LineIterator it = btor2parser_iter_init(model);
  Btor2Line *line;
  while ((line = btor2parser_iter_next(&it))) {
    parseModelLine(line);
  }

  for (size_t i = 0; i < states.size(); i++) {
    Btor2Line *state = states.at(i);
    if (!nexts[state->id]) {
      std::cerr << "state " << state->id << " without next function\n";
      return false;
    }
  }
  return true;
}

///===----------------------------------------------------------------------===//
/// This function's goal is to create the MLIR Operation that corresponds to 
/// the given Btor2Line*, cur, into the basic block designated by the provided 
/// builder.
///
/// e.x:
///     Operation * res = createMLIR(cur, cur->args);
///
///===----------------------------------------------------------------------===//
Operation * Deserialize::createMLIR(const Btor2Line *line, const int64_t *kids) {
  Operation *res = nullptr;

  switch (line->tag) {
  // binary ops
  case BTOR2_TAG_slt:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::slt, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_slte:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::sle, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sgt:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::sgt, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sgte:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::sge, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_neq:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::ne, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_eq:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::eq, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ugt:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::ugt, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ugte:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::uge, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ult:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::ult, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ulte:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::ule, 
                            cache.at(kids[0]), cache.at(kids[1]));
    break;
  
  case BTOR2_TAG_concat:
    res = buildConcatOp(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_add:
    res = buildBinaryOp<btor::AddOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_and:
    res = buildBinaryOp<btor::AndOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_implies:
    res = buildBinaryOp<btor::ImpliesOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_iff:
    res = buildBinaryOp<btor::IffOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_nand:
    res = buildBinaryOp<btor::NandOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_nor:
    res = buildBinaryOp<btor::NorOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_or:
    res = buildBinaryOp<btor::OrOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sdiv:
    res = buildBinaryOp<btor::SDivOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_srem:
    res = buildBinaryOp<btor::SRemOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sub:
    res = buildBinaryOp<btor::SubOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_udiv:
    res = buildBinaryOp<btor::UDivOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_urem:
    res = buildBinaryOp<btor::URemOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_mul:
    res = buildBinaryOp<btor::MulOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_smod:
    res = buildBinaryOp<btor::SModOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_xnor:
    res = buildBinaryOp<btor::XnorOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_xor:
    res = buildBinaryOp<btor::XOrOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sll:
    res = buildBinaryOp<btor::ShiftLLOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sra:
    res = buildBinaryOp<btor::ShiftRAOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_srl:
    res = buildBinaryOp<btor::ShiftRLOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_rol:
    res = buildBinaryOp<btor::RotateLOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ror:
    res = buildBinaryOp<btor::RotateROp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_saddo:
    res = buildOverflowOp<btor::SAddOverflowOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sdivo:
    res = buildOverflowOp<btor::SDivOverflowOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_smulo:
    res = buildOverflowOp<btor::SMulOverflowOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ssubo:
    res = buildOverflowOp<btor::SSubOverflowOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_uaddo:
    res = buildOverflowOp<btor::UAddOverflowOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_umulo:
    res = buildOverflowOp<btor::UMulOverflowOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_usubo:
    res = buildOverflowOp<btor::USubOverflowOp>(cache.at(kids[0]), cache.at(kids[1]));
    break;

  // unary ops
  case BTOR2_TAG_dec:
    res = builder.create<btor::DecOp>(unknownLoc, cache.at(kids[0]));
    break;
  case BTOR2_TAG_inc:
    res = builder.create<btor::IncOp>(unknownLoc, cache.at(kids[0]));
    break;
  case BTOR2_TAG_neg:
    res = builder.create<btor::NegOp>(unknownLoc, cache.at(kids[0]));
    break;
  case BTOR2_TAG_not:
    res = builder.create<btor::NotOp>(unknownLoc, cache.at(kids[0]));
    break;
  case BTOR2_TAG_redand:
    res = builder.create<btor::RedAndOp>(unknownLoc, builder.getIntegerType(1),
                                         cache.at(kids[0]));
    break;
  case BTOR2_TAG_redor:
    res = builder.create<btor::RedOrOp>(unknownLoc, builder.getIntegerType(1),
                                        cache.at(kids[0]));
    break;
  case BTOR2_TAG_redxor:
    res = builder.create<btor::RedXorOp>(unknownLoc, builder.getIntegerType(1),
                                         cache.at(kids[0]));
    break;
  case BTOR2_TAG_const:
    res = buildConstantOp(line->sort.bitvec.width,
                        std::string(line->constant), 2);
    break;
  case BTOR2_TAG_constd:
    res = buildConstantOp(line->sort.bitvec.width, 
                        std::string(line->constant), 10);
    break;
  case BTOR2_TAG_consth:
    res = buildConstantOp(line->sort.bitvec.width, 
                        std::string(line->constant), 16);
    break;
  case BTOR2_TAG_one:
    res = buildConstantOp(line->sort.bitvec.width, 
                        std::string("one"), 10);
    break;
  case BTOR2_TAG_ones:
    res = buildConstantOp(line->sort.bitvec.width,
                        std::string("ones"), 10);
    break;
  case BTOR2_TAG_zero:
    res = buildConstantOp(line->sort.bitvec.width, 
                        std::string("zero"), 10);
    break;
  case BTOR2_TAG_bad:
    res = builder.create<btor::AssertNotOp>(unknownLoc, cache.at(kids[0]));
    break;

  // indexed ops
  case BTOR2_TAG_slice:
    res = buildSliceOp(cache.at(kids[0]), kids[1], kids[2]);
    break;
  case BTOR2_TAG_sext:
    res = builder.create<btor::SExtOp>(
        unknownLoc, cache.at(kids[0]),
        builder.getIntegerType(line->sort.bitvec.width));
    break;
  case BTOR2_TAG_uext:
    res = builder.create<btor::UExtOp>(
        unknownLoc, cache.at(kids[0]),
        builder.getIntegerType(line->sort.bitvec.width));
    break;

  // ternary ops
  case BTOR2_TAG_ite:
    res = builder.create<btor::IteOp>(unknownLoc, cache.at(kids[0]),
                                      cache.at(kids[1]), cache.at(kids[2]));
    break;

  // unmapped ops
  case BTOR2_TAG_read:
  case BTOR2_TAG_constraint:
  case BTOR2_TAG_init:
  case BTOR2_TAG_input:
  case BTOR2_TAG_next:
  case BTOR2_TAG_sort:
  case BTOR2_TAG_state:
  case BTOR2_TAG_fair:
  case BTOR2_TAG_justice:
  case BTOR2_TAG_output:
  case BTOR2_TAG_write:
  default:
    break;
  }
  return res;
}

void Deserialize::createNegateLine(int64_t curAt, Value child) {
  auto res = builder.create<btor::NotOp>(unknownLoc, cache.at(curAt * -1));
  assert(res && res->getNumResults() == 1);
  cache[curAt] = res->getResult(0);
}

bool Deserialize::isValidChild(Btor2Line * line) {
  auto tag = reachedLines.at(line->id)->tag;
  if (tag == BTOR2_TAG_init || tag == BTOR2_TAG_constraint ||
      tag == BTOR2_TAG_next || tag == BTOR2_TAG_read ||
      tag == BTOR2_TAG_state || tag == BTOR2_TAG_input ||
      tag == BTOR2_TAG_sort || tag == BTOR2_TAG_fair ||
      tag == BTOR2_TAG_justice || tag == BTOR2_TAG_output ||
      tag == BTOR2_TAG_write) {
    return false;
  }
  return true;
}

///===----------------------------------------------------------------------===//
/// This function's goal is to add the MLIR Operation that corresponds to 
/// the given Btor2Line* into the basic block designated by the provided 
/// builder. Then, the MLIR Value of the newly minted operation is added
/// into our cache for future reference within the basic block. 
///
/// e.x:
///      for (auto it = nexts.begin(); it != nexts.end(); ++it) {
///          toOp(*it);
///      }
///
///  We can see that for each next operation in btor2, we will compute all
///  the prerequisite operations before storing the result in our cache
///===----------------------------------------------------------------------===//
void Deserialize::toOp(Btor2Line *line) {
  if (cache.find(line->id) != cache.end()) {
    return;
  }

  Operation *res = nullptr;
  std::vector<Btor2Line *> todo;
  todo.push_back(line);
  while (!todo.empty()) {
    auto cur = todo.back();
    uint32_t oldsize = todo.size();
    for (uint32_t i = 0; i < cur->nargs; ++i) {
      if (cur->args[i] > 0 
      && !isValidChild(reachedLines.at(cur->args[i]))) {
        continue;
      }

      if (cache.find(cur->args[i]) == cache.end()) {
        if (cur->args[i] < 0) {
          // if original operation is cached, negate it
          if (cache.find(cur->args[i] * -1) != cache.end()) {
            createNegateLine(cur->args[i], cache.at(cur->args[i] * -1)); 
          } else {
            todo.push_back(reachedLines.at(cur->args[i] * -1));
          }
        } else {
          todo.push_back(reachedLines.at(cur->args[i]));
        }
      }
    }
    if (todo.size() != oldsize) {
      continue;
    }
    if (!isValidChild(cur) 
    || cache.find(cur->id) != cache.end()) {
      todo.pop_back();
      continue;
    }
    res = createMLIR(cur, cur->args);
    assert(res && res->getNumResults() < 2 && res->getResult(0));
    cache[cur->id] = res->getResult(0);
    todo.pop_back();
  }
}

OwningOpRef<FuncOp> Deserialize::buildInitFunction() {
  // collect the return types for our init function
  std::vector<Type> returnTypes(states.size(), nullptr);
  for (uint32_t i = 0; i < states.size(); ++i) {
    returnTypes[i] = builder.getIntegerType(states.at(i)->sort.bitvec.width);
    assert(returnTypes[i]);
  }
  ArrayRef<Type> outputs(returnTypes);

  // create init function signature
  OperationState state(unknownLoc, FuncOp::getOperationName());
  FuncOp::build(builder, state, "init",
                FunctionType::get(context, {}, outputs));
  OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));

  // create basic block and accompanying builder
  Region &region = funcOp->getBody();
  OpBuilder::InsertionGuard guard(builder);
  auto *body = builder.createBlock(&region);
  builder.setInsertionPointToStart(body);

  // clear cache so that values are mapped to the right Basic Block
  cache.clear();
  for (auto it = inits.begin(); it != inits.end(); ++it) {
    toOp(*it);
  }

  // close with a fitting returnOp
  std::vector<Value> testResults(states.size(), nullptr);
  std::map<uint32_t, Value> undefOpsBySort;
  uint32_t j = 0; // counters over inits vector
  for (uint32_t i = 0, sz = states.size(); i < sz; ++i) {
    if (states.at(i)->init > 0) {
      // get the result of init's second argument since
      // that is what we assign our state to  
      testResults[i] = cache.at(inits.at(j++)->args[1]);
    } else {
      auto sort = returnTypes.at(i).getIntOrFloatBitWidth();
      if (undefOpsBySort.find(sort) == undefOpsBySort.end()) {
          auto res = builder.create<btor::UndefOp>(unknownLoc,
                                    returnTypes.at(i));
          assert(res && res->getNumResults() == 1);
          undefOpsBySort[sort] = res->getResult(0);
      }
      testResults[i] = undefOpsBySort.at(sort);
    }
    assert(testResults[i]);
  }
  ArrayRef<Value> results(testResults);

  builder.create<ReturnOp>(unknownLoc, ValueRange({results}));

  return funcOp;
}

OwningOpRef<FuncOp> Deserialize::buildNextFunction() {
  // collect the return types for our init function
  std::vector<Type> returnTypes(nexts.size(), nullptr);
  for (uint32_t i = 0; i < nexts.size(); ++i) {
    returnTypes[i] = builder.getIntegerType(nexts.at(i)->sort.bitvec.width);
    assert(returnTypes[i]);
  }
  ArrayRef<Type> outputs(returnTypes);

  // create next function signature
  OperationState state(unknownLoc, FuncOp::getOperationName());
  FuncOp::build(builder, state, "next",
                FunctionType::get(context, outputs, outputs));
  OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));
  Region &region = funcOp->getBody();
  OpBuilder::InsertionGuard guard(builder);
  auto *body = builder.createBlock(&region, {}, TypeRange({outputs}));
  builder.setInsertionPointToStart(body);

  // clear cache so that values are mapped to the right Basic Block
  cache.clear();
  // initialize states with block arguments
  for (uint32_t i = 0; i < nexts.size(); ++i) {
    cache[nexts.at(i)->args[0]] = body->getArguments()[i];
  }

  // start with nexts, then add bads, for logic sharing
  for (auto it = nexts.begin(); it != nexts.end(); ++it) {
    toOp(*it);
  }
  for (auto it = bads.begin(); it != bads.end(); ++it) {
    toOp(*it);
  }

  // close with a fitting returnOp
  std::vector<Value> testResults(nexts.size(), nullptr);
  for (uint32_t i = 0; i < nexts.size(); ++i) {
    testResults[i] = cache.at(nexts.at(i)->args[1]);
    assert(testResults[i]);
  }
  ArrayRef<Value> results(testResults);

  builder.create<ReturnOp>(unknownLoc, ValueRange({results}));
  return funcOp;
}

static OwningModuleRef deserializeModule(const llvm::MemoryBuffer *input,
                                         MLIRContext *context) {
  context->loadDialect<btor::BtorDialect>();
  context->loadDialect<StandardOpsDialect>();

  OwningModuleRef owningModule(ModuleOp::create(FileLineColLoc::get(
      context, input->getBufferIdentifier(), /*line=*/0, /*column=*/0)));

  auto btorFile = fopen(input->getBufferIdentifier().str().c_str(), "r");
  if (btorFile != NULL) {
    Deserialize deserialize(context);
    deserialize.setModelFile(btorFile);
    if (!deserialize.parseModel())
      return {};

    // extract relevant inits and nexts
    deserialize.filterInits();
    deserialize.filterNexts();

    OwningOpRef<FuncOp> initFunc = deserialize.buildInitFunction();
    if (!initFunc)
      return {};

    OwningOpRef<FuncOp> nextFunc = deserialize.buildNextFunction();
    if (!nextFunc)
      return {};

    owningModule->getBody()->push_front(nextFunc.release());
    owningModule->getBody()->push_front(initFunc.release());
  }

  return owningModule;
}

namespace mlir {
namespace btor {
void registerFromBtorTranslation() {
  TranslateToMLIRRegistration fromBtor(
      "import-btor", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
        return deserializeModule(
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context);
      });
}
} // namespace btor
} // namespace mlir
