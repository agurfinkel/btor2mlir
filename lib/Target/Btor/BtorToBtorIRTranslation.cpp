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

#include <assert.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "btor2tools/btor2parser/btor2parser.h"
#include "btor2tools/btorsimbv.h"
#include "btor2tools/btorsimrng.h"
#include "btor2tools/btorsimstate.h"
#include "btor2tools/btorsimvcd.h"

using namespace mlir;
using namespace mlir::btor;

static FILE *model_file;
static Btor2Parser *model;

static std::vector<Btor2Line *> inputs;
static std::vector<Btor2Line *> states;
static std::vector<Btor2Line *> bads;
static std::vector<Btor2Line *> constraints;
static std::vector<Btor2Line *> justices;

static std::vector<int64_t> reached_bads;
static int64_t num_unreached_bads;

static int64_t num_format_lines;
static std::vector<Btor2Line *> inits;
static std::vector<Btor2Line *> nexts;

static std::map<int64_t, Btor2Line *> reached_lines;
static std::map<int64_t, Value> cache;

static void parse_model_line(Btor2Line *l) {
  reached_lines[l->id] = l;
  switch (l->tag) {
  case BTOR2_TAG_bad: {
    bads.push_back(l);
    reached_bads.push_back(-1);
    num_unreached_bads++;
  } break;

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

  case BTOR2_TAG_sort: {
    switch (l->sort.tag) {
    case BTOR2_TAG_SORT_bitvec:
    case BTOR2_TAG_SORT_array:
    default:
      break;
    }
  } break;

  case BTOR2_TAG_state:
    states.push_back(l);
    break;

  case BTOR2_TAG_add:
  case BTOR2_TAG_and:
  case BTOR2_TAG_concat:
  case BTOR2_TAG_const:
  case BTOR2_TAG_constd:
  case BTOR2_TAG_consth:
  case BTOR2_TAG_dec:
  case BTOR2_TAG_eq:
  case BTOR2_TAG_implies:
  case BTOR2_TAG_inc:
  case BTOR2_TAG_ite:
  case BTOR2_TAG_mul:
  case BTOR2_TAG_nand:
  case BTOR2_TAG_neg:
  case BTOR2_TAG_neq:
  case BTOR2_TAG_nor:
  case BTOR2_TAG_not:
  case BTOR2_TAG_one:
  case BTOR2_TAG_ones:
  case BTOR2_TAG_or:
  case BTOR2_TAG_output:
  case BTOR2_TAG_redand:
  case BTOR2_TAG_redor:
  case BTOR2_TAG_redxor:
  case BTOR2_TAG_sdiv:
  case BTOR2_TAG_sext:
  case BTOR2_TAG_sgt:
  case BTOR2_TAG_sgte:
  case BTOR2_TAG_iff:
  case BTOR2_TAG_slice:
  case BTOR2_TAG_sll:
  case BTOR2_TAG_slt:
  case BTOR2_TAG_slte:
  case BTOR2_TAG_sra:
  case BTOR2_TAG_srem:
  case BTOR2_TAG_srl:
  case BTOR2_TAG_sub:
  case BTOR2_TAG_udiv:
  case BTOR2_TAG_uext:
  case BTOR2_TAG_ugt:
  case BTOR2_TAG_ugte:
  case BTOR2_TAG_ult:
  case BTOR2_TAG_ulte:
  case BTOR2_TAG_urem:
  case BTOR2_TAG_xnor:
  case BTOR2_TAG_xor:
  case BTOR2_TAG_zero:
  case BTOR2_TAG_read:
  case BTOR2_TAG_write:
  case BTOR2_TAG_fair:
  case BTOR2_TAG_justice:
  case BTOR2_TAG_rol:
  case BTOR2_TAG_ror:
  case BTOR2_TAG_saddo:
  case BTOR2_TAG_sdivo:
  case BTOR2_TAG_smod:
  case BTOR2_TAG_smulo:
  case BTOR2_TAG_ssubo:
  case BTOR2_TAG_uaddo:
  case BTOR2_TAG_umulo:
  case BTOR2_TAG_usubo:
  default:
    break;
  }
}

static void parse_model() {
  assert(model_file);
  model = btor2parser_new();
  if (!btor2parser_read_lines(model, model_file)) {
    std::cerr << "parse error at: " << btor2parser_error(model) << "\n";
    exit(1);
  }
  num_format_lines = btor2parser_max_id(model);
  inits.resize(num_format_lines, nullptr);
  nexts.resize(num_format_lines, nullptr);
  Btor2LineIterator it = btor2parser_iter_init(model);
  Btor2Line *line;
  while ((line = btor2parser_iter_next(&it)))
    parse_model_line(line);

  for (size_t i = 0; i < states.size(); i++) {
    Btor2Line *state = states[i];
    if (!nexts[state->id]) {
      std::cerr << "state " << state->id << " without next function\n";
      exit(1);
    }
  }
}

void filterInits() {
  std::vector<Btor2Line *> filteredInits;
  for (auto it = inits.begin(); it != inits.end(); ++it) {
    if (*it) {
      filteredInits.push_back(*it);
    }
  }
  inits.clear();
  inits = filteredInits;
}

void filterNexts() {
  std::vector<Btor2Line *> filteredNexts;
  for (auto it = nexts.begin(); it != nexts.end(); ++it) {
    if (*it) {
      filteredNexts.push_back(*it);
    }
  }
  nexts.clear();
  nexts = filteredNexts;
}

///===----------------------------------------------------------------------===//
/// This function's goal is to create the MLIR Operation that corresponds to 
/// the given Btor2Line*, cur, into the basic block designated by the provided 
/// builder.
///
/// e.x:
///     Operation * res = createMLIR(cur, builder, cur->args);
///
///===----------------------------------------------------------------------===//
Operation *createMLIR(const Btor2Line *line, 
                      OpBuilder &builder,
                      const int64_t *kids) {
  Location unknownLoc = UnknownLoc::get(builder.getContext());
  Operation *res = nullptr;

  switch (line->tag) {
  // binary ops
  case BTOR2_TAG_slt:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::slt,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_slte:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::sle,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sgt:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::sgt,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_sgte:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::sge,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_neq:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::ne,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_eq:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::eq,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ugt:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::ugt,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ugte:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::uge,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ult:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::ult,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_ulte:
    res = builder.create<btor::CmpOp>(unknownLoc, btor::BtorPredicate::ule,
                                      cache.at(kids[0]), cache.at(kids[1]));
    break;
  case BTOR2_TAG_add:
    res = builder.create<btor::AddOp>(unknownLoc, cache.at(kids[0]),
                                      cache.at(kids[1]));
    break;
  case BTOR2_TAG_and:
    res = builder.create<btor::AndOp>(unknownLoc, cache.at(kids[0]),
                                      cache.at(kids[1]));
    break;
  case BTOR2_TAG_concat: {
    Value lhs = cache.at(kids[0]), rhs = cache.at(kids[1]);
    auto sum = lhs.getType().getIntOrFloatBitWidth() +
               rhs.getType().getIntOrFloatBitWidth();
    auto resType = builder.getIntegerType(sum);
    res = builder.create<btor::ConcatOp>(unknownLoc, resType, lhs, rhs);
  } break;
  case BTOR2_TAG_implies:
    res = builder.create<btor::ImpliesOp>(unknownLoc, cache.at(kids[0]),
                                          cache.at(kids[1]));
    break;
  case BTOR2_TAG_iff:
    res = builder.create<btor::IffOp>(unknownLoc, cache.at(kids[0]),
                                      cache.at(kids[1]));
    break;
  case BTOR2_TAG_nand:
    res = builder.create<btor::NandOp>(unknownLoc, cache.at(kids[0]),
                                       cache.at(kids[1]));
    break;
  case BTOR2_TAG_nor:
    res = builder.create<btor::NorOp>(unknownLoc, cache.at(kids[0]),
                                      cache.at(kids[1]));
    break;
  case BTOR2_TAG_or:
    res = builder.create<btor::OrOp>(unknownLoc, cache.at(kids[0]),
                                     cache.at(kids[1]));
    break;
  case BTOR2_TAG_sdiv:
    res = builder.create<btor::SDivOp>(unknownLoc, cache.at(kids[0]),
                                       cache.at(kids[1]));
    break;
  case BTOR2_TAG_srem:
    res = builder.create<btor::SRemOp>(unknownLoc, cache.at(kids[0]),
                                       cache.at(kids[1]));
    break;
  case BTOR2_TAG_sub:
    res = builder.create<btor::SubOp>(unknownLoc, cache.at(kids[0]),
                                      cache.at(kids[1]));
    break;
  case BTOR2_TAG_udiv:
    res = builder.create<btor::UDivOp>(unknownLoc, cache.at(kids[0]),
                                       cache.at(kids[1]));
    break;
  case BTOR2_TAG_urem:
    res = builder.create<btor::URemOp>(unknownLoc, cache.at(kids[0]),
                                       cache.at(kids[1]));
    break;
  case BTOR2_TAG_mul:
    res = builder.create<btor::MulOp>(unknownLoc, cache.at(kids[0]),
                                      cache.at(kids[1]));
    break;
  case BTOR2_TAG_smod:
    res = builder.create<btor::SModOp>(unknownLoc, cache.at(kids[0]),
                                       cache.at(kids[1]));
    break;
  case BTOR2_TAG_xnor:
    res = builder.create<btor::XnorOp>(unknownLoc, cache.at(kids[0]),
                                       cache.at(kids[1]));
    break;
  case BTOR2_TAG_xor:
    res = builder.create<btor::XOrOp>(unknownLoc, cache.at(kids[0]),
                                      cache.at(kids[1]));
    break;
  case BTOR2_TAG_sll:
    res = builder.create<btor::ShiftLLOp>(unknownLoc, cache.at(kids[0]),
                                          cache.at(kids[1]));
    break;
  case BTOR2_TAG_sra:
    res = builder.create<btor::ShiftRAOp>(unknownLoc, cache.at(kids[0]),
                                          cache.at(kids[1]));
    break;
  case BTOR2_TAG_srl:
    res = builder.create<btor::ShiftRLOp>(unknownLoc, cache.at(kids[0]),
                                          cache.at(kids[1]));
    break;
  case BTOR2_TAG_rol:
    res = builder.create<btor::RotateLOp>(unknownLoc, cache.at(kids[0]),
                                          cache.at(kids[1]));
    break;
  case BTOR2_TAG_ror:
    res = builder.create<btor::RotateROp>(unknownLoc, cache.at(kids[0]),
                                          cache.at(kids[1]));
    break;
  case BTOR2_TAG_saddo:
    res = builder.create<btor::SAddOverflowOp>(
        unknownLoc, builder.getIntegerType(1), cache.at(kids[0]),
        cache.at(kids[1]));
    break;
  case BTOR2_TAG_sdivo:
    res = builder.create<btor::SDivOverflowOp>(
        unknownLoc, builder.getIntegerType(1), cache.at(kids[0]),
        cache.at(kids[1]));
    break;
  case BTOR2_TAG_smulo:
    res = builder.create<btor::SMulOverflowOp>(
        unknownLoc, builder.getIntegerType(1), cache.at(kids[0]),
        cache.at(kids[1]));
    break;
  case BTOR2_TAG_ssubo:
    res = builder.create<btor::SSubOverflowOp>(
        unknownLoc, builder.getIntegerType(1), cache.at(kids[0]),
        cache.at(kids[1]));
    break;
  case BTOR2_TAG_uaddo:
    res = builder.create<btor::UAddOverflowOp>(
        unknownLoc, builder.getIntegerType(1), cache.at(kids[0]),
        cache.at(kids[1]));
    break;
  case BTOR2_TAG_umulo:
    res = builder.create<btor::UMulOverflowOp>(
        unknownLoc, builder.getIntegerType(1), cache.at(kids[0]),
        cache.at(kids[1]));
    break;
  case BTOR2_TAG_usubo:
    res = builder.create<btor::USubOverflowOp>(
        unknownLoc, builder.getIntegerType(1), cache.at(kids[0]),
        cache.at(kids[1]));
    break;

  // unary ops
  case BTOR2_TAG_const: {
    auto opType = builder.getIntegerType(line->sort.bitvec.width);
    res = builder.create<btor::ConstantOp>(
        unknownLoc, opType,
        builder.getIntegerAttr(opType, line->constant[0] - '0'));
  } break;
  case BTOR2_TAG_constd: {
    auto opType = builder.getIntegerType(line->sort.bitvec.width);
    std::string input(line->constant);
    res = builder.create<btor::ConstantOp>(
        unknownLoc, opType, builder.getIntegerAttr(opType, std::stoi(input)));
  } break;
  case BTOR2_TAG_consth: {
    auto opType = builder.getIntegerType(line->sort.bitvec.width);
    std::string input("0x");
    input += line->constant;
    res = builder.create<btor::ConstantOp>(
        unknownLoc, opType,
        builder.getIntegerAttr(opType, std::stoi(input, nullptr, 16)));
  } break;

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
  case BTOR2_TAG_one: {
    auto opType = builder.getIntegerType(line->sort.bitvec.width);
    res = builder.create<btor::ConstantOp>(unknownLoc, opType,
                                           builder.getIntegerAttr(opType, 1));
  } break;
  case BTOR2_TAG_ones: {
    auto width = line->sort.bitvec.width;
    auto opType = builder.getIntegerType(width);
    auto value = pow(2, width) - 1;
    res = builder.create<btor::ConstantOp>(
        unknownLoc, opType, builder.getIntegerAttr(opType, value));
  } break;
  case BTOR2_TAG_zero: {
    auto opType = builder.getIntegerType(line->sort.bitvec.width);
    res = builder.create<btor::ConstantOp>(unknownLoc, opType,
                                           builder.getIntegerAttr(opType, 0));
  } break;
  case BTOR2_TAG_bad:
    res = builder.create<btor::AssertNotOp>(unknownLoc, cache.at(kids[0]));
    break;

  // indexed ops
  case BTOR2_TAG_slice: {
    auto operandWidth = reached_lines.at(kids[0])->sort.bitvec.width;
    auto opType = builder.getIntegerType(operandWidth);
    assert(operandWidth > kids[1] && kids[1] >= kids[2]);
    auto resType = builder.getIntegerType(kids[1] - kids[2] + 1);

    auto u = builder.create<btor::ConstantOp>(
        unknownLoc, opType, builder.getIntegerAttr(opType, kids[1]));
    auto l = builder.create<btor::ConstantOp>(
        unknownLoc, opType, builder.getIntegerAttr(opType, kids[2]));
    res = builder.create<btor::SliceOp>(unknownLoc, resType, cache.at(kids[0]),
                                        u->getResult(0), l->getResult(0));
  } break;
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

void cleanupCreatedLines() {
  for (auto it = reached_lines.begin(); it != reached_lines.end(); ++it) {
    if (it->first < 0) {
      Btor2Line *line = it->second;
      delete[] line->args;
      delete line;
    }
  }
}

void createNegateLine(int64_t curAt, Btor2Line *line) {
  Btor2Line *res = new Btor2Line;
  int64_t *args = new int64_t[1];
  args[0] = curAt * -1;

  res->id = curAt;
  res->lineno = curAt;
  res->name = "not";
  res->tag = BTOR2_TAG_not;
  res->sort = line->sort;
  res->init = line->init;
  res->next = line->next;
  res->constant = line->constant;
  res->symbol = line->symbol;
  res->nargs = 1;
  res->args = args;

  reached_lines[curAt] = res;
}

bool isValidChild(Btor2Line * line) {
  auto tag = reached_lines.at(line->id)->tag;
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
///          toOp(*it, builder);
///      }
///
///  We can see that for each next operation in btor2, we will compute all
///  the prerequisite operations before storing the result in our cache
///===----------------------------------------------------------------------===//
void toOp(Btor2Line *line, OpBuilder &builder) {

  if (cache.find(line->id) != cache.end())
    return;

  Operation *res = nullptr;
  std::vector<Btor2Line *> todo;
  todo.push_back(line);
  while (!todo.empty()) {
    auto cur = todo.back();
    uint32_t oldsize = todo.size();
    for (uint32_t i = 0; i < cur->nargs; ++i) {
      if (cur->args[i] > 0 
      && !isValidChild(reached_lines.at(cur->args[i]))) {
        continue;
      }

      if (cache.find(cur->args[i]) == cache.end()) {
        if (cur->args[i] < 0) {
          createNegateLine(cur->args[i], reached_lines.at(cur->args[i] * -1));
          todo.push_back(reached_lines.at(cur->args[i]));
          todo.push_back(reached_lines.at(cur->args[i] * -1));
        } else {
          todo.push_back(reached_lines.at(cur->args[i]));
        }
      }
    }
    if (todo.size() != oldsize) {
      continue;
    }
    if (!isValidChild(cur)) {
      todo.pop_back();
      continue;
    }
    // some operations could have been resolved prior to this
    if (cache.find(cur->id) == cache.end()) {
      res = createMLIR(cur, builder, cur->args);
      assert(res && res->getNumResults() < 2 && res->getResult(0));
      cache[cur->id] = res->getResult(0);
    }
    todo.pop_back();
  }
}

OwningOpRef<FuncOp> buildInitFunction(MLIRContext *context) {
  Location unknownLoc = UnknownLoc::get(context);
  OpBuilder builder(context);

  // collect the return types for our init function
  std::vector<Type> returnTypes(inits.size(), nullptr);
  for (uint32_t i = 0; i < inits.size(); ++i) {
    returnTypes[i] = builder.getIntegerType(inits.at(i)->sort.bitvec.width);
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
    toOp(*it, builder);
  }

  // close with a fitting returnOp
  std::vector<Value> testResults(inits.size(), nullptr);
  for (uint32_t i = 0; i < inits.size(); ++i) {
    testResults[i] = cache.at(inits.at(i)->args[1]);
    assert(testResults[i]);
  }
  ArrayRef<Value> results(testResults);

  builder.create<ReturnOp>(unknownLoc, ValueRange({results}));

  return funcOp;
}

OwningOpRef<FuncOp> buildNextFunction(MLIRContext *context) {
  Location unknownLoc = UnknownLoc::get(context);
  OpBuilder builder(context);

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
    toOp(*it, builder);
  }
  for (auto it = bads.begin(); it != bads.end(); ++it) {
    toOp(*it, builder);
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

  model_file = fopen(input->getBufferIdentifier().str().c_str(), "r");

  if (model_file != NULL) {
    parse_model();
    fclose(model_file);

    // extract relevant inits and nexts
    filterInits();
    filterNexts();

    OwningOpRef<FuncOp> initFunc = buildInitFunction(context);
    if (!initFunc)
      return {};

    OwningOpRef<FuncOp> nextFunc = buildNextFunction(context);
    if (!nextFunc)
      return {};

    owningModule->getBody()->push_front(nextFunc.release());
    owningModule->getBody()->push_front(initFunc.release());

    btor2parser_delete(model);
    cleanupCreatedLines();
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
