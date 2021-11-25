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

void Deserialize::parseModelLine(Btor2Line *l) {
  setLineWithId(l->id, l);
  switch (l->tag) {
  case BTOR2_TAG_bad:
    m_bads.push_back(l);
    break;

  case BTOR2_TAG_constraint:
    m_constraints.push_back(l);
    break;

  case BTOR2_TAG_init:
    m_inits.push_back(l);
    break;

  case BTOR2_TAG_next:
    m_nexts.push_back(l);
    break;

  case BTOR2_TAG_state:
    m_states.push_back(l);
    break;

  default:
    break;
  }
}

bool Deserialize::parseModelIsSuccessful() {
  if (!m_modelFile)
    return false;
  m_model = btor2parser_new();
  if (!btor2parser_read_lines(m_model, m_modelFile)) {
    std::cerr << "parse error at: " << btor2parser_error(m_model) << "\n";
    return false;
  }
  // register each line that has been parsed
  auto numLines = btor2parser_max_id (m_model);
  m_lines.resize(numLines + 1, nullptr);
  Btor2LineIterator it = btor2parser_iter_init(m_model);
  Btor2Line *line;
  while ((line = btor2parser_iter_next(&it))) {
    parseModelLine(line);
  }
  // ensure each state has a next function
  for (auto state : m_states) {
    if (!getLineById(state->next)) {
      std::cerr << "state " << state->id << " without next function\n";
      return false;
    }
  }

  return true;
}

///===----------------------------------------------------------------------===//
/// This function's goal is to create the MLIR Operation that corresponds to 
/// the given Btor2Line*, cur, into the basic block designated by the class 
/// field m_builder. Make sure that the kids have already been created before 
/// calling this method
///
/// e.x:
///     Operation * res = createMLIR(cur, cur->args);
///
///===----------------------------------------------------------------------===//
Operation * Deserialize::createMLIR(const Btor2Line *line, 
                                const SmallVector<Value> &kids,
                                const SmallVector<unsigned> &arguments) {
  Operation *res = nullptr;

  switch (line->tag) {
  // binary ops
  case BTOR2_TAG_slt:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::slt, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_slte:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::sle, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_sgt:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::sgt, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_sgte:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::sge, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_neq:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::ne, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_eq:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::eq, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_ugt:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::ugt, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_ugte:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::uge, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_ult:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::ult, 
                                        kids[0], kids[1]);
    break;
  case BTOR2_TAG_ulte:
    res = buildComparisonOp<btor::CmpOp>(btor::BtorPredicate::ule, 
                                        kids[0], kids[1]);
    break;
  
  case BTOR2_TAG_concat:
    res = buildConcatOp(kids[0], kids[1]);
    break;
  case BTOR2_TAG_add:
    res = buildBinaryOp<btor::AddOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_and:
    res = buildBinaryOp<btor::AndOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_implies:
    res = buildBinaryOp<btor::ImpliesOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_iff:
    res = buildBinaryOp<btor::IffOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_nand:
    res = buildBinaryOp<btor::NandOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_nor:
    res = buildBinaryOp<btor::NorOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_or:
    res = buildBinaryOp<btor::OrOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_sdiv:
    res = buildBinaryOp<btor::SDivOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_srem:
    res = buildBinaryOp<btor::SRemOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_sub:
    res = buildBinaryOp<btor::SubOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_udiv:
    res = buildBinaryOp<btor::UDivOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_urem:
    res = buildBinaryOp<btor::URemOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_mul:
    res = buildBinaryOp<btor::MulOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_smod:
    res = buildBinaryOp<btor::SModOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_xnor:
    res = buildBinaryOp<btor::XnorOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_xor:
    res = buildBinaryOp<btor::XOrOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_sll:
    res = buildBinaryOp<btor::ShiftLLOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_sra:
    res = buildBinaryOp<btor::ShiftRAOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_srl:
    res = buildBinaryOp<btor::ShiftRLOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_rol:
    res = buildBinaryOp<btor::RotateLOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_ror:
    res = buildBinaryOp<btor::RotateROp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_saddo:
    res = buildOverflowOp<btor::SAddOverflowOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_sdivo:
    res = buildOverflowOp<btor::SDivOverflowOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_smulo:
    res = buildOverflowOp<btor::SMulOverflowOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_ssubo:
    res = buildOverflowOp<btor::SSubOverflowOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_uaddo:
    res = buildOverflowOp<btor::UAddOverflowOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_umulo:
    res = buildOverflowOp<btor::UMulOverflowOp>(kids[0], kids[1]);
    break;
  case BTOR2_TAG_usubo:
    res = buildOverflowOp<btor::USubOverflowOp>(kids[0], kids[1]);
    break;

  // unary ops
  case BTOR2_TAG_dec:
    res = buildUnaryOp<btor::DecOp>(kids[0]);
    break;
  case BTOR2_TAG_inc:
    res = buildUnaryOp<btor::IncOp>(kids[0]);
    break;
  case BTOR2_TAG_neg:
    res = buildUnaryOp<btor::NegOp>(kids[0]);
    break;
  case BTOR2_TAG_not:
    res = buildUnaryOp<btor::NotOp>(kids[0]);
    break;
  case BTOR2_TAG_bad:
    res = buildUnaryOp<btor::AssertNotOp>(kids[0]);
    break;
  case BTOR2_TAG_redand:
    res = buildReductionOp<btor::RedAndOp>(kids[0]);
    break;
  case BTOR2_TAG_redor:
    res = buildReductionOp<btor::RedOrOp>(kids[0]);
    break;
  case BTOR2_TAG_redxor:
    res = buildReductionOp<btor::RedXorOp>(kids[0]);
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
  case BTOR2_TAG_input:
    res = buildInputOp(line->sort.bitvec.width);
    break;
  case BTOR2_TAG_constraint:
    res = buildUnaryOp<btor::AssumeOp>(kids[0]);
    break;


  // indexed ops
  case BTOR2_TAG_slice:
    res = buildSliceOp(kids[0], arguments[0], arguments[1]);
    break;
  case BTOR2_TAG_sext:
    res = buildExtOp<btor::SExtOp>(kids[0], line->sort.bitvec.width);
    break;
  case BTOR2_TAG_uext:
    res = buildExtOp<btor::UExtOp>(kids[0], line->sort.bitvec.width);
    break;

  // ternary ops
  case BTOR2_TAG_ite:
    res = buildIteOp(kids[0], kids[1], kids[2]);
    break;

  // unmapped ops
  case BTOR2_TAG_read:
  case BTOR2_TAG_init:
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

///===----------------------------------------------------------------------===//
/// Some Btor lines may refer to a negated version of a prior line. This
/// function creates a negated version of the original line, and stores it in
/// the cache, only after the caller ensures that the original line has been 
/// created and saved in the cache
///===----------------------------------------------------------------------===//
void Deserialize::createNegateLine(int64_t negativeLine, const Value &child) {
  auto res = buildUnaryOp<btor::NotOp>(getFromCacheById(std::abs(negativeLine)));
  assert(res);
  assert(res->getNumResults() == 1);
  setCacheWithId(negativeLine, res->getResult(0));
}

///===----------------------------------------------------------------------===//
/// This method determines if a Btor2Line has a return value
///===----------------------------------------------------------------------===//
bool Deserialize::hasReturnValue(Btor2Line * line) {
  bool hasReturnValue = true;
  switch (line->tag) {
  case BTOR2_TAG_constraint:
  case BTOR2_TAG_bad:
    hasReturnValue = false;
    break;
  default:
    break;
  }
  return hasReturnValue;
}

///===----------------------------------------------------------------------===//
/// We use this method to check if a line needs to have a corresponding MLIR
/// operation created
///===----------------------------------------------------------------------===//
bool Deserialize::needsMLIROp(Btor2Line * line) {
  bool isValid = true;
  switch (line->tag) {
  case BTOR2_TAG_init:
  case BTOR2_TAG_next:
  case BTOR2_TAG_state:
  case BTOR2_TAG_read:
  case BTOR2_TAG_sort:
  case BTOR2_TAG_fair:
  case BTOR2_TAG_justice:
  case BTOR2_TAG_output:
  case BTOR2_TAG_write:
    isValid = false;
    break;
  default:
    break;
  }
  return isValid;
}

///===----------------------------------------------------------------------===//
/// This function's goal is to add the MLIR Operation that corresponds to 
/// the given Btor2Line* into the basic block designated by the class field 
/// m_builder. Then, the MLIR Value of the newly minted operation is added
/// into our cache for future reference within the basic block. 
///
/// e.x:
///      for (next : m_nexts) {
///          toOp(next);
///      }
///
///  We can see that for each next operation in btor2, we will compute all
///  the prerequisite operations before storing the result in our cache
///===----------------------------------------------------------------------===//
void Deserialize::toOp(Btor2Line *line) {
  if (valueAtIdIsInCache(line->id)) {
    return;
  }

  Operation *res = nullptr;
  std::vector<Btor2Line *> todo;
  todo.push_back(line);
  while (!todo.empty()) {
    auto cur = todo.back();
    unsigned oldsize = todo.size();

    for (unsigned i = 0; i < cur->nargs; ++i) {
      auto arg_i = cur->args[i];
      // exit early if we do not need to compute this line
      if (arg_i > 0 && !needsMLIROp(getLineById(arg_i))) {
        continue;
      }
      // only look at uncomputed lines
      if (!valueAtIdIsInCache(arg_i)) {
        if (arg_i < 0) {
          // if original operation is cached, negate it on the fly
          if (valueAtIdIsInCache(std::abs(arg_i))) {
            createNegateLine(arg_i, getFromCacheById(std::abs(arg_i))); 
          } else {
            todo.push_back(getLineById(std::abs(arg_i)));
          }
        } else {
          todo.push_back(getLineById(arg_i));
        }
      }
    }

    if (todo.size() != oldsize) {
      continue;
    }

    if (!needsMLIROp(cur) 
    || valueAtIdIsInCache(cur->id)) {
      todo.pop_back();
      continue;
    }

    SmallVector<Value> kids;
    SmallVector<unsigned> arguments;
    if (cur->tag != BTOR2_TAG_slice) {
      for (unsigned i = 0; i < cur->nargs; ++i) {
          kids.push_back(getFromCacheById(cur->args[i]));
      }
    } else {
        kids.push_back(getFromCacheById(cur->args[0]));
        arguments.push_back(cur->args[1]);
        arguments.push_back(cur->args[2]);
    }
    res = createMLIR(cur, kids, arguments);
    setCacheWithId(cur->id, res);
    todo.pop_back();
  }
}

OwningOpRef<FuncOp> Deserialize::buildInitFunction() {
  // collect the return types for our init function
  std::vector<Type> returnTypes(m_states.size(), nullptr);
  for (unsigned i = 0; i < m_states.size(); ++i) {
    returnTypes[i] = getIntegerTypeOf(m_states.at(i));
    assert(returnTypes[i]);
  }

  // create init function signature
  OperationState state(m_unknownLoc, FuncOp::getOperationName());
  FuncOp::build(m_builder, state, "init",
                FunctionType::get(m_context, {}, returnTypes));
  OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));

  // create basic block with accompanying m_builder
  Region &region = funcOp->getBody();
  OpBuilder::InsertionGuard guard(m_builder);
  auto *body = m_builder.createBlock(&region);
  m_builder.setInsertionPointToStart(body);

  // clear cache so that values are mapped to the right Basic Block
  m_cache.clear();
  for (auto init : m_inits) { toOp(init); }

  // close with a fitting returnOp
  auto results = collectReturnValuesForInit(returnTypes);
  buildReturnOp(results);

  return funcOp;
}

OwningOpRef<FuncOp> Deserialize::buildNextFunction() {
  // collect the return types for our init function
  std::vector<Type> returnTypes(m_states.size(), nullptr);
  for (unsigned i = 0; i < m_states.size(); ++i) {
    returnTypes[i] = getIntegerTypeOf(m_states.at(i));
    assert(returnTypes[i]);
  }

  // create next function signature
  OperationState state(m_unknownLoc, FuncOp::getOperationName());
  FuncOp::build(m_builder, state, "next",
                FunctionType::get(m_context, returnTypes, returnTypes));
  OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));
  Region &region = funcOp->getBody();
  OpBuilder::InsertionGuard guard(m_builder);
  auto *body = m_builder.createBlock(&region, {}, returnTypes);
  m_builder.setInsertionPointToStart(body);

  // clear cache so that values are mapped to the right Basic Block
  m_cache.clear();
  // initialize states with block arguments
  for (unsigned i = 0; i < m_states.size(); ++i) {
    auto stateId = m_states.at(i)->id;
    setCacheWithId(stateId, body->getArguments()[i]);
  }

  // start with nexts, then add bads, for logic sharing
  for (auto next : m_nexts) { toOp(next); }
  for (auto constraint : m_constraints) { toOp(constraint); }
  for (auto bad : m_bads) { toOp(bad); }

  // close with a fitting returnOp
  std::vector<Value> results(m_states.size(), nullptr);
  for (unsigned i = 0; i < m_states.size(); ++i) {
    results[i] = getFromCacheById(m_states.at(i)->next);
  }
  buildReturnOp(results);
  return funcOp;
}

static OwningModuleRef deserializeModule(const llvm::MemoryBuffer *input,
                                         MLIRContext *context) {
  context->loadDialect<btor::BtorDialect>();
  context->loadDialect<StandardOpsDialect>();

  OwningModuleRef owningModule(ModuleOp::create(FileLineColLoc::get(
      context, input->getBufferIdentifier(), /*line=*/0, /*column=*/0)));

  Deserialize deserialize(context, input->getBufferIdentifier().str());
  if (deserialize.parseModelIsSuccessful()) {
    OwningOpRef<FuncOp> initFunc = deserialize.buildInitFunction();
    if (!initFunc)
      return owningModule;

    OwningOpRef<FuncOp> nextFunc = deserialize.buildNextFunction();
    if (!nextFunc)
      return owningModule;

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
