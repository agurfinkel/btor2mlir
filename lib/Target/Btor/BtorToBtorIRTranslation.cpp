#include "Target/Btor/BtorToBtorIRTranslation.h"
#include "Dialect/Btor/IR/BtorDialect.h"
#include "Dialect/Btor/IR/BtorOps.h"


#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <assert.h>
#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <unordered_map>

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

static std::vector<Btor2Line *> reached_lines;

static void parse_model_line (Btor2Line *l) {
    reached_lines.push_back (l);
    switch (l->tag) {
        case BTOR2_TAG_bad: {
            bads.push_back (l);
            reached_bads.push_back (-1);
            num_unreached_bads++;
        }
        break;

        case BTOR2_TAG_constraint: constraints.push_back (l); break;

        case BTOR2_TAG_init: inits[l->args[0]] = l; break;

        case BTOR2_TAG_input: inputs.push_back (l); break;

        case BTOR2_TAG_next: nexts[l->args[0]] = l; break;

        case BTOR2_TAG_sort: {
            switch (l->sort.tag) {
                case BTOR2_TAG_SORT_bitvec:
                case BTOR2_TAG_SORT_array:
                default:
                break;
            }
        }
        break;

        case BTOR2_TAG_state: states.push_back (l); break;

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

static void parse_model () {
    assert (model_file);
    model = btor2parser_new ();
    if (!btor2parser_read_lines (model, model_file))
    std::cout << "parse error at: " << btor2parser_error (model) << "\n";
    num_format_lines = btor2parser_max_id (model);
    inits.resize (num_format_lines, nullptr);
    nexts.resize (num_format_lines, nullptr);
    Btor2LineIterator it = btor2parser_iter_init (model);
    Btor2Line *line;
    while ((line = btor2parser_iter_next (&it))) parse_model_line (line);

    for (size_t i = 0; i < states.size (); i++)
    {
    Btor2Line *state = states[i];
    if (!nexts[state->id])
    {
        std::cout << "state " << state->id << " without next function\n";
    }
    }
}

void filterInits() {
    std::vector<Btor2Line *> filteredInits;
    for( auto it = inits.begin(); it != inits.end(); ++it ) {
        if( *it ) {
            filteredInits.push_back( *it );
        }
    }
    inits.clear();
    inits = filteredInits;
}

void filterNexts() {
    std::vector<Btor2Line *> filteredNexts;
    for( auto it = nexts.begin(); it != nexts.end(); ++it ) {
        if( *it ) {
            filteredNexts.push_back( *it );
        }
    }
    nexts.clear();
    nexts = filteredNexts;
}


Operation * createMLIR( Btor2Line * line, 
                    OpBuilder builder,
                    std::vector<Value> cache,
                    int64_t * kids  ) {
    Location unknownLoc = UnknownLoc::get(builder.getContext());
    Operation * res = nullptr;

    switch (line->tag) {
        case BTOR2_TAG_bad: {
            res = builder.create<btor::AssertOp>(unknownLoc, 
                                        cache.at(kids[0]));
        }
        break;
        case BTOR2_TAG_constraint: break;
        case BTOR2_TAG_init: break;
        case BTOR2_TAG_input: break;
        case BTOR2_TAG_next: break;
        case BTOR2_TAG_sort: break;
        case BTOR2_TAG_state: break;
        case BTOR2_TAG_fair: break;
        case BTOR2_TAG_justice: break;
        case BTOR2_TAG_output: break;

        // binary ops
        case BTOR2_TAG_add: 
            res = builder.create<btor::AddOp>(unknownLoc, 
                                        cache.at(kids[0]), 
                                        cache.at(kids[1]));
        break;
        case BTOR2_TAG_and:
             res = builder.create<btor::AndOp>(unknownLoc, 
                                        cache.at(kids[0]), 
                                        cache.at(kids[1]));
        break;
        case BTOR2_TAG_concat: break;
        case BTOR2_TAG_eq: 
            res = builder.create<btor::CmpOp>(unknownLoc, 
                                        btor::BtorPredicate::eq,
                                        cache.at(kids[0]), 
                                        cache.at(kids[1]));
        break;
        case BTOR2_TAG_implies: break;
        case BTOR2_TAG_nand: break;
        case BTOR2_TAG_nor: break;
        case BTOR2_TAG_neq: break;
        case BTOR2_TAG_or: break;
        case BTOR2_TAG_redand: break;
        case BTOR2_TAG_redor: break;
        case BTOR2_TAG_redxor: break;
        case BTOR2_TAG_sdiv: break;
        case BTOR2_TAG_sgt: break;
        case BTOR2_TAG_sgte: break;
        case BTOR2_TAG_sll: break;
        case BTOR2_TAG_slt: break;
        case BTOR2_TAG_slte: break;
        case BTOR2_TAG_sra: break;
        case BTOR2_TAG_srem: break;
        case BTOR2_TAG_srl: break;
        case BTOR2_TAG_sub: break;
        case BTOR2_TAG_udiv: break;
        case BTOR2_TAG_ugt: break;
        case BTOR2_TAG_ugte: break;
        case BTOR2_TAG_ult: break;
        case BTOR2_TAG_ulte: break;
        case BTOR2_TAG_urem: break;
        case BTOR2_TAG_xnor: break;
        case BTOR2_TAG_xor: break;
        case BTOR2_TAG_rol: break;
        case BTOR2_TAG_ror: break;
        case BTOR2_TAG_saddo: break;
        case BTOR2_TAG_sdivo: break;
        case BTOR2_TAG_smod: break;
        case BTOR2_TAG_smulo: break;
        case BTOR2_TAG_ssubo: break;
        case BTOR2_TAG_uaddo: break;
        case BTOR2_TAG_umulo: break;
        case BTOR2_TAG_usubo: break;
        case BTOR2_TAG_mul:
             res = builder.create<btor::MulOp>(unknownLoc, 
                                            cache.at(kids[0]), 
                                            cache.at(kids[1]));
        break;

        // unary ops
        case BTOR2_TAG_const: break;
        case BTOR2_TAG_constd: break;
        case BTOR2_TAG_consth: break;
        case BTOR2_TAG_dec: 
            res = builder.create<btor::DecOp>(unknownLoc, 
                                            cache.at(kids[0]));
        break;
        case BTOR2_TAG_inc: 
            res = builder.create<btor::IncOp>(unknownLoc, 
                                            cache.at(kids[0]));
        break;
        case BTOR2_TAG_neg: break;
        case BTOR2_TAG_not: break;
        case BTOR2_TAG_one: {
            auto opType = builder.getIntegerType( line->sort.bitvec.width );
            res = builder.create<btor::ConstantOp>(unknownLoc, opType, 
                                                builder.getIntegerAttr(opType, 1));
        } 
        break;
        case BTOR2_TAG_ones: {
            auto width = line->sort.bitvec.width;
            auto opType = builder.getIntegerType( width );
            auto value = pow(2, width) - 1;
            res = builder.create<btor::ConstantOp>(unknownLoc, opType, 
                                                builder.getIntegerAttr(opType, value));
        }
        break;


        // indexed ops
        case BTOR2_TAG_slice: break;
        case BTOR2_TAG_sext: break;
        case BTOR2_TAG_uext: break;
        
        case BTOR2_TAG_zero: {
             auto opType = builder.getIntegerType( line->sort.bitvec.width );
             res = builder.create<btor::ConstantOp>(unknownLoc, opType, 
                                                builder.getIntegerAttr(opType, 0));
        }
        break;
        case BTOR2_TAG_read: break;

        // ternary ops
        case BTOR2_TAG_ite: break;
        case BTOR2_TAG_write: break;

        default:
        break;
    }
    return res;
}

bool isValidChild( uint32_t line ) {
    auto tag = reached_lines.at( line )->tag;
    if( tag == BTOR2_TAG_init 
    ||  tag == BTOR2_TAG_next 
    ||  tag == BTOR2_TAG_state
    ||  tag == BTOR2_TAG_sort ) {
        return false;
    }
    return true;
}

void toOp( Btor2Line * line, 
           OpBuilder builder, 
           std::vector<Value> &cache) {
    if( cache[ line->id ] != nullptr ) 
        return;

    Operation * res = nullptr; 
    std::vector<Btor2Line *> todo;
    todo.push_back( line );
    while( !todo.empty() ) {
        auto cur = todo.back();
        uint32_t oldsize = todo.size();
        for( uint32_t i = 0; i < cur->nargs; ++i ) {
            if( !isValidChild( cur->args[i] ) ) {
                continue;
            }

            if( cache[ cur->args[i] ] == nullptr ) {
                todo.push_back( reached_lines.at( cur->args[i] ) );
            }
        } 
        if( todo.size() != oldsize ) {
            continue;
        }
        if( !isValidChild( cur->id ) ) {
            todo.pop_back();
            continue;
        }
        res = createMLIR( cur, builder, cache, cur->args );
        assert( res );
        cache[ cur->id ] = res->getResult(0);
        todo.pop_back();
    }
}

OwningOpRef<FuncOp> buildInitFunction(MLIRContext *context) {
    Location unknownLoc = UnknownLoc::get(context);
    OpBuilder builder(context);

    // collect the return types for our init function
    std::vector<Type> returnTypes( inits.size(), nullptr); 
    for( uint32_t i = 0; i < inits.size(); ++i ) {
        returnTypes[i] = builder.getIntegerType( 
                                inits.at( i )->sort.bitvec.width );

        assert( returnTypes[i] );
    }
    ArrayRef<Type> outputs( returnTypes );
    
    // create init function signature
    OperationState state(unknownLoc, FuncOp::getOperationName());
    FuncOp::build(builder, state, "init",
                    FunctionType::get( context, {}, outputs ));
    OwningOpRef<FuncOp> funcOp = cast<FuncOp>( Operation::create(state) );

    // create basic block and accompanying builder
    Region &region = funcOp->getBody();
    OpBuilder::InsertionGuard guard( builder );
    auto *body = builder.createBlock( &region );
    builder.setInsertionPointToStart( body );

    std::vector<Value> cache( reached_lines.size(), nullptr );
    for( auto it = inits.begin(); it != inits.end(); ++it ) {
        toOp( *it, builder, cache );
    }

    // close with a fitting returnOp
    std::vector<Value> testResults( inits.size(), nullptr); 
    for( uint32_t i = 0; i < inits.size(); ++i ) {
        testResults[i] = cache.at( inits.at( i )->args[1] );
        assert( testResults[i] );
    }
    ArrayRef<Value> results( testResults );

    builder.create<ReturnOp>( unknownLoc, ValueRange({ results }) );

    return funcOp;
}

OwningOpRef<FuncOp> buildNextFunction(MLIRContext *context) {
    Location unknownLoc = UnknownLoc::get(context);
    OpBuilder builder(context);

    // collect the return types for our init function
    std::vector<Type> returnTypes( nexts.size(), nullptr); 
    for( uint32_t i = 0; i < nexts.size(); ++i ) {
        returnTypes[i] = builder.getIntegerType( 
                                nexts.at( i )->sort.bitvec.width );

        assert( returnTypes[i] );
    }
    ArrayRef<Type> outputs( returnTypes );

    // create next function signature
    OperationState state(unknownLoc, FuncOp::getOperationName());
    FuncOp::build(builder, state, "next",
                    FunctionType::get( context, outputs, outputs ));
    OwningOpRef<FuncOp> funcOp = cast<FuncOp>( Operation::create(state) );
    Region &region = funcOp->getBody();
    OpBuilder::InsertionGuard guard( builder );
    auto *body = builder.createBlock( &region, {}, TypeRange({ outputs }) );
    builder.setInsertionPointToStart( body );

    std::vector<Value> cache( reached_lines.size(), nullptr );
    // initialize states with block arguments
    for( uint32_t i = 0; i < nexts.size(); ++i ) {
        cache[ nexts.at( i )->args[ 0 ] ] = body->getArguments()[i];
    }

    // start with nexts, then add bads, for logic sharing
    for( auto it = nexts.begin(); it != nexts.end(); ++it ) {
        toOp( *it, builder, cache );
    }
    for( auto it = bads.begin(); it != bads.end(); ++it ) {
        toOp( *it, builder, cache );
    }
    
    // close with a fitting returnOp
    std::vector<Value> testResults( nexts.size(), nullptr); 
    for( uint32_t i = 0; i < nexts.size(); ++i ) {
        testResults[i] = cache.at( nexts.at( i )->args[1] );
        assert( testResults[i] );
    }
    ArrayRef<Value> results( testResults );

    builder.create<ReturnOp>( unknownLoc, ValueRange({ results }) );
    return funcOp;
}


static OwningModuleRef deserializeModule(const llvm::MemoryBuffer *input,
                                         MLIRContext *context) {
    context->loadDialect<btor::BtorDialect>();
    context->loadDialect<StandardOpsDialect>();
    
    model_file = fopen( input->getBufferIdentifier().str().c_str(), "r" );
    
    if (model_file != NULL) {
        parse_model();
        fclose (model_file);
    }

    // extract relevant inits and nexts
    filterInits();
    filterNexts();

    // ensure all operations can be accessed by id
    auto iterator = reached_lines.begin();
    reached_lines.insert ( iterator , reached_lines.front() );

    // // verify correct parsing and structure for module creation
    // std::cout << reached_lines.size() << " reached lines" << "\n";
    // for( auto it = reached_lines.begin(); it != reached_lines.end(); ++it ) {
    //     std::cout << (*it)->name << ": line " << (*it)->lineno << "\n";
    //     std::cout << "   sort " << (*it)->sort.bitvec.width << "\n";
    //     std::cout << "   init: " << (*it)->init << " next: " << (*it)->next << "\n";
    //     std::cout << "   args: ";
    //     for( uint32_t i = 0; i < (*it)->nargs; ++i ) {
    //         std::cout << (*it)->args[i] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // std::cout << inits.size() << " init lines" << "\n";
    // std::cout << nexts.size() << " next lines" << "\n";
    // std::cout << states.size() << " states lines" << "\n";
    
    OwningOpRef<FuncOp> initFunc = buildInitFunction(context);
    if (!initFunc)
        return {};

    OwningOpRef<FuncOp> nextFunc = buildNextFunction(context);
    if (!nextFunc)
        return {};

    OwningModuleRef owningModule(ModuleOp::create(FileLineColLoc::get(
      context, input->getBufferIdentifier(), /*line=*/0, /*column=*/0)));
    owningModule->getBody()->push_front(nextFunc.release());
    owningModule->getBody()->push_front(initFunc.release());


    return owningModule;
}

namespace mlir {
namespace btor {
void registerFromBtorTranslation() {
    TranslateToMLIRRegistration fromBtor(
        "import-btor",
        [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
            assert(sourceMgr.getNumBuffers() == 1 && "expected one buffer");
            return deserializeModule(
                sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), context);
        });
    }
} // namespace btor
} // namespace mlir
