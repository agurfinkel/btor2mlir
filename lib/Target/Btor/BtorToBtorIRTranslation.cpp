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

namespace {

struct Btor2Operation {
    Btor2Line * line;
    Operation * init, next;
};

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

} // end namespace

// assume an array exists that 
// tells us if a line has been read
std::unordered_map<int, std::unique_ptr<OperationState>> mlirOps;


Operation * createMLIR( Btor2Line * line, OpBuilder builder, SmallVector<Value> kids ) {
    // auto lineno = line->lineno;
    // mlirOps.find( (*it)->lineno ) != mlirOps.end()
    std::cout << line->id << " op has " << kids.size() << " kids \n";
    Location unknownLoc = UnknownLoc::get(builder.getContext());

    if( line->tag == BTOR2_TAG_one ) {
        auto opType = builder.getIntegerType( line->sort.bitvec.width );
        Operation * constOp = builder.create<btor::ConstantOp>(unknownLoc, opType, 
                                                    builder.getIntegerAttr(opType, 1));
        std::cout << "  created constant op for line: " << line->id << "\n";
        return constOp;
    } else if( line->tag == BTOR2_TAG_zero ) {
        auto opType = builder.getIntegerType( line->sort.bitvec.width );
        Operation * constOp = builder.create<btor::ConstantOp>(unknownLoc, opType, 
                                                    builder.getIntegerAttr(opType, 0));
        std::cout << "  created constant op for line: " << line->id << "\n";
        return constOp;
    } else if( line->tag == BTOR2_TAG_add ) {
        Operation * addOp = builder.create<btor::AddOp>(unknownLoc, 
                                                    kids[0], 
                                                    kids[1]);
        std::cout << "  created add op for line: " << line->id << "\n";
        return addOp;
    }
    return nullptr;
}

bool isValidChild( uint32_t line ) {
    auto tag = reached_lines.at( line )->tag;
    if( tag == BTOR2_TAG_init 
    ||  tag == BTOR2_TAG_next 
    ||  tag == BTOR2_TAG_state
    ||  tag == BTOR2_TAG_sort ) {
        std::cout << "   false for tag value = " << tag << "\n";
        return false;
    }
    std::cout << "   true for tag value = " << tag << "\n";
    return true;
}

void toOp( Btor2Line * line, 
           OpBuilder builder, 
           std::vector<Value> &cache) {
    std::cout << "toOp called with id: " << line->id << "\n";
    if( cache[ line->id ] != nullptr ) { 
        // return cache.at( btorLine.id )
        std::cout << "inserting: " << line->id << "\n";
        // builder.insert( cache[ line->id ] );
    } else {
        Operation * res = nullptr; 
        std::vector<Btor2Line *> todo;
        todo.push_back( line );
        while( !todo.empty() ) {
            auto cur = todo.back();
            uint32_t oldsize = todo.size();
            // mlirOp[] kids; // use smallvector
            std::cout << "working on: " << cur->id << "\n";
            SmallVector<Value> kids;
            for( uint32_t i = 0; i < cur->nargs; ++i ) {
                if( !isValidChild( cur->args[i] ) ) {
                    std::cout << "    ignore line " << cur->args[i] << "\n";
                    continue;
                }

                if( cache[ cur->args[i] ] ) {
                    std::cout << "   add to kids line " << cur->args[i] << "\n";
                    kids.push_back( cache.at( cur->args[i] ) );
                } else {
                    std::cout << "   add to todo line " << cur->args[i] << "\n";
                    todo.push_back( reached_lines.at( cur->args[i] ) );
                }
            } 
            std::cout << "  kids ready for: " << cur->id << "\n";
            std::cout << std::flush;
            if( todo.size() != oldsize ) {
                std::cout << " reset kids for line " << cur->id << "\n";
                continue;
            }
            if( !isValidChild( cur->id ) ) {
                std::cout << " invalid line " << cur->id << "\n";
                todo.pop_back();
                continue;
            }
            res = createMLIR( cur, builder, kids );
            assert( res );
            cache[ cur->id ] = res->getResult(0);
            std::cout << " finished line " << cur->id << "\n";
            todo.pop_back();
        }
        // // assert( res )
        // // return res
    }
}

OwningOpRef<FuncOp> buildInitFunction(MLIRContext *context) {
    Location unknownLoc = UnknownLoc::get(context);
    OpBuilder builder(context);

    // collect the return types for our init function
    std::vector<Type> returnTypes( inits.size(), nullptr); 
    for( uint32_t i = 0; i < inits.size(); ++i ) {
        std::cout << "get type for line " << inits.at( i )->id << "\n";
        returnTypes[i] = builder.getIntegerType( 
                                inits.at( i )->sort.bitvec.width );

        assert( returnTypes[i] );
        std::cout << "added result for line " << inits.at( i )->id << "\n";
    }
    ArrayRef<Type> outputs( returnTypes );
    
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

    std::vector<Value> cache( reached_lines.size(), nullptr );
    // std::cout << "cache has size: " << cache.size() << "\n";
    for( auto it = inits.begin(); it != inits.end(); ++it ) {
        toOp( *it, builder, cache );
    }

    // close with a fitting returnOp
    // std::cout << "\n" << "create return op " << "\n";
    std::vector<Value> testResults( inits.size(), nullptr); 
    for( uint32_t i = 0; i < inits.size(); ++i ) {
        // std::cout << "get result for line " << inits.at( i )->args[1] << "\n";
        testResults[i] = cache.at( inits.at( i )->args[1] );
        assert( testResults[i] );
        // std::cout << "added result for line " << inits.at( i )->args[1] << "\n";
    }
    // std::cout << "itereate return value " << "\n";
    ArrayRef<Value> results( testResults );

    builder.create<ReturnOp>(unknownLoc, ValueRange({ results }));
    // std::cout << "done" << "\n";
    return funcOp;
}

OwningOpRef<FuncOp> buildNextFunction(MLIRContext *context) {
    Location unknownLoc = UnknownLoc::get(context);
    OpBuilder builder(context);

    // collect the return types for our init function
    std::cout << "\nNext Function: \n";
    std::vector<Type> returnTypes( nexts.size(), nullptr); 
    for( uint32_t i = 0; i < nexts.size(); ++i ) {
        std::cout << "get type for line " << nexts.at( i )->id << "\n";
        returnTypes[i] = builder.getIntegerType( 
                                nexts.at( i )->sort.bitvec.width );

        assert( returnTypes[i] );
        std::cout << "added result for line " << nexts.at( i )->id << "\n";
    }
    ArrayRef<Type> outputs( returnTypes );

    OperationState state(unknownLoc, FuncOp::getOperationName());
    FuncOp::build(builder, state, "next",
                    FunctionType::get(context, outputs, { })
                    /*, builder.getStringAttr("private")*/);
    OwningOpRef<FuncOp> funcOp = cast<FuncOp>(Operation::create(state));
    Region &region = funcOp->getBody();
    OpBuilder::InsertionGuard guard( builder );
    auto *body = builder.createBlock( &region, {}, TypeRange({ outputs }) );
    builder.setInsertionPointToStart( body );

    std::vector<Value> cache( reached_lines.size(), nullptr );
    // initialize states with block arguments
    for( uint32_t i = 0; i < nexts.size(); ++i ) {
        cache[ nexts.at( i )->args[ 0 ] ] = body->getArguments()[i];
    }

    // std::cout << "cache has size: " << cache.size() << "\n";
    for( auto it = nexts.begin(); it != nexts.end(); ++it ) {
        toOp( *it, builder, cache );
    }

    // auto opType = builder.getIntegerType( 4 );
    // Operation * constOp = builder.create<btor::ConstantOp>(unknownLoc, opType, 
    //                                                 builder.getIntegerAttr(opType, 0));
    // Value test = nullptr;
    // builder.create<btor::AddOp>( unknownLoc, 
    //                     body->getArguments()[0], 
    //                     constOp->getResult(0));

    // returnOp
    OperationState returnState(unknownLoc, ReturnOp::getOperationName());
    ReturnOp::build(builder, returnState);
    auto retOp = cast<ReturnOp>(Operation::create(returnState));
    builder.insert( retOp );
    
    return funcOp;
}


static OwningModuleRef deserializeModule(const llvm::MemoryBuffer *input,
                                         MLIRContext *context) {
    context->loadDialect<btor::BtorDialect>();
    context->loadDialect<StandardOpsDialect>();
    
    // need to figure out how to get file name
    // from llvm::MemoryBuffer
    model_file = fopen ("test/Btor/count4.btor2","r");
    
    if (model_file != NULL) {
        std::cout << "reading file ... \n";
        parse_model();
        fclose (model_file);
    }

    // extract relevant inits and nexts
    filterInits();
    filterNexts();

    // ensure all operations can be accessed by id
    auto iterator = reached_lines.begin();
    reached_lines.insert ( iterator , reached_lines.front() );

    // verify correct parsing and structure for module creation
    std::cout << reached_lines.size() << " reached lines" << "\n";
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

    std::cout << inits.size() << " init lines" << "\n";
    std::cout << nexts.size() << " next lines" << "\n";
    std::cout << states.size() << " states lines" << "\n";
    
    // To be fillied with our btor specific code
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
