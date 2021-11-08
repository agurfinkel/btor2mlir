#include "Target/Btor/BtorToBtorIRTranslation.h"
#include "Dialect/Btor/IR/BtorDialect.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Translation.h"
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

static void
parse_model_line (Btor2Line *l)
{
    reached_lines.push_back (l);
  switch (l->tag)
  {
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
      switch (l->sort.tag)
      {
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

static void
parse_model ()
{
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

static OwningModuleRef deserializeModule(const llvm::MemoryBuffer *input,
                                         MLIRContext *context) {
    context->loadDialect<btor::BtorDialect>();
    
    // need to figure out how to get file name
    // from llvm::MemoryBuffer
    model_file = fopen ("test/Btor/count4.btor2","r");
    
    if (model_file != NULL) {
        std::cout << "reading file ... \n";
        parse_model();
        fclose (model_file);
    }

    // verify correct parsing and structure for module creation
    std::cout << reached_lines.size() << " reached lines" << "\n";
    for( auto it = reached_lines.begin(); it != reached_lines.end(); ++it ) {
        std::cout << (*it)->name << ": line " << (*it)->lineno << "\n";
        std::cout << "   init: " << (*it)->init << " next: " << (*it)->next << "\n";
        std::cout << "   args: ";
        for( uint32_t i = 0; i < (*it)->nargs; ++i ) {
            std::cout << (*it)->args[i] << " ";
        }
        std::cout << "\n";
    }

    // To be fillied with our btor specific code
    OwningModuleRef module;
    return module;
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
