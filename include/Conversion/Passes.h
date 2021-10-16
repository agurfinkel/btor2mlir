#ifndef BTOR_CONVERSION_PASSES_H
#define BTOR_CONVERSION_PASSES_H

#include "Conversion/BtorToStandard/ConvertBtorToStandardPass.h"
#include "Conversion/BtorToLLVM/ConvertBtorToLLVMPass.h"

namespace mlir {
namespace btor {
    /// Generate the code for registering conversion passes.
    #define GEN_PASS_REGISTRATION
    #include "Conversion/Passes.h.inc"
} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_PASSES_H