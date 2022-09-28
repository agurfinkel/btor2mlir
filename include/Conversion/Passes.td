#ifndef BTOR_CONVERSION_PASSES
#define BTOR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// BtorToArithmetic
//===----------------------------------------------------------------------===//

def ConvertBtorToArithmetic : Pass<"convert-btor-to-math", "ModuleOp"> {
  let summary = "Convert Btor Dialect to Arithmetic dialect";
  let description = [{
    Convert btor operations into the Arithmetic dialect operations.

    If other operations are present and their results are required by the Arithmetic 
    dialect operations, the pass will fail.  Any Arithmetic operations or types
    already present in the IR will be kept as is.
  }];
  let constructor = "mlir::btor::createLowerToMathPass()";
  let dependentDialects = ["ArithmeticDialect"];
}

//===----------------------------------------------------------------------===//
// BtorToLLVM
//===----------------------------------------------------------------------===//

def ConvertBtorToLLVM : Pass<"convert-btor-to-llvm", "ModuleOp"> {
  let summary = "Convert Btor Dialect to LLVM dialect";
  let description = [{
    Convert btor operations into LLVM instructions
  }];
  let constructor = "mlir::btor::createLowerToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // BTOR_CONVERSION_PASSES
