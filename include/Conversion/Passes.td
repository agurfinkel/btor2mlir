#ifndef BTOR_CONVERSION_PASSES
#define BTOR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// BtorToStandard
//===----------------------------------------------------------------------===//

def ConvertBtorToStandard : Pass<"convert-btor-to-std", "ModuleOp"> {
  let summary = "Convert operations from the Btor to the Standard dialect";
  let description = [{
    Convert btor operations into the Standard dialect operations.

    If other operations are present and their results are required by the Standard 
    dialect operations, the pass will fail.  Any Standard operations or types
    already present in the IR will be kept as is.
  }];
  let constructor = "mlir::btor::createLowerToStandardPass()";
  let dependentDialects = ["StandardOpsDialect"];
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
