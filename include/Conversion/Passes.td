#ifndef BTOR_CONVERSION_PASSES
#define BTOR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// BtorToArithmetic
//===----------------------------------------------------------------------===//

def ConvertBtorToArithmetic : Pass<"convert-btor-to-arith", "ModuleOp"> {
  let summary = "Convert Btor Dialect to Arithmetic dialect";
  let description = [{
    Convert btor operations into the Arithmetic dialect operations.

    If other operations are present and their results are required by the Arithmetic 
    dialect operations, the pass will fail.  Any Arithmetic operations or types
    already present in the IR will be kept as is.
  }];
  let constructor = "btor::createConvertBtorToArithmeticPass()";
  let dependentDialects = ["arith::ArithmeticDialect"];
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

//===----------------------------------------------------------------------===//
// BtorNDToLLVM
//===----------------------------------------------------------------------===//

def ConvertBtorNDToLLVM : Pass<"convert-btornd-to-llvm", "ModuleOp"> {
  let summary = "Convert Btor Dialect nd Operations to LLVM dialect";
  let description = [{
    Convert nd btor operations into LLVM instructions
  }];
  let constructor = "mlir::btor::createLowerBtorNDToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// BtorToVector
//===----------------------------------------------------------------------===//

def ConvertBtorToVector : Pass<"convert-btor-to-vector", "ModuleOp"> {
  let summary = "Convert Btor Dialect to Vector dialect";
  let description = [{
    Convert btor array operations into vector instructions
  }];
  let constructor = "mlir::btor::createLowerToVectorPass()";
  let dependentDialects = ["vector::VectorDialect"];
}

#endif // BTOR_CONVERSION_PASSES
