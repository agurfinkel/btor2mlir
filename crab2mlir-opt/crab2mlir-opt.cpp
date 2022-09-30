//===- crab2mlir-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Dialect/Crab/IR/Crab.h"

using namespace mlir;
using namespace llvm;

namespace {
    static cl::opt<std::string> inputFilename(
        cl::Positional, cl::desc("<input file>"), cl::init("-"));

    static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                                cl::value_desc("filename"),
                                                cl::init("-"));
}

static OwningOpRef<ModuleOp> processCrabBuffer(raw_ostream &os, 
                                std::unique_ptr<MemoryBuffer> ownedBuffer,
                                DialectRegistry &registry) {
    // Tell sourceMgr about this buffer; parser will pick this up
    SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
    // New context for our buffer
    MLIRContext context(registry);
    SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);

    OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, &context));
    if (module) {
        module->print(os);
        os << '\n';
    }

    return module;
}

int main(int argc, char **argv) {
    // Register our used dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::crab::CrabDialect>();
    registry.insert<arith::ArithmeticDialect,
                    func::FuncDialect,
                    cf::ControlFlowDialect>();

    // Set up needed tools
    InitLLVM y(argc, argv);
    cl::ParseCommandLineOptions(argc, argv);

    // Set up the input file.
    std::string errorMessage;
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
        llvm::errs() << errorMessage << "\n";
        return EXIT_FAILURE;
    }

    auto output = openOutputFile(outputFilename, &errorMessage);
    if (!output) {
        llvm::errs() << errorMessage << "\n";
        return EXIT_FAILURE;
    }

    OwningOpRef<ModuleOp> module = processCrabBuffer(output->os(), std::move(file), registry);
    assert(module);

    return EXIT_SUCCESS;
}
