#include "Conversion/BtorToMemRef/ConvertBtorToMemRefPass.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "../PassDetail.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::btor;

#define PASS_NAME "convert-btor-to-memref"

namespace {
struct BtorToMemRefLoweringPass : public PassWrapper<BtorToMemRefLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<memref::MemRefDialect>();
    }
    StringRef getArgument() const final { return PASS_NAME; }
    void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//


} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Lowering Definitions
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToMemRefConversionPatterns(RewritePatternSet &patterns) {
//   patterns.add<AddLowering, MulLowering, AndLowering>(patterns.getContext());
}

void BtorToMemRefLoweringPass::runOnOperation() {
    RewritePatternSet patterns(&getContext());
    populateBtorToMemRefConversionPatterns(patterns);
    ConversionTarget target(getContext());
    // target.addIllegalOp<mlir::btor::AddOp, mlir::btor::MulOp, mlir::btor::AndOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::btor::createLowerToMemRefPass() {
    return std::make_unique<BtorToMemRefLoweringPass>(); 
}
