#include "Conversion/BtorToStandard/ConvertBtorToStandardPass.h"
#include "Dialect/Btor/IR/BtorOps.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::btor;

#define PASS_NAME "convert-btor-to-std"

namespace {
struct BtorToStandardLoweringPass : public PassWrapper<BtorToStandardLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<StandardOpsDialect>();
    }
    StringRef getArgument() const final { return PASS_NAME; }
    void runOnOperation() override;
};
} // end anonymous namespace

void BtorToStandardLoweringPass::runOnOperation() {}

/// Create a pass for lowering operations the remaining `Btor` operations
// to the Standard dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToStandardPass() {
    return std::make_unique<BtorToStandardLoweringPass>(); 
}

void mlir::btor::registerBtorToStandardPass() {
    PassRegistration<BtorToStandardLoweringPass>();
} 