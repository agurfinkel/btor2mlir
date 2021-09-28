set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BtorConversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix BtorConversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix BtorConversion)
add_public_tablegen_target(BTORConversionPassIncGen)

add_mlir_doc(Passes BtorConversionPasses ./ -gen-pass-doc)