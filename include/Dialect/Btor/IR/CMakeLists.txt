set(LLVM_TARGET_DEFINITIONS BtorOps.td)
mlir_tablegen(BtorOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(BtorOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect(BtorOps btor)

set(LLVM_TARGET_DEFINITIONS BtorTypes.td)
mlir_tablegen(BtorTypes.h.inc -gen-typedef-decls -typedefs-dialect=btor)
mlir_tablegen(BtorTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=btor)
add_public_tablegen_target(MLIRBtorTypesIncGen)

add_mlir_doc(BtorOps BtorOps Dialects/ -gen-op-doc)