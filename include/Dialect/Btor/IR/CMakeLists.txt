set(LLVM_TARGET_DEFINITIONS BtorOps.td)
mlir_tablegen(BtorOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(BtorOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect(BtorOps btor)

add_mlir_doc(BtorOps BtorOps Dialects/ -gen-op-doc)