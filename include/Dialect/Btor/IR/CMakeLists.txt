set(LLVM_TARGET_DEFINITIONS BtorOps.td)
add_mlir_dialect(BtorOps btor)

set(LLVM_TARGET_DEFINITIONS BtorAttributes.td)
mlir_tablegen(BtorAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(BtorAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(BtorOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(BtorOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(BtorAttributesIncGen)

add_mlir_doc(BtorAttributes BtorAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(BtorOps BtorOps Dialects/ -gen-op-doc)