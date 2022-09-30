set(LLVM_TARGET_DEFINITIONS SmtOps.td)
mlir_tablegen(SmtOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SmtOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect(SmtOps smt)

add_mlir_doc(SmtOps SmtOps Dialects/ -gen-op-doc)
