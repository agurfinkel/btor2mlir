get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSmt
        )
add_llvm_executable(smt2mlir-opt smt2mlir-opt.cpp)

llvm_update_compile_flags(smt2mlir-opt)
target_link_libraries(smt2mlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(smt2mlir-opt)
