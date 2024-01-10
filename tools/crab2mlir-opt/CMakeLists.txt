get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRSmt
        )
add_llvm_executable(crab2mlir-opt crab2mlir-opt.cpp)

llvm_update_compile_flags(crab2mlir-opt)
target_link_libraries(crab2mlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(crab2mlir-opt)
