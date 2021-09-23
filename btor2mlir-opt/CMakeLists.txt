get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRBtor
        )
add_llvm_executable(btor2mlir-opt btor2mlir-opt.cpp)

llvm_update_compile_flags(btor2mlir-opt)
target_link_libraries(btor2mlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(btor2mlir-opt)
