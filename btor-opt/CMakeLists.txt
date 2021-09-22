get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRBtor
        )
add_llvm_executable(btor-opt btor-opt.cpp)

llvm_update_compile_flags(btor-opt)
target_link_libraries(btor-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(btor-opt)
