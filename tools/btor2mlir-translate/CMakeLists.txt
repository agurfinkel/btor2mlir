set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(btor2mlir-translate
  btor2mlir-translate.cpp
  )
llvm_update_compile_flags(btor2mlir-translate)
target_link_libraries(btor2mlir-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRV
  MLIRTranslation
  MLIRSupport
  )

mlir_check_link_libraries(btor2mlir-translate)
