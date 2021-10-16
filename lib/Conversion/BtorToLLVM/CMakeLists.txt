add_mlir_conversion_library(MLIRBtorToLLVM
    BtorToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/Conversion/BtorToLLVM

    DEPENDS
    BTORConversionPassIncGen
    intrinsics_gen
    
    LINK_COMPONENTS
    Core
    
    LINK_LIBS PUBLIC
    MLIRLLVMCommonConversion
    MLIRLLVMIR
    MLIRTransforms
    )